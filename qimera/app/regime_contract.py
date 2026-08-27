"""QiMera Regime Filter Interface Contract (QiMera.md §6.2 / M1).

Calculates historical trend and volatility regimes per candle using the same
mathematical models as Meruem's RegimeAgent (GARCH conditional volatility and
2-state Markov HMM) but optimized to process the entire history in one pass.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure backend is in Python path to import app services
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

# ``app`` here is Meruem's backend package (backend/app) — both this repo and
# Meruem ship an ``app`` package, so qimera's own modules import as
# ``qimera.app.*`` while Meruem services are imported via the top-level ``app``
# name, which resolves to backend/app when backend/ is ahead of qimera/ on
# sys.path (conftest + the sys.path setup below guarantee that).
from app.services.regime import _normal_pdf, _log_returns


def _garch_vol_all(closes: list[float],
                   alpha_grid=(0.05, 0.10, 0.15, 0.20),
                   beta_grid=(0.80, 0.85, 0.90, 0.95)) -> np.ndarray:
    """Computes GARCH(1,1) vol_ratio for all bars in one pass.
    
    Returns an array of vol_ratios aligned with closes (padded with 1.0 at start).
    """
    n_closes = len(closes)
    out = np.ones(n_closes)
    rets = _log_returns(closes)
    n = len(rets)
    if n < 60:
        return out

    r = rets - rets.mean()
    long_run = float(r.var())
    if long_run <= 0 or not np.isfinite(long_run):
        return out

    best = None
    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha + beta >= 1.0:
                continue
            omega = (1.0 - alpha - beta) * long_run
            s2 = long_run
            loglik = 0.0
            for rt in r:
                s2 = omega + alpha * rt * rt + beta * s2
                loglik += -0.5 * (np.log(max(s2, 1e-12)) + rt * rt / max(s2, 1e-12))
            if best is None or loglik > best[0]:
                best = (loglik, alpha, beta, omega)
                
    if best is None:
        return out

    _, alpha, beta, omega = best
    s2 = long_run
    s2_history = np.zeros(n)
    for i, rt in enumerate(r):
        s2 = omega + alpha * rt * rt + beta * s2
        s2_history[i] = s2

    cond_vols = np.sqrt(np.maximum(s2_history, 1e-12))
    long_vol = np.sqrt(max(long_run, 1e-12))
    
    # Pad the first element of closes (before the first log return) with 1.0
    out[1:] = np.round(cond_vols / long_vol, 3)
    return out


def _markov_regime_all(closes: list[float], iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Computes 2-state Gaussian HMM (CALM vs TURBULENT) high-vol probability for all bars.
    
    Returns (prob_highs, states) aligned with closes.
    """
    n_closes = len(closes)
    prob_highs = np.zeros(n_closes)
    states = np.array(["CALM"] * n_closes, dtype=object)
    
    rets = _log_returns(closes)
    n = len(rets)
    if n < 80:
        return prob_highs, states

    r = rets - rets.mean()
    sd = float(r.std())
    if sd <= 0 or not np.isfinite(sd):
        return prob_highs, states

    scaled = r / sd
    A = np.array([[0.98, 0.02], [0.02, 0.98]], dtype=float)
    mu = np.array([-0.2, 0.2], dtype=float)
    sig = np.array([0.7, 1.5], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)

    for _ in range(iters):
        B = np.clip(_normal_pdf(scaled, mu, sig), 1e-12, None)
        alpha = np.empty((n, 2))
        c = np.empty(n)
        a0 = pi * B[0]
        c0 = a0.sum()
        alpha[0] = a0 / c0
        c[0] = c0
        for t in range(1, n):
            at = (alpha[t - 1] @ A) * B[t]
            ct = at.sum()
            alpha[t] = at / ct
            c[t] = ct
        beta = np.empty((n, 2))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma = np.clip(gamma, 1e-12, None)
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        num = xi = np.zeros((n - 1, 2, 2))
        for t in range(n - 1):
            joint = (alpha[t][:, None] * A) * (B[t + 1][None, :] * beta[t + 1][None, :])
            xi[t] = joint / (joint.sum() + 1e-12)
        num = xi.sum(axis=0)
        den = gamma[:-1].sum(axis=0)
        A = num / (den[:, None] + 1e-12)
        A = np.clip(A, 1e-4, 1.0)
        A /= A.sum(axis=1, keepdims=True)
        mu = (gamma * scaled[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-12)
        var_num = (gamma * (scaled[:, None] - mu[None, :]) ** 2).sum(axis=0)
        sig = np.sqrt(var_num / (gamma.sum(axis=0) + 1e-12))
        sig = np.clip(sig, 1e-4, None)
        pi = gamma[0]

    hi = int(np.argmax(sig))
    lo = 1 - hi
    sig_ratio = float(sig[hi] / sig[lo]) if sig[lo] > 0 else 1.0
    separation = min(1.0, (sig_ratio - 1.0) / 0.5)
    
    # Calculate high-vol probability for all returns
    hi_probs = gamma[:, hi] * separation
    
    prob_highs[1:] = np.round(hi_probs, 3)
    states[1:] = np.where(hi_probs >= 0.5, "TURBULENT", "CALM")
    
    return prob_highs, states


def compute_historical_regimes(candles: list[dict]) -> list[dict]:
    """Applies the regime filter contract to a list of candles.
    
    Contract returns per bar:
      {
        "timestamp": int,
        "regime_label": str,
        "confidence_score": float
      }
    """
    if len(candles) < 30:
        return [
            {
                "timestamp": c["time"],
                "regime_label": "UNDEFINED",
                "confidence_score": 0.5,
            }
            for c in candles
        ]

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    n = len(candles)

    # 1. Volatility indicators (GARCH + HMM) in one pass
    vol_ratios = _garch_vol_all(closes)
    markov_probs, markov_states = _markov_regime_all(closes)

    # 2. Sequential/Rolling Trend Indicators (EMA, ADX, ATR, RSI)
    # We can use simple rolling lists/buffers to avoid O(n^2) re-evaluation
    regimes = []
    
    # Pre-calculate ema9 and ema26 vectors for trend direction
    # Simple vector loop or list comprehension
    k_9 = 2.0 / 10.0
    k_26 = 2.0 / 27.0
    ema9 = np.zeros(n)
    ema26 = np.zeros(n)
    ema9[0] = closes[0]
    ema26[0] = closes[0]
    for i in range(1, n):
        ema9[i] = closes[i] * k_9 + ema9[i - 1] * (1.0 - k_9)
        ema26[i] = closes[i] * k_26 + ema26[i - 1] * (1.0 - k_26)

    # Calculate ADX, ATR, RSI values per index
    # To remain 100% numerically identical to regime.py, we evaluate them on a sliding window of max 240 bars
    # but we can optimize this by maintaining rolling statistics where applicable.
    # To guarantee exact parity, we can fetch sub-slices when needed:
    for i in range(n):
        c = candles[i]
        t = c["time"]
        
        if i < 30:
            regimes.append({
                "timestamp": t,
                "regime_label": "UNDEFINED",
                "confidence_score": 0.5,
            })
            continue

        # EMA Trend direction
        trend_dir = 1 if ema9[i] > ema26[i] else (-1 if ema9[i] < ema26[i] else 0)
        
        # Slice for rolling TA indicators
        sub_closes = closes[max(0, i - 239) : i + 1]
        sub_highs = highs[max(0, i - 239) : i + 1]
        sub_lows = lows[max(0, i - 239) : i + 1]
        
        # Standard TA indicators
        from app.services.regime import _adx, _atr, _rsi
        adx_val = _adx(sub_closes, sub_highs, sub_lows)
        adx_bounded = min(60.0, adx_val)
        
        atr_val = _atr(sub_highs, sub_lows, sub_closes)
        px = closes[i]
        
        # Calculate historical average ATR
        # In regime.py:
        # atr_hist = [ _atr(...) for i in range(40, len(closes), 20) ]
        atr_hist = [
            _atr(sub_highs[: j + 1], sub_lows[: j + 1], sub_closes[: j + 1])
            for j in range(40, len(sub_closes), 20)
        ]
        if atr_hist and len(atr_hist) > 1:
            hist_avg = sum(atr_hist) / len(atr_hist)
            vol_regime = float(atr_val / hist_avg) if hist_avg else 1.0
        else:
            vol_regime = 1.0

        if adx_bounded >= 25:
            trend_label = "STRONG_UP" if trend_dir > 0 else ("STRONG_DOWN" if trend_dir < 0 else "RANGING")
        elif adx_bounded >= 15:
            trend_label = "UP" if trend_dir > 0 else ("DOWN" if trend_dir < 0 else "RANGING")
        else:
            trend_label = "RANGING"

        if vol_regime >= 1.15:
            vol_label = "HIGH"
        elif vol_regime <= 0.85:
            vol_label = "LOW"
        else:
            vol_label = "NORMAL"

        # Combine Trend + Volatility for first-class regime labels
        regime_label = f"{trend_label}_{vol_label}"
        
        # Calculate confidence from the HMM probability
        prob_high = markov_probs[i]
        confidence = prob_high if vol_label == "HIGH" else (1.0 - prob_high)
        
        regimes.append({
            "timestamp": t,
            "regime_label": regime_label,
            "confidence_score": float(np.round(confidence, 3)),
        })

    return regimes
