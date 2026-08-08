"""Market Regime Agent.

Classifies the current market condition per watchlist asset and across the
portfolio. Evaluates Trend Direction, Trend Strength, Volatility Regime,
Momentum & Exhaustion, and Market Breadth using deterministic OHLCV indicators
(no external TA dependency). The 5-minute scheduler recomputes it; the
Validator reads the cached regime as extra reference when judging a candidate.
"""

import asyncio
import threading
from datetime import datetime, timezone

import numpy as np
from sqlmodel import Session, select

from app.models.entities import Asset

_STATE: dict = {
    "refreshed_at": None,
    "assets": {},  # {symbol: {read}}
    "breadth": {},
    "overall": "NEUTRAL",
    "error": None,
}
_LOCK = threading.Lock()

_TIMEFRAME = "1h"
_BARS = 240


# --------------------------------------------------------------------------
# Indicator helpers (self-contained, no TA lib)
# --------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(values: list[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    roc = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(x, 0.0) for x in roc[-period:]]
    losses = [max(-x, 0.0) for x in roc[-period:]]
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return sum(trs[-period:]) / period


def _adx(closes, highs, lows, period: int = 14) -> float:
    """Wilder-style ADX proxy: spread of +DI/-DI in %."""
    n = len(closes)
    if n < period * 2:
        return 0.0
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    atr_ = _smooth(trs, period)
    pdi = _smooth(plus_dm, period) / atr_ * 100.0 if atr_ else 0.0
    mdi = _smooth(minus_dm, period) / atr_ * 100.0 if atr_ else 0.0
    denom = pdi + mdi
    if denom == 0:
        return 0.0
    return abs(pdi - mdi) / denom * 100.0


def _smooth(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    return sum(values[-period:]) / min(period, len(values))


# --------------------------------------------------------------------------
# Quant layers: GARCH(1,1) conditional volatility + 2-state Markov regime
# --------------------------------------------------------------------------
# Decision (documented in Pending.md Phase 4): regime detection stays a
# DETERMINISTIC indicator stack (EMA/ADX/ATR/RSI + numpy GARCH + HMM-EM). No
# LLM in the hot path (it must never shape releases) and no XGBoost yet
# (the scanner transition to XGBoost is a separate later phase). These two
# layers are pure-numpy so the 5-min refresh stays cheap and dependency-free.

def _log_returns(closes: list[float]) -> np.ndarray:
    arr = np.asarray(closes, dtype=float)
    if len(arr) < 2:
        return np.array([])
    rets = np.diff(np.log(np.maximum(arr, 1e-12)))
    return rets[np.isfinite(rets)]


def _normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((x[:, None] - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def _garch_vol(closes: list[float],
               alpha_grid=(0.05, 0.10, 0.15, 0.20),
               beta_grid=(0.80, 0.85, 0.90, 0.95)) -> dict:
    """GARCH(1,1) conditional variance via variance-targeting + grid MLE.

    Returns the one-step conditional volatility (latest sigma_t) vs the long-run
    unconditional sigma, plus fit params. ``vol_ratio > 1`` = current vol above
    its unconditional level (vol shock / clustering), ``< 1`` = compressing.
    """
    rets = _log_returns(closes)
    if len(rets) < 60:
        return {"cond_vol": 0.0, "long_run_vol": 0.0, "vol_ratio": 1.0,
                "alpha": 0.0, "beta": 0.0, "omega": 0.0, "note": "insufficient"}

    r = rets - rets.mean()
    long_run = float(r.var())
    if long_run <= 0 or not np.isfinite(long_run):
        return {"cond_vol": 0.0, "long_run_vol": 0.0, "vol_ratio": 1.0,
                "alpha": 0.0, "beta": 0.0, "omega": 0.0, "note": "no_variance"}

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
        return {"cond_vol": 0.0, "long_run_vol": 0.0, "vol_ratio": 1.0,
                "alpha": 0.0, "beta": 0.0, "omega": 0.0, "note": "unfit"}

    _, alpha, beta, omega = best
    s2 = long_run
    for rt in r:
        s2 = omega + alpha * rt * rt + beta * s2
    cond_vol = float(np.sqrt(max(s2, 1e-12)))
    long_vol = float(np.sqrt(max(long_run, 1e-12)))
    return {
        "cond_vol": round(cond_vol, 8),
        "long_run_vol": round(long_vol, 8),
        "vol_ratio": round(cond_vol / long_vol, 3) if long_vol > 0 else 1.0,
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "omega": round(float(omega), 10),
    }


def _markov_regime(closes: list[float], iters: int = 30) -> dict:
    """2-state Gaussian HMM (low-CALM / high-TURBULENT vol) via plain EM.

    Returns the filtered probability that the CURRENT bar sits in the HIGH-vol
    state, per-state vol levels, and high-state persistence (transition A[hi,hi]
    -> a regime switch reads as a DROP in what follows: see the orchestrator).
    """
    rets = _log_returns(closes)
    n = len(rets)
    if n < 80:
        return {"state": "CALM", "prob_high": 0.0, "vol_calm": 0.0,
                "vol_turb": 0.0, "persistence": 0.0, "note": "insufficient"}

    r = rets - rets.mean()
    sd = float(r.std())
    if sd <= 0 or not np.isfinite(sd):
        return {"state": "CALM", "prob_high": 0.0, "vol_calm": 0.0,
                "vol_turb": 0.0, "persistence": 0.0, "note": "no_variance"}

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
        xi = np.zeros((n - 1, 2, 2))
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
    # Guard against degenerate EM on homogeneous data: if the two states barely
    # separate (sig_ratio ~ 1) treat the series as calm, not TURBULENT. Blend the
    # raw filtered probability with the separation so noise can't flip the label.
    separation = min(1.0, (sig_ratio - 1.0) / 0.5)
    hi_prob = float(gamma[-1, hi]) * separation
    return {
        "state": "TURBULENT" if hi_prob >= 0.5 else "CALM",
        "prob_high": round(hi_prob, 3),
        "vol_calm": round(float(sig[lo]) * sd, 6),
        "vol_turb": round(float(sig[hi]) * sd, 6),
        "persistence": round(float(A[hi, hi]) * separation, 3),
    }


# --------------------------------------------------------------------------
# Per-asset analysis
# --------------------------------------------------------------------------

def analyze_asset(symbol: str, rows: list[dict]) -> dict:
    """Return the regime read for one symbol from its OHLCV rows."""
    if not rows or len(rows) < 30:
        return {
            "symbol": symbol, "label": "UNDEFINED", "trend_direction": 0,
            "trend_label": "RANGING", "trend_strength": 0.0,
            "volatility_regime": "NORMAL", "volatility_pct": 0.0, "atr_ratio": 1.0,
            "momentum_pct": 0.0, "rsi": 50.0, "exhausted": False,
            "momentum_label": "FLAT",
            "garch": None, "markov": None,
            "note": "insufficient bars to classify regime",
        }

    closes = [float(r["close"]) for r in rows]
    highs = [float(r["high"]) for r in rows]
    lows = [float(r["low"]) for r in rows]

    garch = _garch_vol(closes)
    markov = _markov_regime(closes)

    ema_fast = _ema(closes, 9)
    ema_slow = _ema(closes, 26)
    trend_dir = 1 if ema_fast > ema_slow else (-1 if ema_fast < ema_slow else 0)
    px = closes[-1]

    adx = _adx(closes, highs, lows)
    adx_bounded = min(60.0, adx)

    atr = _atr(highs, lows, closes)
    vol_pct = min(100.0, (atr / px * 100.0) * 4.0) if px else 0.0
    atr_hist = [
        _atr(highs[: i + 1], lows[: i + 1], closes[: i + 1])
        for i in range(40, len(closes), 20)
    ]
    if atr_hist and len(atr_hist) > 1:
        hist_avg = sum(atr_hist) / len(atr_hist)
        vol_regime = float(atr / hist_avg) if hist_avg else 1.0
    else:
        vol_regime = 1.0

    rsi = _rsi(closes, 14)
    ret20 = (px - closes[-21]) / closes[-21] if len(closes) > 21 and closes[-21] else 0.0
    momentum = float(ret20 * 100.0)
    exhausted = rsi >= 70 or rsi <= 30

    if adx_bounded >= 25:
        trend_label = ("STRONG_UP" if trend_dir > 0 else
                       "STRONG_DOWN" if trend_dir < 0 else "RANGING")
    elif adx_bounded >= 15:
        trend_label = ("UP" if trend_dir > 0 else "DOWN" if trend_dir < 0 else "RANGING")
    else:
        trend_label = "RANGING"

    if vol_regime >= 1.15:
        vol_label = "HIGH"
    elif vol_regime <= 0.85:
        vol_label = "LOW"
    else:
        vol_label = "NORMAL"

    if exhausted:
        mom_label = "EXHAUSTED"
    elif momentum > 0.5:
        mom_label = "MOMENTUM_UP"
    elif momentum < -0.5:
        mom_label = "MOMENTUM_DOWN"
    else:
        mom_label = "FLAT"

    vol_regime_ga = (garch.get("vol_ratio") or 1.0) if garch else 1.0
    if vol_regime_ga >= 1.15:
        garch_label = "GARCH_EXPANDING"
    elif vol_regime_ga <= 0.85:
        garch_label = "GARCH_COMPRESSING"
    else:
        garch_label = "GARCH_STABLE"

    return {
        "symbol": symbol,
        "last_close": round(px, 5),
        "trend_direction": trend_dir,
        "trend_label": trend_label,
        "trend_strength": round(adx_bounded, 2),
        "volatility_regime": vol_label,
        "volatility_pct": round(vol_pct, 2),
        "atr_ratio": round(vol_regime, 2),
        "momentum_pct": round(momentum, 2),
        "rsi": round(rsi, 1),
        "exhausted": exhausted,
        "momentum_label": mom_label,
        "garch": garch,
        "garch_label": garch_label,
        "markov": markov,
        "label": (f"{trend_label} trend (ADX {adx_bounded:.0f}), {vol_label} volatility, "
                  f"RSI {rsi:.0f}, {mom_label}, {garch_label}"
                  f"{', ' + markov['state'] if markov else ''}."),
    }


def _breadth(readings: list[dict]) -> dict:
    up = sum(1 for a in readings if a["trend_direction"] > 0)
    down = sum(1 for a in readings if a["trend_direction"] < 0)
    total = len(readings) or 1
    up_pct = up / total * 100.0
    down_pct = down / total * 100.0
    if up_pct >= 60 and up > down:
        label = "BROAD-UP"
    elif down_pct >= 60 and down > up:
        label = "BROAD-DOWN"
    else:
        label = "MIXED"
    return {"up_pct": round(up_pct, 1), "down_pct": round(down_pct, 1), "label": label}


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class RegimeAgent:
    """Market regime classifier with an in-process cache."""

    def _market_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[dict]:
        async def _fetch():
            from app.services.market_data import MarketDataService

            md = MarketDataService()
            try:
                data = await md.fetch(symbol=symbol, timeframe=timeframe, limit=limit)
                return data.get("candles") or []
            finally:
                await md.close()

        try:
            return asyncio.run(_fetch())
        except Exception:  # noqa: BLE001
            return []

    def refresh(self, session: Session) -> dict:
        try:
            assets = session.exec(select(Asset)).all()
            readings = []
            for asset in assets:
                rows = self._market_ohlcv(asset.symbol, _TIMEFRAME, _BARS)
                reading = analyze_asset(asset.symbol, rows)
                reading["asset_class"] = asset.asset_class
                readings.append(reading)

            breadth = _breadth(readings)
            up_pct = breadth["up_pct"]
            if up_pct >= 65:
                overall = "BULLISH_REGIME"
            elif up_pct <= 35:
                overall = "BEARISH_REGIME"
            else:
                overall = "MIXED_REGIME"

            with _LOCK:
                _STATE.update({
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                    "assets": {r["symbol"]: r for r in readings},
                    "breadth": breadth,
                    "overall": overall,
                    "error": None,
                })
        except Exception as exc:  # noqa: BLE001
            with _LOCK:
                _STATE["error"] = f"{type(exc).__name__}: {exc}"
        return state()


def state() -> dict:
    """Thread-safe snapshot of the latest regime cache."""
    with _LOCK:
        return {
            "refreshed_at": _STATE["refreshed_at"],
            "assets": dict(_STATE["assets"] or {}),
            "breadth": dict(_STATE["breadth"] or {}),
            "overall": _STATE["overall"],
            "error": _STATE["error"],
        }
