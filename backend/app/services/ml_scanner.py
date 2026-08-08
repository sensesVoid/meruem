"""ML Scanner Service — XGBoost-based trade setup classifier.

Replaces the Ollama LLM scanner with a fast, deterministic XGBoost gradient
boosting classifier trained exclusively on strategies that passed the QA gate
(BacktestRun.accepted == 1, Strategy.status == 'active').

Training approach (per Pending.md Phase 9):
  - Time-decay walk-forward: each sample is weighted by recency using an
    exponential decay so the model adapts to the most recent validated patterns.
  - Walk-forward expanding window: 70 % train / 30 % OOS test split identical
    to the backtest gate so the metrics are directly comparable.
  - Features: OHLCV-derived indicator vector (the same one Orchestrator already
    computes) + regime labels (trend / volatility / RSI etc.) — NO raw price.
  - Labels: 1 = BUY opportunity, -1 = SELL opportunity, 0 = NO_SIGNAL
    synthesised from the strategy's BacktestRun spec (direction + entry/sl/tp
    confirmed profitable in-sample).
  - Retraining trigger: manual API call OR weekly scheduler task.
  - Persistence: backend/data/models/xgb_scanner.joblib (joblib pickle).
  - Fallback: when no trained model exists, predict() returns None so the
    Orchestrator falls back to the Ollama LLM scanner transparently.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import BACKEND_DIR

MODELS_DIR: pathlib.Path = BACKEND_DIR / "data" / "models"
MODEL_PATH: pathlib.Path = MODELS_DIR / "xgb_scanner.joblib"

# ---- thread-safe state ----
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "trained_at": None,
    "n_samples": 0,
    "n_features": 0,
    "feature_names": [],
    "train_accuracy": 0.0,
    "test_accuracy": 0.0,
    "n_classes": 3,
    "strategy_ids": [],
    "error": None,
    "status": "not_trained",   # not_trained | training | ready | error
}


def _state_snap() -> dict:
    with _LOCK:
        return dict(_STATE)


# ---- feature engineering ----

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


def _adx(closes: list[float], highs: list[float], lows: list[float], period: int = 14) -> float:
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

    def smooth(seq: list[float]) -> float:
        return sum(seq[-period:]) / min(period, len(seq)) if seq else 0.0

    atr_ = smooth(trs)
    pdi = smooth(plus_dm) / atr_ * 100.0 if atr_ else 0.0
    mdi = smooth(minus_dm) / atr_ * 100.0 if atr_ else 0.0
    denom = pdi + mdi
    return abs(pdi - mdi) / denom * 100.0 if denom else 0.0


FEATURE_NAMES = [
    "rsi_14", "atr_14_pct", "adx_14",
    "ema9_above_ema26",        # 1/0
    "ema9_pct_above_ema26",    # signed %
    "mom_20",                  # 20-bar return %
    "close_vs_sma50",          # close / sma50 - 1
    "vol_ratio",               # atr / avg_atr (volatility regime)
    "candle_body_pct",         # body / range ratio of last candle
    "upper_wick_pct",          # upper wick / range
    "lower_wick_pct",          # lower wick / range
    "trend_dir",               # +1/0/-1
    "markov_turbulent",        # 1 if turbulent HMM state
    "garch_expanding",         # 1 if vol expanding
]


def extract_features(rows: list[dict]) -> np.ndarray | None:
    """Convert a list of OHLCV dicts into a 1-D feature vector.

    Returns None when there are insufficient bars.
    """
    if not rows or len(rows) < 60:
        return None

    closes = [float(r["close"]) for r in rows]
    highs = [float(r["high"]) for r in rows]
    lows = [float(r["low"]) for r in rows]
    last = rows[-1]

    rsi = _rsi(closes, 14)
    atr = _atr(highs, lows, closes, 14)
    adx = _adx(closes, highs, lows, 14)
    ema9 = _ema(closes, 9)
    ema26 = _ema(closes, 26)
    px = closes[-1]

    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else px

    # Historical ATR average for vol ratio
    atr_hist = []
    for i in range(40, len(closes), 20):
        atr_hist.append(_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14))
    hist_avg = sum(atr_hist) / len(atr_hist) if atr_hist else atr or 1.0
    vol_ratio = (atr / hist_avg) if hist_avg else 1.0

    # 20-bar momentum
    mom20 = (px - closes[-21]) / closes[-21] * 100.0 if len(closes) > 21 and closes[-21] else 0.0

    # Candle geometry
    o = float(last.get("open", px))
    h = float(last.get("high", px))
    l = float(last.get("low", px))
    c = float(last.get("close", px))
    rng = max(h - l, 1e-10)
    body_pct = abs(c - o) / rng
    upper_wick = (h - max(o, c)) / rng
    lower_wick = (min(o, c) - l) / rng

    trend_dir = 1.0 if ema9 > ema26 else (-1.0 if ema9 < ema26 else 0.0)

    # Regime integration (best-effort)
    markov_turbulent = 0.0
    garch_expanding = 0.0
    try:
        from app.services.regime import _garch_vol, _markov_regime  # type: ignore
        garch = _garch_vol(closes)
        markov = _markov_regime(closes)
        garch_expanding = 1.0 if (garch.get("vol_ratio") or 1.0) >= 1.15 else 0.0
        markov_turbulent = 1.0 if markov.get("state") == "TURBULENT" else 0.0
    except Exception:  # noqa: BLE001
        pass

    return np.array([
        rsi,
        (atr / px * 100.0) if px else 0.0,
        adx,
        1.0 if ema9 > ema26 else 0.0,
        ((ema9 / ema26) - 1.0) * 100.0 if ema26 else 0.0,
        mom20,
        (px / sma50 - 1.0) * 100.0 if sma50 else 0.0,
        vol_ratio,
        body_pct,
        upper_wick,
        lower_wick,
        trend_dir,
        markov_turbulent,
        garch_expanding,
    ], dtype=np.float32)


# ---- label generation from strategy / backtest specs ----

def _label_from_spec(spec: dict) -> int | None:
    """Convert a BacktestRun confluenc spec into a label (1=BUY,-1=SELL,0=NONE).

    The spec written by the synthesizer/refiner is expected to carry a
    ``direction`` key or the RSI/SMA fast vs slow ordering implies it.
    """
    direction = str(spec.get("direction", "") or spec.get("side", "")).upper()
    if direction in ("BUY", "LONG"):
        return 1
    if direction in ("SELL", "SHORT"):
        return -1
    # Infer from params: fast > slow => bullish
    fast = spec.get("fast") or spec.get("fast_period") or 0
    slow = spec.get("slow") or spec.get("slow_period") or 0
    if fast and slow:
        return 1 if int(fast) < int(slow) else -1
    return None


# ---- dataset builder ----

def _fetch_training_rows(symbol: str, timeframe: str, bars: int = 500) -> list[dict]:
    """OHLCV rows for ML training features, in priority order:

    1. The extracted Parquet dataset for ``(symbol, timeframe)`` (most recent
       ``bars`` rows) — long-history archives as documented;
    2. Live ``MarketDataService`` (MT5 / Binance / Yahoo) fetch.
    """
    from app.services.dataset_service import read_dataset_candles

    try:
        rows = read_dataset_candles(symbol, timeframe, bars=bars)
        if rows:
            return rows
    except Exception:  # noqa: BLE001 - fall back to live data
        pass

    import asyncio

    from app.services.market_data import MarketDataService

    async def _live():
        md = MarketDataService()
        try:
            data = await md.fetch(symbol=symbol, timeframe=timeframe, limit=bars)
            return data.get("candles") or []
        except Exception:  # noqa: BLE001
            return []
        finally:
            await md.close()

    try:
        return asyncio.run(_live())
    except Exception:  # noqa: BLE001
        return []


def _build_dataset(session) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Pull all accepted BacktestRuns from active green-gate strategies and build
    the training (X, y, weights) arrays.

    A balanced synthetic dataset is generated: for each accepted run we sample
    N=30 feature windows from the historical parquet file (if available) or from
    MarketDataService, applying time-decay weighting so recent patterns dominate.
    """
    from sqlmodel import select

    from app.models.entities import BacktestRun
    from app.models.entities import Strategy as StrategyModel

    # Only train on strategies that passed the full QA gate (green).
    active_ids = [
        s.id for s in session.exec(
            select(StrategyModel).where(StrategyModel.status == "active")
        ).all()
    ]
    if not active_ids:
        return None

    runs = session.exec(
        select(BacktestRun).where(
            BacktestRun.strategy_id.in_(active_ids),
            BacktestRun.accepted == 1,
        ).order_by(BacktestRun.created_at.desc())
    ).all()
    if not runs:
        return None

    Xs, ys, ws, strat_ids_used = [], [], [], []
    now_ts = time.time()

    DECAY_HALF_LIFE_DAYS = 60.0  # 60-day half-life for time-decay weight

    for run in runs:
        try:
            spec = json.loads(run.spec_json or "{}")
            metrics = json.loads(run.metrics_json or "{}")
        except (json.JSONDecodeError, TypeError):
            continue

        label = _label_from_spec(spec)
        if label is None:
            continue

        # Time-decay weight based on how old this run is
        age_days = (now_ts - run.created_at.timestamp()) / 86400.0
        weight = 2 ** (-age_days / DECAY_HALF_LIFE_DAYS)

        # Pull OHLCV for feature extraction (extracted dataset preferred)
        rows = _fetch_training_rows(run.symbol, run.timeframe, bars=500)
        if len(rows) < 60:
            continue

        # Walk-forward windows: extract features from multiple sub-windows to
        # increase sample count and teach temporal generalisation.
        n_rows = len(rows)
        step = max(10, n_rows // 20)
        for start in range(60, n_rows, step):
            window = rows[:start]
            feats = extract_features(window)
            if feats is None:
                continue
            Xs.append(feats)
            ys.append(label)
            # Older windows get progressively less weight within the same run
            recency = start / n_rows
            ws.append(weight * recency)

        if run.strategy_id not in strat_ids_used:
            strat_ids_used.append(run.strategy_id)

    if not Xs:
        return None

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int32)
    w = np.array(ws, dtype=np.float32)
    w = w / w.sum()  # normalise to sum-1

    return X, y, w, strat_ids_used


# ---- training ----

def train(session) -> dict:
    """Train / retrain the XGBoost scanner classifier and persist it.

    Returns a status dict. Thread-safe; blocks until training completes.
    """
    import joblib
    import xgboost as xgb

    with _LOCK:
        _STATE["status"] = "training"
        _STATE["error"] = None

    try:
        result = _build_dataset(session)
        if result is None:
            msg = (
                "No training data: you need at least one active strategy with "
                "an accepted BacktestRun (green QA gate). Synthesize & approve "
                "a strategy first."
            )
            with _LOCK:
                _STATE.update({"status": "error", "error": msg})
            return {"ok": False, "error": msg}

        X, y, w, strat_ids = result
        n_samples = len(X)

        # Walk-forward train/test split (chronological — same 70/30 as backtest gate)
        split = int(n_samples * 0.70)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        w_train = w[:split]

        # XGBoost multi-class (0=SELL, 1=NO_SIGNAL, 2=BUY) after mapping labels
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_m = np.array([label_map[v] for v in y_train], dtype=np.int32)
        y_test_m = np.array([label_map[v] for v in y_test], dtype=np.int32)

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            num_class=3,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(
            X_train, y_train_m,
            sample_weight=w_train * n_samples,  # XGBoost prefers unnormalised weights
            eval_set=[(X_test, y_test_m)],
            verbose=False,
        )

        train_acc = float((clf.predict(X_train) == y_train_m).mean())
        test_acc = float((clf.predict(X_test) == y_test_m).mean()) if len(X_test) > 0 else 0.0

        # Persist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_bundle = {
            "clf": clf,
            "feature_names": FEATURE_NAMES,
            "label_map": label_map,
            "inv_label_map": {v: k for k, v in label_map.items()},
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(model_bundle, MODEL_PATH)

        with _LOCK:
            _STATE.update({
                "trained_at": model_bundle["trained_at"],
                "n_samples": n_samples,
                "n_features": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES,
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "strategy_ids": strat_ids,
                "error": None,
                "status": "ready",
            })

        return {
            "ok": True,
            "n_samples": n_samples,
            "n_features": len(FEATURE_NAMES),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "strategy_ids": strat_ids,
            "trained_at": model_bundle["trained_at"],
        }

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        with _LOCK:
            _STATE.update({"status": "error", "error": msg})
        return {"ok": False, "error": msg}


# ---- inference ----

def predict(rows: list[dict]) -> dict | None:
    """Run inference on a window of OHLCV rows.

    Returns a dict with direction/confidence/probabilities, or None if the
    model is not trained (caller should fall back to Ollama scanner).
    """
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(MODEL_PATH)
        clf = bundle["clf"]
        inv_map = bundle["inv_label_map"]

        feats = extract_features(rows)
        if feats is None:
            return None

        X = feats.reshape(1, -1)
        proba = clf.predict_proba(X)[0]  # [P(SELL), P(NONE), P(BUY)]
        pred_class = int(np.argmax(proba))
        label = inv_map[pred_class]
        confidence = float(proba[pred_class])

        direction_map = {1: "BUY", -1: "SELL", 0: None}
        return {
            "direction": direction_map.get(label),
            "confidence": round(confidence, 4),
            "probabilities": {
                "sell": round(float(proba[0]), 4),
                "none": round(float(proba[1]), 4),
                "buy": round(float(proba[2]), 4),
            },
            "label": label,
        }
    except Exception:  # noqa: BLE001
        return None


def status() -> dict:
    """Return a live status snapshot (thread-safe)."""
    snap = _state_snap()
    if MODEL_PATH.exists() and snap["status"] == "not_trained":
        # Model file exists but state was lost on restart → reload metadata
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            with _LOCK:
                _STATE.update({
                    "trained_at": bundle.get("trained_at"),
                    "feature_names": bundle.get("feature_names", FEATURE_NAMES),
                    "n_features": len(bundle.get("feature_names", FEATURE_NAMES)),
                    "status": "ready",
                })
            snap = _state_snap()
        except Exception:  # noqa: BLE001
            pass
    snap["model_path"] = str(MODEL_PATH)
    snap["model_exists"] = MODEL_PATH.exists()
    return snap
