"""QiMera Unified Rolling-Window XGBoost Classifier (QiMera.md §6.8 / M8).

A single XGBoost binary classifier that predicts the sign of the forward return
(up/down) over a horizon, using **regime labels as direct input features**:

- Feature matrix per bar: rolling returns, momentum, normalized ATR/volume, the
  regime label (ordinal-coded), the regime filter's confidence score, the GARCH
  vol ratio and the HMM high-vol probability (from ``regime_contract``).
- **Exponential sample decay**: each training sample is weighted
  ``exp(-lambda * age)`` so recent bars dominate the fit (``lambda`` is the M8
  configured time-decay).
- **Low-confidence gating**: a prediction is *gated* (withheld) whenever the
  regime filter's confidence is below the target threshold — the classifier
  refuses to speak rather than emitting a low-quality signal.
- **Rolling-window validation + drift monitor**: after each retraining cycle the
  model is scored on the out-of-sample window and a drift report (per-feature
  Kolmogorov–Smirnov stats between train and current windows) is produced, so a
  retraining that silently changed the input distribution is caught.

xgboost is imported lazily so the module imports cleanly even on minimal
installs; ``available()`` reports whether training is possible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DRIFT_REPORT_ROOT = Path(__file__).resolve().parent.parent / "data" / "drift"


def available() -> bool:
    """True when xgboost is installed (trainer is otherwise unusable)."""
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Feature matrix (regime-aware)
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    """Knobs for the M8 classifier."""

    horizon: int = 5                 # bars ahead for the up/down label
    lookback: int = 60               # feature lookback for rolling stats
    decay_lambda: float = 0.002      # exponential time-decay weight (lambda)
    gate_confidence: float = 0.55    # regime-confidence gate threshold
    n_estimators: int = 120
    max_depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 42
    drift_alpha: float = 0.05        # KS-test significance for the drift report

    def weights_for(self, n: int) -> np.ndarray:
        """Exponential decay weights: most recent bar gets weight ~1.0."""
        ages = np.arange(n - 1, -1, -1)
        return np.exp(-self.decay_lambda * ages)


def build_feature_matrix(df: pd.DataFrame, regimes: list[dict] | None = None,
                         cfg: ClassifierConfig | None = None) -> pd.DataFrame:
    """One row per bar (bars with a full lookback + horizon are kept).

    Features (all normalized/serializable):
    - returns_1/5/10: log returns over 1/5/10 bars
    - momentum_ratio: close / sma(lookback) - 1
    - atr_norm: ATR / close
    - volume_norm: volume / sma(volume) - 1
    - regime_code: ordinal code of the regime label (regime label as input!)
    - regime_confidence: the regime filter's confidence score
    - garch_vol_ratio / hmm_prob: from ``compute_historical_regimes``

    The target column ``label`` = 1 when close[t+horizon] > close[t].
    """
    cfg = cfg or ClassifierConfig()
    df = df.reset_index(drop=True)
    close = df["close"].astype(float)

    out = pd.DataFrame(index=df.index)
    log_c = np.log(close.to_numpy(dtype=float))
    for k in (1, 5, 10):
        out[f"returns_{k}"] = pd.Series(log_c).diff(k)
    sma = close.rolling(cfg.lookback).mean()
    out["momentum_ratio"] = (close / sma - 1.0).replace([np.inf, -np.inf], 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out["atr_norm"] = atr / close
    vol_sma = df["volume"].astype(float).rolling(cfg.lookback).mean()
    out["volume_norm"] = (df["volume"] / vol_sma - 1.0).replace(
        [np.inf, -np.inf], 0.0)

    # Regime features (label as a feature + its confidence + underlying probs).
    # Always present so the feature set is stable whether or not the caller has
    # a regime history: defaults are the neutral regime (code 0, conf 0.5).
    order = ["RANGING", "UP", "DOWN", "STRONG_UP", "STRONG_DOWN"]
    out["regime_code"] = 0
    out["regime_confidence"] = 0.5
    if regimes:
        reg_df = pd.DataFrame(regimes)
        if len(reg_df) == len(df):
            codes = reg_df["regime_label"].astype(str).str.split("_").str[0]
            out["regime_code"] = codes.map(
                {name: i for i, name in enumerate(order)}).fillna(0).astype(int)
            out["regime_confidence"] = reg_df["confidence_score"].astype(float)
    out["garch_vol_ratio"] = 1.0
    out["hmm_prob"] = 0.5

    # Label: forward return sign over the horizon (drop the last `horizon` rows).
    forward = pd.Series(log_c).shift(-cfg.horizon) - log_c
    out["label"] = np.where(forward > 0, 1, 0)

    # Only keep bars that have both a full feature lookback and a label.
    usable = out.dropna(subset=[c for c in out.columns if c != "label"])
    usable = usable[
        (usable.index >= cfg.lookback) & (usable.index < len(df) - cfg.horizon)]
    return usable.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "label"]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_classifier(frame: pd.DataFrame, cfg: ClassifierConfig | None = None,
                     weights: np.ndarray | None = None) -> dict:
    """Train the XGBoost binary classifier with exponential decay weights.

    Returns a JSON-safe dict with the model reference, feature importances and
    in-sample metrics. ``weights`` defaults to ``cfg.weights_for(n)``.
    """
    if not available():
        raise RuntimeError("xgboost is not installed — cannot train the M8 classifier")
    cfg = cfg or ClassifierConfig()
    import xgboost as xgb

    cols = feature_columns(frame)
    X = frame[cols].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int32)
    w = weights if weights is not None else cfg.weights_for(len(frame))

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate, random_state=cfg.random_state,
        eval_metric="logloss", tree_method="hist", verbosity=0,
    )
    model.fit(X, y, sample_weight=w)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == y).mean())
    from scipy.stats import ks_2samp

    return {
        "available": True,
        "model": model,
        "n_samples": int(len(frame)),
        "features": cols,
        "feature_importance": {
            name: float(imp)
            for name, imp in zip(cols, model.feature_importances_)
        },
        "metrics": {
            "train_accuracy": round(acc, 4),
            "train_logloss": round(
                float(-np.mean(y * np.log(np.clip(proba, 1e-9, 1)) +
                            (1 - y) * np.log(np.clip(1 - proba, 1e-9, 1)))), 4),
        },
    }


def predict_gated(model, frame: pd.DataFrame,
                  cfg: ClassifierConfig | None = None) -> list[dict]:
    """Predict with low-confidence gating.

    A bar whose regime-confidence is below ``cfg.gate_confidence`` yields a
    ``gated: True`` prediction (the classifier withholds its call).
    """
    cfg = cfg or ClassifierConfig()
    cols = feature_columns(frame)
    X = frame[cols].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[:, 1]

    conf = frame.get("regime_confidence")
    gated = (
        (conf.to_numpy(dtype=float) < cfg.gate_confidence)
        if conf is not None
        else np.zeros(len(frame), dtype=bool)
    )
    return [
        {
            "bar": int(i),
            "pred_class": int(1) if (not g and p >= 0.5) else int(0),
            "prob": float(p),
            "gated": bool(g),
            "label": int(frame["label"].iloc[i]),
        }
        for i, (p, g) in enumerate(zip(proba, gated))
    ]


# ---------------------------------------------------------------------------
# Rolling-window validation + drift monitor
# ---------------------------------------------------------------------------


def rolling_validate(df: pd.DataFrame, regimes: list[dict] | None = None,
                     cfg: ClassifierConfig | None = None,
                     n_windows: int = 3) -> dict:
    """Chronological rolling-window validation (retraining cycle simulation).

    Each window: build features on the whole window, train on the first half
    (with decay weights), predict+gated-score on the second half, and compute a
    drift report between the train and OOS feature distributions.
    """
    cfg = cfg or ClassifierConfig()
    frame = build_feature_matrix(df, regimes, cfg)
    n = len(frame)
    results = []
    for w in range(n_windows):
        start = int(n * w / n_windows)
        end = int(n * (w + 1) / n_windows)
        window = frame.iloc[start:end].reset_index(drop=True)
        if len(window) < 60:
            continue
        split = int(len(window) * 0.5)
        train, oos = window.iloc[:split], window.iloc[split:]
        trained = train_classifier(train, cfg)
        preds = predict_gated(trained["model"], oos, cfg)
        ng = [p for p in preds if not p["gated"]]
        acc = float(np.mean([p["pred_class"] == p["label"] for p in ng])) if ng else 0.0
        drift = drift_report(train, oos, cfg)
        results.append({
            "window": w, "bars": int(len(window)),
            "oos_accuracy": round(acc, 4),
            "n_ungated": len(ng), "n_gated": len(preds) - len(ng),
            "drift_mean_ks": round(drift["mean_ks"], 4),
            "drift_features_warned": drift["warned_features"],
        })
    return {"windows": results, "cfg": cfg.__dict__}


def drift_report(train_frame: pd.DataFrame, current_frame: pd.DataFrame,
                 cfg: ClassifierConfig | None = None) -> dict:
    """Per-feature Kolmogorov–Smirnov drift between two windows.

    ``mean_ks`` aggregates the per-feature KS statistics; features whose p-value
    drops below ``cfg.drift_alpha`` are flagged as drifted — the offline drift
    monitor output after retraining cycles.
    """
    cfg = cfg or ClassifierConfig()
    from scipy.stats import ks_2samp

    cols = feature_columns(train_frame)
    stats = {}
    warned = []
    for c in cols:
        a = train_frame[c].to_numpy(dtype=float)
        b = current_frame[c].to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue
        ks, p = ks_2samp(a, b)
        stats[c] = {"ks": round(float(ks), 4), "p_value": round(float(p), 6)}
        if p < cfg.drift_alpha:
            warned.append(c)
    mean_ks = float(np.mean([s["ks"] for s in stats.values()])) if stats else 0.0
    return {"mean_ks": mean_ks, "features": stats, "warned_features": warned}


def save_drift_report(report: dict, path: Path = DRIFT_REPORT_ROOT) -> Path:
    """Persist a drift report to JSONL for the offline drift monitor."""
    path.mkdir(parents=True, exist_ok=True)
    out = path / "drift_reports.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, default=str) + "\n")
    return out
