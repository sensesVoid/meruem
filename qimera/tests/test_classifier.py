"""Tests for the M8 unified rolling-window XGBoost classifier."""

import json

import numpy as np
import pandas as pd
import pytest

from qimera.app.classifier import (
    ClassifierConfig, available, build_feature_matrix, drift_report,
    feature_columns, predict_gated, rolling_validate, train_classifier,
)
from qimera.app.regime_contract import compute_historical_regimes


@pytest.fixture
def df():
    """Trending synthetic OHLCV — the label (up/horizon) is learnable."""
    np.random.seed(42)
    n = 600
    times = np.arange(1690000000, 1690000000 + n * 900, 900)
    close = 1.09 * np.cumprod(1 + np.random.normal(0.0004, 0.002, n))
    open_p = close / (1 + np.random.normal(0, 0.0005, n))
    high = np.maximum(open_p, close) * (1 + np.random.uniform(0, 0.001, n))
    low = np.minimum(open_p, close) * (1 - np.random.uniform(0, 0.001, n))
    return pd.DataFrame({
        "time": times,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": np.random.uniform(10, 1000, n),
    })


def _candles(df):
    return [
        {k: float(row[k]) for k in ("time", "open", "high", "low", "close", "volume")}
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def test_feature_matrix_shape_and_columns(df):
    frame = build_feature_matrix(df)
    cfg = ClassifierConfig()
    assert "label" in frame.columns
    assert len(frame) <= len(df) - cfg.horizon - cfg.lookback
    cols = feature_columns(frame)
    assert {"returns_1", "momentum_ratio", "atr_norm", "regime_code"} <= set(cols)
    assert frame["label"].isin([0, 1]).all()
    assert frame.isna().sum().sum() == 0


def test_feature_matrix_includes_regime_features(df):
    candles = _candles(df)
    regimes = compute_historical_regimes(candles)
    frame = build_feature_matrix(df, regimes)
    assert "regime_code" in frame.columns
    assert "regime_confidence" in frame.columns
    assert frame["regime_confidence"].between(0, 1).all()


def test_decay_weights_recent_bars_dominate():
    cfg = ClassifierConfig(decay_lambda=0.1)
    w = cfg.weights_for(100)
    assert w[-1] > w[0]  # most recent bar weighs most
    assert w[0] == pytest.approx(np.exp(-0.1 * 99), rel=1e-9)


# ---------------------------------------------------------------------------
# Training + gating
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not available(), reason="xgboost not installed")
def test_train_classifier_learns_trending_data(df):
    cfg = ClassifierConfig()
    frame = build_feature_matrix(df)
    result = train_classifier(frame, cfg)
    assert result["available"] is True
    assert result["model"] is not None
    assert result["metrics"]["train_accuracy"] > 0.5
    assert set(result["feature_importance"]) == set(feature_columns(frame))
    assert "regime_code" in result["feature_importance"]


@pytest.mark.skipif(not available(), reason="xgboost not installed")
def test_predict_gated_withholds_low_confidence(df):
    cfg = ClassifierConfig()
    frame = build_feature_matrix(df)
    result = train_classifier(frame, cfg)

    # Force every bar below the gate -> everything withheld.
    low = frame.copy()
    low["regime_confidence"] = 0.1
    preds = predict_gated(result["model"], low, cfg)
    assert len(preds) == len(low)
    assert all(p["gated"] for p in preds)

    # No gating when confidence is maximal.
    high = frame.copy()
    high["regime_confidence"] = 0.99
    preds2 = predict_gated(result["model"], high, cfg)
    assert all(not p["gated"] for p in preds2)
    assert all(p["pred_class"] in (0, 1) for p in preds2)


# ---------------------------------------------------------------------------
# Rolling validation + drift monitor
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not available(), reason="xgboost not installed")
def test_rolling_validate_produces_windows(df):
    cfg = ClassifierConfig()
    res = rolling_validate(df, n_windows=3, cfg=cfg)
    assert len(res["windows"]) == 3
    for w in res["windows"]:
        assert "oos_accuracy" in w
        assert "drift_mean_ks" in w
        # Predictions only cover the OOS half of each window.
        assert 0 < w["n_ungated"] + w["n_gated"] <= w["bars"]
    json.dumps(res)


def test_drift_report_flags_shifted_feature():
    n = 200
    a = pd.DataFrame({
        "returns_1": np.random.normal(0, 0.01, n),
        "regime_code": np.zeros(n, dtype=int),
    })
    b = pd.DataFrame({
        "returns_1": np.random.normal(0.05, 0.01, n),  # mean shift
        "regime_code": np.zeros(n, dtype=int),
    })
    report = drift_report(a, b, ClassifierConfig(drift_alpha=0.05))
    assert "returns_1" in report["warned_features"]
    assert report["mean_ks"] > 0


def test_drift_report_no_warning_for_same_distribution():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    a = pd.DataFrame({"returns_1": x})
    b = pd.DataFrame({"returns_1": rng.normal(0, 1, 200)})
    report = drift_report(a, b, ClassifierConfig(drift_alpha=0.05))
    assert "returns_1" not in report["warned_features"]


def test_save_drift_report_writes_jsonl(tmp_path):
    from qimera.app.classifier import save_drift_report

    p = save_drift_report({"mean_ks": 0.5, "features": {}}, path=tmp_path)
    assert p.is_file()
    assert json.loads(p.read_text(encoding="utf-8"))["mean_ks"] == 0.5
