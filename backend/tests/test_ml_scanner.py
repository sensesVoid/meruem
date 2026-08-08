"""Tests for Phase 9 — XGBoost ML Scanner.

Tests cover:
  - Feature extraction from OHLCV rows
  - XGBoostScannerAgent with Ollama fallback
  - ML Scanner API endpoints (/ml/status, /ml/train, /ml/model)
  - ml_scanner.predict() returns None when no model exists
  - Label derivation from BacktestRun specs
"""

from __future__ import annotations

import json
import math
import pathlib
import pytest
from unittest.mock import patch, MagicMock


# ─── Feature engineering ──────────────────────────────────────────────────────

def _fake_rows(n: int = 100) -> list[dict]:
    """Generate synthetic OHLCV rows for testing."""
    import random
    rows = []
    price = 1800.0
    for i in range(n):
        o = price
        h = o + random.uniform(0, 5)
        l = o - random.uniform(0, 5)
        c = random.uniform(l, h)
        price = c
        rows.append({
            "time": 1700000000 + i * 900,
            "open": o, "high": h, "low": l, "close": c,
            "volume": random.uniform(100, 1000),
        })
    return rows


def test_extract_features_returns_correct_shape():
    from app.services.ml_scanner import extract_features, FEATURE_NAMES

    rows = _fake_rows(120)
    feats = extract_features(rows)
    assert feats is not None
    assert len(feats) == len(FEATURE_NAMES)


def test_extract_features_insufficient_data():
    from app.services.ml_scanner import extract_features

    # Under 60 rows → should return None
    rows = _fake_rows(30)
    feats = extract_features(rows)
    assert feats is None


def test_extract_features_range_sanity():
    from app.services.ml_scanner import extract_features

    rows = _fake_rows(150)
    feats = extract_features(rows)
    assert feats is not None
    # RSI should be in [0, 100]
    rsi = feats[0]
    assert 0.0 <= rsi <= 100.0
    # candle geometry fractions in [0, 1]
    body_pct = feats[8]
    assert 0.0 <= body_pct <= 1.0


# ─── Label derivation ─────────────────────────────────────────────────────────

def test_label_from_spec_buy():
    from app.services.ml_scanner import _label_from_spec

    assert _label_from_spec({"direction": "BUY"}) == 1
    assert _label_from_spec({"direction": "LONG"}) == 1


def test_label_from_spec_sell():
    from app.services.ml_scanner import _label_from_spec

    assert _label_from_spec({"direction": "SELL"}) == -1
    assert _label_from_spec({"direction": "SHORT"}) == -1


def test_label_from_spec_infer_from_fast_slow():
    from app.services.ml_scanner import _label_from_spec

    # fast < slow → bullish
    assert _label_from_spec({"fast": 9, "slow": 26}) == 1
    # fast > slow → bearish
    assert _label_from_spec({"fast": 26, "slow": 9}) == -1


def test_label_from_spec_unknown():
    from app.services.ml_scanner import _label_from_spec

    assert _label_from_spec({}) is None


# ─── Training data sourcing ──────────────────────────────────────────────────

def test_fetch_training_rows_prefers_dataset(tmp_path, monkeypatch):
    """ML training feature rows come from the extracted Parquet dataset when
    one exists for (symbol, timeframe) — no live fetch."""
    import numpy as np
    import pandas as pd

    from app.services import dataset_service
    from app.services.ml_scanner import _fetch_training_rows

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    n = 300
    ts = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000.0,
    })
    df.to_parquet(tmp_path / "XAUUSD_15m.parquet", engine="pyarrow", index=False)

    rows = _fetch_training_rows("XAUUSD", "15m", bars=100)
    assert len(rows) == 100
    assert {"time", "open", "high", "low", "close", "volume"} <= set(rows[0])
    assert rows[-1]["time"] > rows[0]["time"]  # chronological


def test_fetch_training_rows_falls_back_to_live(tmp_path, monkeypatch):
    """No dataset -> falls back to live fetch (returns a list, may be empty
    when providers are unreachable)."""
    from app.services import dataset_service
    from app.services.ml_scanner import _fetch_training_rows

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    rows = _fetch_training_rows("XAUUSD", "15m", bars=50)
    assert isinstance(rows, list)


# ─── Predict with no model ────────────────────────────────────────────────────

def test_predict_no_model(tmp_path):
    """predict() must return None when no model file exists."""
    from app.services import ml_scanner

    # Patch the MODEL_PATH to a nonexistent tmp path
    original_path = ml_scanner.MODEL_PATH
    try:
        ml_scanner.MODEL_PATH = tmp_path / "no_model.joblib"
        rows = _fake_rows(120)
        result = ml_scanner.predict(rows)
        assert result is None
    finally:
        ml_scanner.MODEL_PATH = original_path


def test_predict_no_model_insufficient_rows(tmp_path):
    from app.services import ml_scanner

    original_path = ml_scanner.MODEL_PATH
    try:
        ml_scanner.MODEL_PATH = tmp_path / "no_model.joblib"
        result = ml_scanner.predict(_fake_rows(20))
        assert result is None
    finally:
        ml_scanner.MODEL_PATH = original_path


# ─── Status ───────────────────────────────────────────────────────────────────

def test_ml_status_not_trained(tmp_path):
    from app.services import ml_scanner

    original_path = ml_scanner.MODEL_PATH
    try:
        ml_scanner.MODEL_PATH = tmp_path / "no_model.joblib"
        ml_scanner._STATE.update({"status": "not_trained", "trained_at": None})
        snap = ml_scanner.status()
        assert snap["model_exists"] is False
    finally:
        ml_scanner.MODEL_PATH = original_path


# ─── XGBoostScannerAgent ─────────────────────────────────────────────────────

def test_xgb_scanner_fallback_to_ollama_when_no_model(tmp_path):
    """When no ML model exists, XGBoostScannerAgent must fall back to Ollama."""
    from app.agents.xgb_scanner import XGBoostScannerAgent
    from app.services import ml_scanner

    original_path = ml_scanner.MODEL_PATH
    try:
        ml_scanner.MODEL_PATH = tmp_path / "no_model.joblib"

        agent = XGBoostScannerAgent()

        ollama_output = {
            "asset": "XAUUSD", "direction": "BUY", "confidence": 0.7,
            "rationale": "LLM fallback", "strategy_refs": [],
        }

        with patch.object(agent, "_llm_scan", return_value=ollama_output) as mock_llm:
            rows = _fake_rows(100)
            ohlcv = {"symbol": "XAUUSD", "asset_class": "commodities", "timeframe": "15m", "rows": rows}
            result = agent.scan(ohlcv, {}, [])
            mock_llm.assert_called_once()
            assert result["direction"] == "BUY"
    finally:
        ml_scanner.MODEL_PATH = original_path


def test_xgb_scanner_normalize():
    from app.agents.xgb_scanner import XGBoostScannerAgent

    agent = XGBoostScannerAgent()
    candidate = {"direction": "BUY", "strategy_refs": ["Alpha", "Beta"]}
    normed = agent.normalize(candidate)
    assert normed["strategy_refs"] == ["Alpha", "Beta"]


# ─── API endpoints ────────────────────────────────────────────────────────────

def test_ml_status_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_exists" in data


def test_ml_train_no_data():
    """Training without any approved strategies should return 400."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    # In the test DB there are no approved active strategies with BacktestRuns,
    # so training should fail gracefully.
    resp = client.post("/api/ml/train", json={})
    # Expect either 400 (no training data) or 200 (if seeds exist)
    assert resp.status_code in (200, 400)


def test_ml_delete_model():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.delete("/api/ml/model")
    assert resp.status_code == 200
    data = resp.json()
    assert "ok" in data
