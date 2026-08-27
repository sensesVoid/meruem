"""Tests for QiMera Data Layer, Regime Contract, and Feature Library (M1)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from qimera.app.regime_contract import compute_historical_regimes
from qimera.app.data_layer import (
    tag_session_boundaries,
    run_quality_gates,
    ingest_market_data,
    load_market_data,
    DATA_ROOT,
    DataQualityGateError
)
from qimera.app.feature_library.normalization import minmax_scale, tanh_scale, sigmoid_scale
from qimera.app.feature_library.primitives import (
    compute_atr,
    atr_normalized,
    rolling_volume_norm,
    swing_pivots,
    price_momentum_norm
)
from qimera.app.feature_library.building_blocks import (
    detect_fvgs,
    compute_anchored_vwap,
    detect_liquidity_sweeps,
    order_blocks_proximity
)
from qimera.app.feature_library.registry import MetadataRegistry


@pytest.fixture
def sample_ohlcv():
    """Generates synthetic price and volume data for testing."""
    np.random.seed(42)
    n = 300
    times = np.arange(1690000000, 1690000000 + n * 900, 900)  # 15m interval
    
    # Random walk close price
    close = 1.09 + np.cumsum(np.random.normal(0, 0.001, n))
    open_p = close + np.random.normal(0, 0.0005, n)
    high = np.maximum(open_p, close) + np.random.uniform(0, 0.001, n)
    low = np.minimum(open_p, close) - np.random.uniform(0, 0.001, n)
    volume = np.random.uniform(10, 1000, n)
    
    candles = [
        {
            "time": int(times[i]),
            "open": float(open_p[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i])
        }
        for i in range(n)
    ]
    return candles


def test_regime_contract(sample_ohlcv):
    regimes = compute_historical_regimes(sample_ohlcv)
    assert len(regimes) == len(sample_ohlcv)
    
    # First 30 candles should be UNDEFINED due to lookback requirements
    assert all(r["regime_label"] == "UNDEFINED" for r in regimes[:30])
    
    # Verify that later ones have regime labels and confidence scores
    for r in regimes[30:]:
        assert "regime_label" in r
        assert "_" in r["regime_label"] or r["regime_label"] == "UNDEFINED"
        assert 0.0 <= r["confidence_score"] <= 1.0


def test_data_layer_processing(sample_ohlcv):
    df = pd.DataFrame(sample_ohlcv)
    
    # Test session boundary tagging
    boundaries = tag_session_boundaries(df, "15m")
    assert len(boundaries) == len(df)
    assert boundaries.iloc[0] == True  # first element always true
    
    # Test quality gates with sufficient data
    stats = run_quality_gates(df, "15m")
    assert stats["total_bars"] == 300
    assert stats["gaps_count"] == 0
    
    # Test quality gates with insufficient data
    short_df = df.iloc[:100]
    with pytest.raises(DataQualityGateError):
        run_quality_gates(short_df, "15m", min_bars=240)


def test_normalization_utils():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    
    scaled_minmax = minmax_scale(series)
    assert scaled_minmax.min() == pytest.approx(-1.0)
    assert scaled_minmax.max() == pytest.approx(1.0)
    
    scaled_tanh = tanh_scale(series)
    assert (scaled_tanh >= -1.0).all() and (scaled_tanh <= 1.0).all()
    
    scaled_sig = sigmoid_scale(series)
    assert (scaled_sig >= -1.0).all() and (scaled_sig <= 1.0).all()


def test_feature_library_primitives(sample_ohlcv):
    df = pd.DataFrame(sample_ohlcv)
    
    atr = compute_atr(df["high"], df["low"], df["close"])
    assert len(atr) == len(df)
    assert (atr >= 0).all()
    
    norm_atr = atr_normalized(df["high"], df["low"], df["close"])
    assert norm_atr.min() >= -1.0 and norm_atr.max() <= 1.0
    
    norm_vol = rolling_volume_norm(df["volume"])
    assert norm_vol.min() >= -1.0 and norm_vol.max() <= 1.0
    
    sh, sl = swing_pivots(df["high"], df["low"])
    assert len(sh) == len(df)
    assert set(sh.unique()).issubset({-1.0, 1.0})
    assert set(sl.unique()).issubset({-1.0, 1.0})
    
    mom = price_momentum_norm(df["close"])
    assert mom.min() >= -1.0 and mom.max() <= 1.0


def test_feature_library_building_blocks(sample_ohlcv):
    df = pd.DataFrame(sample_ohlcv)
    
    fvgs = detect_fvgs(df["high"], df["low"], df["close"])
    assert len(fvgs) == len(df)
    assert fvgs.min() >= -1.0 and fvgs.max() <= 1.0
    
    boundaries = tag_session_boundaries(df, "15m")
    vwap = compute_anchored_vwap(df["high"], df["low"], df["close"], df["volume"], boundaries)
    assert len(vwap) == len(df)
    assert vwap.min() >= -1.0 and vwap.max() <= 1.0
    
    sweeps = detect_liquidity_sweeps(df["high"], df["low"], df["close"])
    assert len(sweeps) == len(df)
    assert sweeps.min() >= -1.0 and sweeps.max() <= 1.0
    
    ob_prox = order_blocks_proximity(df["high"], df["low"], df["close"], df["volume"])
    assert len(ob_prox) == len(df)
    assert ob_prox.min() >= -1.0 and ob_prox.max() <= 1.0


def test_metadata_registry(tmp_path):
    reg_file = tmp_path / "test_registry.json"
    registry = MetadataRegistry(path=reg_file)
    
    # Should start empty
    features = registry.list_features()
    assert len(features) == 0
    
    # Register a feature
    registry.register_feature(
        feature_id="test_feat",
        name="Test Feature",
        feature_type="primitive",
        function_name="some_func",
        parameters={"param1": "int"},
        description="A dummy test feature"
    )
    
    features = registry.list_features()
    assert len(features) == 1
    assert features[0]["id"] == "test_feat"
    
    # Get feature by ID
    feat = registry.get_feature("test_feat")
    assert feat is not None
    assert feat["name"] == "Test Feature"
    
    # Disallow duplicates
    with pytest.raises(ValueError):
        registry.register_feature(
            feature_id="test_feat",
            name="Test Feature 2",
            feature_type="primitive",
            function_name="some_func2",
            parameters={},
            description="Another dummy"
        )
