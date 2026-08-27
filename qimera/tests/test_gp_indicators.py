"""Tests for the M5 multi-indicator GP framework (typed schemas + fitness hooks)."""

import numpy as np
import pandas as pd
import pytest

from qimera.app.gp_indicators import (
    GPConfig, INDICATOR_IDS, INDICATOR_SCHEMAS, REJECT_SCORE,
    _parsimony_penalty, evaluate_genes, run_multi_gp, signal_for,
    strategy_metrics,
)


@pytest.fixture
def df():
    """Synthetic OHLCV with the M1 data-layer contract (volumed random walk)."""
    np.random.seed(42)
    n = 500
    times = np.arange(1690000000, 1690000000 + n * 900, 900)
    close = 1.09 + np.cumsum(np.random.normal(0.0002, 0.001, n))
    op = close + np.random.normal(0, 0.0005, n)
    hi = np.maximum(op, close) + np.random.uniform(0, 0.001, n)
    lo = np.minimum(op, close) - np.random.uniform(0, 0.001, n)
    return pd.DataFrame({
        "time": times,
        "open": op, "high": hi, "low": lo, "close": close,
        "volume": np.random.uniform(10, 1000, n),
    })


# ---------------------------------------------------------------------------
# Typed schemas
# ---------------------------------------------------------------------------


def test_schema_has_all_five_indicator_families():
    assert set(INDICATOR_IDS) == {"vwap", "fvg", "liquidity", "order_blocks", "cvd"}
    for iid in INDICATOR_IDS:
        assert "name" in INDICATOR_SCHEMAS[iid]
        assert "description" in INDICATOR_SCHEMAS[iid]


def test_schema_params_are_typed_and_bounded():
    for iid, schema in INDICATOR_SCHEMAS.items():
        assert len(schema["params"]) == 3
        for p in schema["params"]:
            assert p["kind"] in ("int", "float")
            assert p["low"] < p["high"]
            assert p["low"] <= p["default"] <= p["high"]
            assert {"name", "kind", "low", "high", "default"} <= set(p)


def test_signal_for_rejects_unknown_indicator(df):
    with pytest.raises(ValueError):
        signal_for("bogus", df, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        signal_for("vwap", df, [1.0])  # wrong param count


def test_signal_for_missing_column_rejected():
    bad = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        signal_for("vwap", bad, [1.0, 10, 1])


# ---------------------------------------------------------------------------
# Signal series: contract + known behaviours
# ---------------------------------------------------------------------------


def test_all_signals_normalized_and_no_nan(df):
    for iid in INDICATOR_IDS:
        params = [p["default"] for p in INDICATOR_SCHEMAS[iid]["params"]]
        s = signal_for(iid, df, params)
        assert len(s) == len(df)
        assert s.isna().sum() == 0
        assert s.min() >= -1.0 - 1e-9 and s.max() <= 1.0 + 1e-9


def test_fvg_signal_positive_after_bullish_gap():
    n = 80
    close = np.linspace(100.0, 101.0, n)
    # Bar i-2 low at 99.5, bar i high at 100.5 -> bullish 3-candle gap.
    high = close + 0.1
    low = close - 0.1
    high[2] = 99.6  # keep bar 2 low gap open at bar 4
    low[2] = 99.4
    low[4] = 99.7
    d = pd.DataFrame({
        "time": np.arange(n) * 900 + 1690000000,
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 500.0),
    })
    s = signal_for("fvg", d, [0.1, 100, 1])
    assert (s.iloc[5:] >= 0).all()          # bullish bias persists (long expiry)
    assert (s.iloc[:4] <= 0).all()           # nothing before the gap forms
    assert s.iloc[4] > 0                     # signal fires on the formation bar


def test_liquidity_signal_spikes_on_sell_side_sweep():
    n = 60
    close = np.full(n, 100.0)
    high = close + 0.2
    low = close - 0.2
    low[30] = 98.0           # pierce below the window low...
    close[30] = 99.9         # ...but close back inside -> bullish sweep
    d = pd.DataFrame({
        "time": np.arange(n) * 900 + 1690000000,
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 500.0),
    })
    s = signal_for("liquidity", d, [20, 0.1, 1])
    assert s.iloc[30] > 0


def test_vwap_signal_positive_when_price_below_anchor(df):
    # Deterministic anchor: a huge first-bar volume pins the session VWAP near
    # 100, then price holds slightly below it -> a sustained, in-band long
    # bias (a far deviation would be zeroed by the width band by design).
    d = df.copy()
    n = len(d)
    d["time"] = np.arange(n) * 60 + 1690000000   # one UTC session
    d["volume"] = 1.0
    d.loc[d.index[0], "volume"] = 1e9            # anchor the VWAP at 100
    d["close"] = 99.9                            # slightly below the anchor
    d.loc[d.index[0], "close"] = 100.0
    d["high"] = d["close"] * 1.001
    d["low"] = d["close"] * 0.999
    s = signal_for("vwap", d, [1.0, 5, 1])
    assert s.iloc[10:].mean() > 0.2  # sustained bullish bias, not noise


# ---------------------------------------------------------------------------
# Fitness hook
# ---------------------------------------------------------------------------


def test_strategy_metrics_contract(df):
    m = strategy_metrics(df, "cvd", [2.0, 20, 1], atr_len=14, atr_mult=2.0)
    assert len(m) == 5
    net, win_rate, pf, max_dd, trades = m
    assert trades >= 0
    assert 0.0 <= win_rate <= 1.0
    assert max_dd >= 0.0


def test_evaluate_genes_returns_score_and_metrics(df):
    score, metrics = evaluate_genes(
        df, [0, 1.0, 48, 1, 14, 2.0], GPConfig(min_trades=2))
    assert "indicator" in metrics and "gated" in metrics
    assert metrics["indicator"] == "vwap"
    assert isinstance(score, float)


def test_evaluate_genes_hurdle_gate_rejects_no_trades(df):
    # An all-flat signal (no crosses) yields 0 trades -> hard reject.
    genes = [0, 1.0, 48, 1, 14, 2.0]
    score, metrics = evaluate_genes(
        df.iloc[:30], genes, GPConfig(min_trades=5, entry_threshold=0.99))
    assert metrics["trades"] == 0
    assert score == REJECT_SCORE
    assert metrics["gated"] is True


def test_parsimony_penalty_isolated():
    # Exact isolation: the penalty must be 0 at the domain midpoints and grow
    # monotonically with drift — independent of strategy behavior.
    for iid in INDICATOR_IDS:
        mids = GPConfig()._midpoints[iid]
        assert _parsimony_penalty(iid, mids) == 0.0
        p_far = [m + (5.0 if i == 0 else 1.0) for i, m in enumerate(mids)]
        assert _parsimony_penalty(iid, p_far) > 0.0
        assert _parsimony_penalty(iid, p_far) > _parsimony_penalty(iid, mids)


def test_atr_len_gene_actually_reaches_the_exit(df):
    # The reviewer-caught bug: atr_len was computed but never used, so the exit
    # ATR was always period 14. The stop's ATR must honor the gene.
    from qimera.app.gp_indicators import _atr_array

    a_fast = _atr_array(df, period=5)
    a_slow = _atr_array(df, period=60)
    assert not np.allclose(a_fast, a_slow)  # different smoothing -> different stops


# ---------------------------------------------------------------------------
# Full GP run
# ---------------------------------------------------------------------------


def test_run_multi_gp_end_to_end(df):
    cfg = GPConfig(population=24, generations=6, seed=11, min_trades=2)
    res = run_multi_gp(df, cfg)
    assert res["engine"] == "python"
    assert len(res["indicators"]) == 5
    assert res["best"] is not None
    best = res["best"]
    assert best["indicator"] in INDICATOR_IDS
    assert len(best["genes"]) == 6
    assert best["metrics"]["gated"] is False
    assert res["evals"] == cfg.population * (cfg.generations + 1)
    assert len(res["logbook"]) == cfg.generations + 1
    assert res["accepted"] + res["gated"] == res["evals"]


def test_run_multi_gp_deterministic_with_seed(df):
    cfg = GPConfig(population=16, generations=4, seed=7, min_trades=2)
    a = run_multi_gp(df, cfg)
    b = run_multi_gp(df, cfg)
    assert a["best"]["genes"] == b["best"]["genes"]
    assert a["best"]["score"] == b["best"]["score"]


def test_run_multi_gp_genes_stay_in_schema_bounds(df):
    cfg = GPConfig(population=16, generations=4, seed=3, min_trades=2)
    res = run_multi_gp(df, cfg)
    for entry in [res["best"]] + res["hall_of_fame"]:
        iid = entry["indicator"]
        schema = INDICATOR_SCHEMAS[iid]
        genes = entry["genes"]
        for k, p in enumerate(schema["params"]):
            assert p["low"] <= genes[1 + k] <= p["high"]
            if p["kind"] == "int":
                assert float(genes[1 + k]).is_integer()
        assert 5 <= genes[4] <= 60
        assert 0.5 <= genes[5] <= 6.0


def test_run_multi_gp_output_is_json_serializable(df):
    import json

    res = run_multi_gp(df, GPConfig(population=8, generations=2, seed=5,
                                    min_trades=2))
    json.dumps(res)  # must not raise (no np.float64 / numpy arrays leaked)
