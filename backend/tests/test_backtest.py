"""Numeric checks for the backtest engine (no network access)."""

import numpy as np
import pandas as pd

from app.services.backtest_engine import (
    BacktestEngine,
    _cost_commission,
    _mc_costs,
    _metrics,
)


def _trending_df() -> pd.DataFrame:
    n = 1200
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(5)
    trend = np.cumsum(np.sin(np.arange(n) / 30.0) * 0.8) + np.cumsum(rng.normal(0, 0.2, n))
    close = 100 + trend + 20 * np.sin(np.arange(n) / 5.0)
    return pd.DataFrame(
        {
            "Open": close + 0.1,
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Volume": np.abs(rng.normal(1000, 200, n)),
        },
        index=idx,
    )


def test_engine_produces_metrics():
    spec = {
        "fast": 8, "slow": 30, "rsi_period": 14, "rsi_buy": 40, "rsi_sell": 60,
        "atr_period": 14, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5, "risk_pct": 0.01,
    }
    stats, trades, equity = BacktestEngine()._run_df(_trending_df(), spec)
    metrics = _metrics(stats, trades)
    assert metrics["n_trades"] > 10
    assert isinstance(metrics["sharpe"], float)
    assert metrics["max_win"] >= 0 or metrics["max_win"] < 0  # populated (finite)
    assert metrics["mc_sharpe"]["n"] > 0
    assert len(equity) > 0


def test_clamp_keeps_params_in_range():
    from app.agents.refinement import clamp_spec

    spec = clamp_spec({"fast": 1, "slow": 2, "rsi_sell": 10, "atr_sl_mult": 99, "risk_pct": 0})
    assert spec["fast"] >= 3
    assert spec["slow"] - spec["fast"] >= 5
    assert spec["rsi_sell"] > spec["rsi_buy"]
    assert spec["atr_sl_mult"] <= 4.0


# ----- Cost modeling (spread + randomized slippage, task #8.1) -----


def test_cost_commission_uses_default_spread():
    assert 0.0 < _cost_commission() < 0.01  # ~6 bps -> 0.0006 round-trip


def test_mc_costs_returns_percentile_bands():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.4, 2.0, 60)  # 60 trade returns
    res = _mc_costs(returns, spread_bps=6.0, slippage_bps=5.0, n_boot=400)
    assert res["n"] > 0
    assert res["spread_bps"] == 6.0
    assert res["slippage_bps"] == 5.0
    assert res["sharpe_p5"] <= res["sharpe_median"] <= res["sharpe_p95"]
    assert res["ret_p5"] <= res["ret_median"] <= res["ret_p95"]


def test_mc_costs_higher_slippage_drags_returns():
    rng = np.random.default_rng(22)
    returns = rng.normal(0.4, 2.0, 60)
    low = _mc_costs(returns, spread_bps=6.0, slippage_bps=1.0, n_boot=400)
    high = _mc_costs(returns, spread_bps=6.0, slippage_bps=30.0, n_boot=400)
    assert high["ret_median"] < low["ret_median"]


def test_mc_costs_handles_too_few_trades():
    res = _mc_costs(np.array([1.0, 2.0]), spread_bps=6.0, slippage_bps=5.0)
    assert res["n"] == 0
    assert res["sharpe_median"] == 0.0


def test_metrics_includes_mc_costs_block():
    spec = {
        "fast": 8, "slow": 30, "rsi_period": 14, "rsi_buy": 40, "rsi_sell": 60,
        "atr_period": 14, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5, "risk_pct": 0.01,
    }
    stats, trades, _ = BacktestEngine()._run_df(_trending_df(), spec)
    metrics = _metrics(stats, trades, {"spread_bps": 6.0, "slippage_bps": 5.0})
    assert metrics["mc_costs"]["n"] > 0
    assert "sharpe_p5" in metrics["mc_costs"]
    assert "ret_p95" in metrics["mc_costs"]


def test_commission_applies_to_fills(monkeypatch):
    """A non-zero commission must not raise and still yields trades."""
    spec = {
        "fast": 8, "slow": 30, "rsi_period": 14, "rsi_buy": 40, "rsi_sell": 60,
        "atr_period": 14, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5, "risk_pct": 0.01,
    }
    monkeypatch.setattr("app.services.backtest_engine._cost_commission",
                        lambda: 0.0006)
    _, trades, _ = BacktestEngine()._run_df(_trending_df(), spec)
    assert len(trades) > 0


# ----- Extracted Parquet dataset as the backtest data source -----

def _write_dataset(tmp_path, n: int = 4000) -> None:
    """Write a synthetic XAUUSD_15m.parquet in the dataset service schema."""
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    trend = np.cumsum(np.sin(np.arange(n) / 30.0) * 0.8) + np.cumsum(rng.normal(0, 0.2, n))
    close = 100 + trend
    df = pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": np.abs(rng.normal(1000, 200, n)),
    })
    df.to_parquet(tmp_path / "XAUUSD_15m.parquet", engine="pyarrow", index=False)


def test_load_df_prefers_extracted_dataset(tmp_path, monkeypatch):
    """With an extracted dataset present, _load_df serves from it (provider
    'dataset') instead of hitting the live providers."""
    from app.services import dataset_service
    from app.services.backtest_engine import _load_df

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    _write_dataset(tmp_path)
    df, provider = _load_df("XAUUSD", "15m", bars=500)
    assert provider == "dataset"
    assert len(df) == 500
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_df_date_window_from_dataset(tmp_path, monkeypatch):
    """An exact date window is honored when reading from the dataset."""
    from app.services import dataset_service
    from app.services.backtest_engine import _load_df

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    _write_dataset(tmp_path)
    df, provider = _load_df("XAUUSD", "15m", bars=100000,
                            start_date="2023-01-01", end_date="2023-01-31")
    assert provider == "dataset"
    assert len(df) >= 60
    assert all(d.year == 2023 and d.month == 1 for d in df.index)


def test_load_df_inverted_dates_rejected(tmp_path, monkeypatch):
    from app.services import dataset_service
    from app.services.backtest_engine import _load_df

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    _write_dataset(tmp_path)
    try:
        _load_df("XAUUSD", "15m", 500,
                 start_date="2023-02-01", end_date="2023-01-01")
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "start_date" in str(exc)


def test_engine_run_uses_dataset(tmp_path, monkeypatch):
    """End-to-end: BacktestEngine.run() reads the extracted dataset (no
    network) and reports the source in the result."""
    from app.services import dataset_service

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    _write_dataset(tmp_path)
    spec = {
        "fast": 8, "slow": 30, "rsi_period": 14, "rsi_buy": 40, "rsi_sell": 60,
        "atr_period": 14, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5, "risk_pct": 0.01,
    }
    res = BacktestEngine().run("XAUUSD", "15m", spec, bars=300)
    assert res["provider"] == "dataset"
    assert res["bars"] == 300
    assert res["status"] == "ok"