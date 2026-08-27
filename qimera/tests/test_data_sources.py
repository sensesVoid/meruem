"""Tests for the on-demand data sources (yfinance / MT5) — offline-safe.

No network calls: yfinance is monkeypatched and the MT5 adapter is stubbed, so
the suite runs identically with or without the terminal / internet.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest

from qimera.app import data_layer as dl
from qimera.app import data_sources as ds


def _fake_yf_df(n: int = 120) -> pd.DataFrame:
    """Single-level OHLCV frame shaped like yfinance's output."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.09, 1.11, n)
    return pd.DataFrame({
        "Open": close - 0.0002,
        "High": close + 0.0005,
        "Low": close - 0.0005,
        "Close": close,
        "Volume": np.random.randint(0, 1000, n),
    }, index=idx)


def test_yfinance_fetch_parses_candles(monkeypatch):
    calls: dict = {}

    def fake_download(ticker, **kwargs):
        calls["ticker"] = ticker
        calls["kwargs"] = kwargs
        return _fake_yf_df()

    monkeypatch.setattr("yfinance.download", fake_download)

    candles = ds.fetch_yfinance("EURUSD", timeframe="15m", limit=10)
    assert len(candles) == 10
    assert calls["ticker"] == "EURUSD=X"
    assert calls["kwargs"]["interval"] == "15m"
    assert calls["kwargs"]["multi_level_index"] is False
    c = candles[0]
    assert set(c) == {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(c["time"], int)
    assert candles[-1]["time"] > candles[0]["time"]


def test_yfinance_unknown_interval_returns_empty(monkeypatch):
    # Yahoo has no 4h interval — no network call should even be attempted.
    assert ds.fetch_yfinance("EURUSD", timeframe="4h") == []


def test_yfinance_empty_frame_returns_empty(monkeypatch):
    monkeypatch.setattr("yfinance.download", lambda ticker, **kw: pd.DataFrame())
    assert ds.fetch_yfinance("AAPL", timeframe="1D") == []


def test_yfinance_metal_futures_fallback(monkeypatch):
    calls: list[str] = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return _fake_yf_df() if ticker == "GC=F" else pd.DataFrame()

    monkeypatch.setattr("yfinance.download", fake_download)

    candles = ds.fetch_yfinance("XAUUSD", timeframe="1D")
    # Spot XAUUSD=X is dead on Yahoo -> falls back to the GC=F futures ticker.
    assert calls == ["XAUUSD=X", "GC=F"]
    assert len(candles) > 0


def test_yfinance_nan_volume_zeroed_and_nan_rows_dropped(monkeypatch):
    df = _fake_yf_df(4)
    df.iloc[1, df.columns.get_loc("Volume")] = np.nan      # NaN volume -> 0
    df.iloc[2, df.columns.get_loc("Open")] = np.nan        # partial NaN row -> dropped

    monkeypatch.setattr("yfinance.download", lambda ticker, **kw: df)
    candles = ds.fetch_yfinance("EURUSD", timeframe="15m")

    assert len(candles) == 3                                  # row 2 dropped
    assert all(pd.notna(c["volume"]) for c in candles)       # no NaN volume leaks
    assert candles[1]["volume"] == 0.0                       # NaN volume became 0
    assert all(pd.notna(c["open"]) and pd.notna(c["close"]) for c in candles)


def test_yfinance_tz_naive_daily_index_uses_utc(monkeypatch):
    # Daily data arrives tz-naive; epochs must be UTC midnight, not local time.
    idx = pd.date_range("2026-01-01", periods=3, freq="D")   # tz-naive
    df = pd.DataFrame({
        "Open": [1.0, 1.0, 1.0], "High": [1.1, 1.1, 1.1],
        "Low": [0.9, 0.9, 0.9], "Close": [1.05, 1.05, 1.05],
        "Volume": [100.0, 100.0, 100.0],
    }, index=idx)

    monkeypatch.setattr("yfinance.download", lambda ticker, **kw: df)
    candles = ds.fetch_yfinance("SPY", timeframe="1D")

    expected = int(pd.Timestamp("2026-01-01").tz_localize("UTC").timestamp())
    assert candles[0]["time"] == expected


def test_yfinance_partial_window_raises():
    with pytest.raises(ValueError):
        ds.fetch_yfinance("EURUSD", start=1_700_000_000)
    with pytest.raises(ValueError):
        ds.fetch_yfinance("EURUSD", end=1_700_000_000)


def test_yfinance_date_window_passed_as_epoch_window(monkeypatch):
    seen: dict = {}

    def fake_download(ticker, **kwargs):
        seen.update(kwargs)
        return _fake_yf_df()

    monkeypatch.setattr("yfinance.download", fake_download)
    ds.fetch_yfinance("BTCUSD", timeframe="1h", start=1_700_000_000, end=1_700_086_400)

    assert seen["start"].timestamp() == 1_700_000_000
    assert seen["end"].timestamp() == 1_700_086_400
    assert "period" not in seen


def test_mt5_fetch_graceful_without_terminal(monkeypatch):
    class FakeAdapter:
        def pull_ohlcv(self, symbol, timeframe, count=200):
            return []

        def pull_ohlcv_range(self, symbol, timeframe, start, end):
            return []

    monkeypatch.setattr(ds, "MT5Adapter", FakeAdapter)
    assert ds.fetch_mt5("EURUSD", "15m", limit=100) == []


def test_mt5_range_uses_pull_ohlcv_range(monkeypatch):
    seen: dict = {}

    class FakeAdapter:
        def pull_ohlcv(self, symbol, timeframe, count=200):
            seen["mode"] = "limit"
            return [{"time": 1, "open": 1.0, "high": 1.0, "low": 1.0,
                     "close": 1.0, "volume": 0}]

        def pull_ohlcv_range(self, symbol, timeframe, start, end):
            seen["mode"] = "range"
            seen["start"], seen["end"] = start, end
            return [{"time": 2, "open": 1.0, "high": 1.0, "low": 1.0,
                     "close": 1.0, "volume": 0}]

    monkeypatch.setattr(ds, "MT5Adapter", FakeAdapter)

    candles = ds.fetch_mt5("EURUSD", "15m", start=1000, end=2000)
    assert seen["mode"] == "range"
    assert seen["start"].timestamp() == 1000
    assert seen["end"].timestamp() == 2000
    assert candles[0]["time"] == 2


def test_dispatcher_unknown_source():
    with pytest.raises(ValueError):
        ds.fetch_ohlcv("EURUSD", source="binance")


def test_ingest_market_data_with_yfinance_source(monkeypatch, tmp_path):
    """End-to-end ingestion path: on-demand source -> regime + gates -> parquet."""
    n = 300
    np.random.seed(3)
    times = np.arange(1_690_000_000, 1_690_000_000 + n * 900, 900)
    close = 1.09 + np.cumsum(np.random.normal(0.0001, 0.001, n))
    fake_candles = [
        {
            "time": int(times[i]),
            "open": float(close[i] - 0.0002),
            "high": float(close[i] + 0.0006),
            "low": float(close[i] - 0.0006),
            "close": float(close[i]),
            "volume": 100.0,
        }
        for i in range(n)
    ]

    monkeypatch.setattr(dl, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "fetch_ohlcv", lambda *a, **k: fake_candles)

    stats = asyncio.run(
        dl.ingest_market_data("EURUSD", timeframe="15m", limit=n, source="yfinance"))

    assert stats["symbol"] == "EURUSD"
    parquet = tmp_path / "EURUSD" / "15m" / "data.parquet"
    assert parquet.is_file()
    loaded = pd.read_parquet(parquet)
    assert len(loaded) == n
    assert "regime_label" in loaded.columns
    assert "confidence_score" in loaded.columns
    assert "session_boundary" in loaded.columns


def test_ingest_rejects_unknown_source():
    with pytest.raises(ValueError):
        asyncio.run(dl.ingest_market_data("EURUSD", source="binance"))


# ---------------------------------------------------------------------------
# Deep-history paging + compact output
# ---------------------------------------------------------------------------


class _PagingAdapter:
    """Fake MT5 adapter: returns ``window_bars`` bars for the first ``horizon``
    date windows, then empty — emulating a broker with finite M1 history."""

    def __init__(self, window_bars=10, horizon=3):
        self.window_bars = window_bars
        self.horizon = horizon
        self.calls = 0

    def pull_ohlcv_range(self, symbol, timeframe, start, end):
        self.calls += 1
        if self.calls > self.horizon:
            return []
        t0 = int(start.timestamp())
        return [{"time": t0 + i * 60, "open": 1.0, "high": 1.0, "low": 1.0,
                 "close": 1.0, "volume": 0.0} for i in range(self.window_bars)]


def test_fetch_mt5_deep_pages_and_merges_chronologically(monkeypatch):
    fake = _PagingAdapter(window_bars=10, horizon=3)
    monkeypatch.setattr(ds, "MT5Adapter", lambda: fake)

    candles = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                                chunk_days=1, delay=0, max_empty_windows=1)

    assert fake.calls == 5            # 3 data windows + 1 empty window probed twice
    assert len(candles) == 30
    times = [c["time"] for c in candles]
    assert times == sorted(times)                       # chronological merge
    assert len(set(times)) == 30                        # no overlaps/duplicates


def test_fetch_mt5_deep_stops_after_consecutive_empty_windows(monkeypatch):
    fake = _PagingAdapter(window_bars=10, horizon=1)
    monkeypatch.setattr(ds, "MT5Adapter", lambda: fake)

    candles = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                                chunk_days=1, delay=0, max_empty_windows=3)

    assert fake.calls == 7            # 1 data window + 3 empty windows (each retried)
    assert len(candles) == 10


def test_fetch_mt5_deep_start_bounds_oldest_bar(monkeypatch):
    fake = _PagingAdapter(window_bars=10, horizon=2)
    monkeypatch.setattr(ds, "MT5Adapter", lambda: fake)

    # ``start`` bounds the crawl: the oldest window is clamped to it, so no
    # bar older than ``start`` is ever returned.
    end = 1_700_000_000
    start = end - 2 * 86_400 + 300
    candles = ds.fetch_mt5_deep("EURUSD", "M1", start=start,
                                end=end, chunk_days=1, delay=0)

    assert fake.calls == 2
    assert len(candles) == 20
    assert candles[0]["time"] == start                  # oldest bar == start
    assert all(c["time"] >= start for c in candles)
    assert candles[-1]["time"] < end


def test_fetch_mt5_deep_invalid_window_returns_empty(monkeypatch):
    monkeypatch.setattr(ds, "MT5Adapter", lambda: _PagingAdapter())
    assert ds.fetch_mt5_deep("EURUSD", "M1", start=2_000_000_000,
                             end=1_000_000_000) == []


def test_fetch_mt5_deep_rejects_out_of_range_end(monkeypatch):
    monkeypatch.setattr(ds, "MT5Adapter", lambda: _PagingAdapter())
    with pytest.raises(ValueError):
        ds.fetch_mt5_deep("EURUSD", "M1", end=10**15)
    with pytest.raises(ValueError):
        ds.fetch_mt5_deep("EURUSD", "M1", end=946_684_800 - 1)


def test_fetch_mt5_deep_dedupes_boundary_bars(monkeypatch):
    """copy_rates_range is inclusive of the window's upper bound — a bar exactly
    on a window boundary appears in both adjacent windows and must be deduped."""

    class _BoundaryAdapter:
        def __init__(self):
            self.calls = 0

        def pull_ohlcv_range(self, symbol, timeframe, start, end):
            self.calls += 1
            if self.calls > 2:
                return []
            t0 = int(start.timestamp())
            t_end = int(end.timestamp())
            bars = [{"time": t0 + i * 60, "open": 1.0, "high": 1.0,
                     "low": 1.0, "close": 1.0, "volume": 0.0}
                    for i in range(5)]
            bars.append({"time": t_end, "open": 1.0, "high": 1.0,
                         "low": 1.0, "close": 1.0, "volume": 0.0})
            return bars

    monkeypatch.setattr(ds, "MT5Adapter", _BoundaryAdapter)

    candles = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                                chunk_days=1, delay=0, max_empty_windows=1)
    times = [c["time"] for c in candles]
    assert times == sorted(times)
    assert len(times) == len(set(times))          # no duplicated boundary bar
    # 6 + 6 raw - 1 shared boundary deduped - 1 bar at `end` (end-exclusive)
    assert len(candles) == 10

    result = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                               chunk_days=1, delay=0, max_empty_windows=1,
                               compact=True)
    assert len(result["time"]) == 10
    assert len(result["time"]) == len(set(result["time"].tolist()))


def test_fetch_mt5_deep_respects_max_bars(monkeypatch):
    fake = _PagingAdapter(window_bars=10, horizon=3)
    monkeypatch.setattr(ds, "MT5Adapter", lambda: fake)

    candles = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                                chunk_days=1, delay=0, max_empty_windows=1,
                                max_bars=15)

    assert len(candles) == 15
    assert candles[-1]["time"] < 1_700_000_000    # keeps the most recent bars


def test_fetch_mt5_deep_compact_arrays(monkeypatch):
    fake = _PagingAdapter(window_bars=10, horizon=2)
    monkeypatch.setattr(ds, "MT5Adapter", lambda: fake)

    result = ds.fetch_mt5_deep("EURUSD", "M1", end=1_700_000_000,
                               chunk_days=1, delay=0, compact=True)

    assert isinstance(result, dict)
    assert set(result) == {"time", "open", "high", "low", "close", "volume"}
    assert result["time"].dtype == np.int64
    assert len(result["close"]) == 20
    assert (result["time"] == np.sort(result["time"])).all()
    assert ds.fetch_mt5_deep("EURUSD", "M1", start=2_000_000_000,
                             end=1_000_000_000, compact=True)["close"].size == 0


def test_candles_to_arrays_roundtrip_shape():
    candles = [{"time": 1, "open": 1.0, "high": 1.1, "low": 0.9,
                "close": 1.05, "volume": 3.0}]
    arr = ds.candles_to_arrays(candles)
    assert arr["time"][0] == 1
    assert arr["close"][0] == 1.05


def test_fetch_ohlcv_rejects_deep_yfinance():
    with pytest.raises(ValueError):
        ds.fetch_ohlcv("EURUSD", source="yfinance", deep=True)


def test_candles_to_arrays_empty():
    arr = ds.candles_to_arrays([])
    assert arr["time"].size == 0
    assert arr["time"].dtype == np.int64
    assert arr["close"].dtype == np.float64
    empty = ds.fetch_mt5_deep("EURUSD", "M1", start=2_000_000_000,
                              end=1_000_000_000, compact=True)
    assert empty["close"].size == 0


def test_fetch_ohlcv_dispatches_deep_mt5(monkeypatch):
    seen: dict = {}

    def fake_deep(symbol, timeframe, start, end, compact=False):
        seen["symbol"] = symbol
        seen["compact"] = compact
        return [{"time": 1, "open": 1.0, "high": 1.0, "low": 1.0,
                 "close": 1.0, "volume": 0}]

    monkeypatch.setattr(ds, "fetch_mt5_deep", fake_deep)
    result = ds.fetch_ohlcv("EURUSD", timeframe="M1", start=1, end=2,
                            source="mt5", deep=True, compact=True)
    assert seen == {"symbol": "EURUSD", "compact": True}
    assert len(result) == 1


def test_api_data_ohlcv_compact_converts_arrays(monkeypatch):
    """compact=true returns arrays converted to JSON-safe lists."""
    from fastapi.testclient import TestClient

    from qimera.app.main import app

    def fake_fetch(*args, **kwargs):
        return {
            "time": np.array([1, 2]),
            "open": np.array([1.0, 1.0]),
            "high": np.array([1.0, 1.0]),
            "low": np.array([1.0, 1.0]),
            "close": np.array([1.0, 1.0]),
            "volume": np.array([0.0, 0.0]),
        }

    monkeypatch.setattr(ds, "fetch_ohlcv", fake_fetch)

    with TestClient(app) as client:
        r = client.get("/api/data/ohlcv", params={
            "symbol": "EURUSD", "source": "mt5", "compact": "true"})
        assert r.status_code == 200
        data = r.json()
        assert data["compact"] is True
        assert isinstance(data["candles"], dict)
        assert data["candles"]["time"] == [1, 2]
        assert isinstance(data["candles"]["close"], list)

        # deep is MT5-only
        r2 = client.get("/api/data/ohlcv", params={
            "symbol": "EURUSD", "source": "yfinance", "deep": "true"})
        assert r2.status_code == 400


def test_ingest_persist_false_returns_in_memory_frame(monkeypatch, tmp_path):
    """Lightweight path: fetch + transform, never write parquet."""
    n = 300
    np.random.seed(9)
    times = np.arange(1_690_000_000, 1_690_000_000 + n * 900, 900)
    close = 1.09 + np.cumsum(np.random.normal(0.0001, 0.001, n))
    fake_candles = [
        {"time": int(times[i]), "open": float(close[i] - 0.0002),
         "high": float(close[i] + 0.0006), "low": float(close[i] - 0.0006),
         "close": float(close[i]), "volume": 100.0}
        for i in range(n)
    ]

    monkeypatch.setattr(dl, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "fetch_ohlcv", lambda *a, **k: fake_candles)

    result = asyncio.run(dl.ingest_market_data(
        "EURUSD", timeframe="15m", limit=n, source="yfinance", persist=False))

    assert result["persisted"] is False
    assert len(result["df"]) == n
    assert "regime_label" in result["df"].columns
    assert not list(tmp_path.rglob("*.parquet"))   # nothing written to disk


def test_api_data_ohlcv_endpoint(monkeypatch):
    """GET /api/data/ohlcv serves the chosen source and rejects unknown ones."""
    from fastapi.testclient import TestClient

    from qimera.app.main import app

    monkeypatch.setattr(
        ds, "fetch_ohlcv",
        lambda *a, **k: [{"time": 1, "open": 1.0, "high": 1.0,
                          "low": 1.0, "close": 1.0, "volume": 0}],
    )

    with TestClient(app) as client:
        r = client.get("/api/data/ohlcv",
                       params={"symbol": "EURUSD", "source": "yfinance"})
        assert r.status_code == 200
        data = r.json()
        assert data["provider"] == "yfinance"
        assert len(data["candles"]) == 1

        r2 = client.get("/api/data/ohlcv",
                        params={"symbol": "EURUSD", "source": "binance"})
        assert r2.status_code == 400

        r3 = client.get("/api/data/ohlcv",
                        params={"symbol": "EURUSD", "source": "mt5",
                                "timeframe": "4h", "start": 0, "end": 0})
        assert r3.status_code == 200
        assert r3.json()["provider"] == "mt5"
