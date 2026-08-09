from datetime import datetime, timezone
import pathlib
import pytest
from app.services.dataset_service import DatasetService


def test_dataset_service_extract_and_list(tmp_path: pathlib.Path):
    svc = DatasetService(storage_dir=tmp_path)

    # Exact date window (not a bar count) — extraction honors the range.
    meta = svc.extract_dataset(
        symbol="XAUUSD", timeframe="15m",
        start_date="2023-01-01", end_date="2023-02-01",
    )

    assert meta["symbol"] == "XAUUSD"
    assert meta["timeframe"] == "15m"
    assert meta["bar_count"] > 0
    assert meta["start_time"] >= "2023-01-01"
    assert meta["end_time"] <= "2023-02-01T23:59:59Z"
    assert meta["file_size_bytes"] > 0
    assert pathlib.Path(meta["file_path"]).exists()

    # List datasets
    datasets = svc.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["id"] == "XAUUSD_15m"
    assert datasets[0]["bar_count"] == meta["bar_count"]


def test_dataset_service_preview_and_delete(tmp_path: pathlib.Path):
    svc = DatasetService(storage_dir=tmp_path)
    meta = svc.extract_dataset(
        symbol="EURUSD", timeframe="1h",
        start_date="2023-01-01", end_date="2023-01-15",
    )
    dataset_id = meta["id"]

    # Preview
    preview = svc.preview_dataset(dataset_id, limit=10)
    assert preview["id"] == dataset_id
    assert "stats" in preview
    assert preview["stats"]["bar_count"] == meta["bar_count"]
    assert len(preview["sample"]) <= 10

    # Delete
    deleted = svc.delete_dataset(dataset_id)
    assert deleted is True
    assert not (tmp_path / f"{dataset_id}.parquet").exists()

    # Confirm delete non-existent
    deleted_again = svc.delete_dataset(dataset_id)
    assert deleted_again is False


def test_resolve_range_defaults_and_validation():
    # Missing start defaults to ~2 years before the given end.
    start, end = DatasetService._resolve_range(None, "2023-01-01")
    assert end.isoformat().startswith("2023-01-01")
    assert (end - start).days <= 730

    # Inverted range is rejected loudly.
    with pytest.raises(ValueError, match="start_date"):
        DatasetService._resolve_range("2024-01-01", "2023-01-01")

    # Bad formats are rejected with a clear message.
    with pytest.raises(ValueError, match="Invalid start_date"):
        DatasetService._resolve_range("not-a-date", "2023-01-01")


def test_extract_watchlist_collects_per_symbol_errors(tmp_path: pathlib.Path, monkeypatch):
    svc = DatasetService(storage_dir=tmp_path)

    def fake_extract(symbol, *a, **k):
        if symbol == "GOOD":
            return {"symbol": "GOOD", "bar_count": 10}
        raise ValueError("no data for BAD")

    monkeypatch.setattr(svc, "extract_dataset", fake_extract)
    res = svc.extract_watchlist(symbols=["GOOD", "BAD"])
    assert len(res["datasets"]) == 1
    assert res["datasets"][0]["symbol"] == "GOOD"
    assert res["errors"] == [{"symbol": "BAD", "error": "no data for BAD"}]


def test_extract_watchlist_requires_symbols():
    with pytest.raises(ValueError, match="watchlist"):
        DatasetService().extract_watchlist(symbols=[])


def test_fetch_ohlcv_collects_provider_errors(tmp_path: pathlib.Path, monkeypatch):
    """Every failed provider (MT5 + fallback) is reported in diagnostics, and
    no rows are returned when the whole chain is down."""
    from datetime import datetime, timezone

    from app.services import dataset_service

    svc = DatasetService(storage_dir=tmp_path)

    class FakeMT5:
        def pull_ohlcv_range(self, *a, **k):
            raise RuntimeError("MT5 terminal init failed")

    class FakeMD:
        def __init__(self):
            pass

        async def fetch_range(self, *a, **k):
            return [], "yahoo"

        async def close(self):
            pass

    monkeypatch.setattr(dataset_service, "MT5Adapter", FakeMT5)
    monkeypatch.setattr(dataset_service, "MarketDataService", FakeMD)

    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 2, 1, tzinfo=timezone.utc)
    rows, provider, diag = svc._fetch_ohlcv("EURUSD", "15m", start, end, "")
    assert rows == []
    assert provider is None
    providers = {d["provider"] for d in diag}
    assert providers == {"mt5", "yahoo"}
    assert any("MT5 terminal init failed" in d["error"] for d in diag)


def test_fetch_ohlcv_binance_only_skips_mt5(tmp_path: pathlib.Path, monkeypatch):
    """BTCUSDT is a Binance-only quote — the local MT5 terminal must never be
    consulted for it, even when a crypto asset_class hint is absent."""
    from datetime import datetime, timezone

    from app.services import dataset_service

    svc = DatasetService(storage_dir=tmp_path)
    called = []

    class ExplodingMT5:
        def pull_ohlcv_range(self, *a, **k):
            called.append("mt5")
            raise AssertionError("MT5 must not be queried for BTCUSDT")

    class FakeMD:
        def __init__(self):
            pass

        async def fetch_range(self, *a, **k):
            called.append("binance")
            return [], "binance"

        async def close(self):
            pass

    monkeypatch.setattr(dataset_service, "MT5Adapter", ExplodingMT5)
    monkeypatch.setattr(dataset_service, "MarketDataService", FakeMD)

    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 2, 1, tzinfo=timezone.utc)
    rows, provider, diag = svc._fetch_ohlcv("BTCUSDT", "15m", start, end, "")
    assert rows == []
    assert provider is None
    assert "mt5" not in called
    providers = {d["provider"] for d in diag}
    assert providers == {"binance"}
    """extract_dataset must raise a ValueError listing which providers failed
    and why — not a silent generic message."""
    svc = DatasetService(storage_dir=tmp_path)

    def fake_fail(symbol, timeframe, start, end, asset_class):
        return [], None, [
            {"provider": "mt5", "ok": False, "error": "terminal down"},
            {"provider": "yahoo", "ok": False, "error": "no candles in range"},
        ]

    monkeypatch.setattr(svc, "_fetch_ohlcv", fake_fail)
    with pytest.raises(ValueError, match="mt5: terminal down.*yahoo: no candles"):
        svc.extract_dataset("EURUSD", "15m", "2023-01-01", "2023-02-01")


def test_extract_dataset_includes_provider_and_warnings(tmp_path, monkeypatch):
    """Successful extraction records the active provider and any provider
    warnings (e.g. MT5 unavailable, data came from the live fallback)."""
    svc = DatasetService(storage_dir=tmp_path)
    rows = [
        {"time": 1700000000 + i * 900, "open": 100.0 + i, "high": 102.0 + i,
         "low": 99.0 + i, "close": 101.0 + i, "volume": 100.0}
        for i in range(20)
    ]

    def fake_ok(symbol, timeframe, start, end, asset_class):
        return rows, "mt5", [
            {"provider": "yahoo", "ok": False, "error": "yahoo unavailable"},
        ]

    monkeypatch.setattr(svc, "_fetch_ohlcv", fake_ok)
    meta = svc.extract_dataset("GBPUSD", "15m", "2023-11-14", "2023-11-15")
    assert meta["symbol"] == "GBPUSD"
    assert meta["timeframe"] == "15m"
    assert meta["provider"] == "mt5"
    assert meta["warnings"] == [
        {"provider": "yahoo", "ok": False, "error": "yahoo unavailable"},
    ]
    assert meta["bar_count"] > 0
    assert pathlib.Path(meta["file_path"]).exists()


def test_read_dataset_candles_slices_and_windows(tmp_path: pathlib.Path, monkeypatch):
    """Extracted datasets read back as OHLCV rows (backtest / ML input), with
    tail-bar and exact-date-window slicing; missing files return None."""
    from app.services import dataset_service

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    svc = DatasetService(storage_dir=tmp_path)
    meta = svc.extract_dataset(
        symbol="XAUUSD", timeframe="15m",
        start_date="2023-01-01", end_date="2023-02-01",
    )

    # Whole file
    rows = dataset_service.read_dataset_candles("xauusd", "15m")
    assert rows is not None and len(rows) == meta["bar_count"]
    assert {"time", "open", "high", "low", "close", "volume"} <= set(rows[0])

    # Most-recent N rows
    tail = dataset_service.read_dataset_candles("XAUUSD", "15m", bars=100)
    assert tail is not None and len(tail) == 100

    # Exact window
    win = dataset_service.read_dataset_candles(
        "XAUUSD", "15m",
        start=datetime(2023, 1, 15, tzinfo=timezone.utc),
        end=datetime(2023, 1, 31, 23, 59, tzinfo=timezone.utc),
    )
    assert win
    assert all(datetime.fromtimestamp(r["time"], tz=timezone.utc).date().month == 1
               for r in win)

    # Missing dataset -> None (caller falls back to live data)
    assert dataset_service.read_dataset_candles("NOPE", "15m") is None


def test_read_dataset_candles_aliases_timeframe_spelling(tmp_path: pathlib.Path, monkeypatch):
    """A dataset stored with MT5-style timeframe (``M15``) is found when the
    backtest asks for the plain spelling (``15m``), and vice versa."""
    from app.services import dataset_service

    monkeypatch.setattr(dataset_service, "DATASETS_DIR", tmp_path)
    svc = DatasetService(storage_dir=tmp_path)
    meta = svc.extract_dataset(
        symbol="EURUSD", timeframe="M15",
        start_date="2023-01-01", end_date="2023-02-01",
    )
    assert meta["id"] == "EURUSD_M15"
    assert (tmp_path / "EURUSD_M15.parquet").exists()

    # Plain spelling resolves the MT5-styled file.
    rows = dataset_service.read_dataset_candles("EURUSD", "15m")
    assert rows is not None and len(rows) == meta["bar_count"]

    # Unknown timeframe for an existing symbol -> None.
    assert dataset_service.read_dataset_candles("EURUSD", "1h") is None


def test_dataset_api_endpoints(tmp_path: pathlib.Path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # Mock DatasetService to write to tmp_path
    monkeypatch.setattr("app.services.dataset_service.DATASETS_DIR", tmp_path)

    # 1. POST /api/datasets/extract with an explicit symbol + exact dates.
    resp = client.post(
        "/api/datasets/extract",
        json={
            "symbol": "BTCUSD", "timeframe": "1h",
            "start_date": "2023-01-01", "end_date": "2023-02-01",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["datasets"]) == 1 and data["errors"] == []
    meta = data["datasets"][0]
    assert meta["symbol"] == "BTCUSD"
    assert meta["bar_count"] > 0
    dataset_id = meta["id"]

    # 2. Watchlist-wide extraction (no symbol): every asset is extracted, and
    # per-symbol failures are returned as errors instead of a hard 400.
    client.post("/api/assets", json={
        "id": "ast-wl", "symbol": "ZZZWATCH", "asset_class": "test",
        "timeframe_watchlist": "15m",
    })
    resp = client.post("/api/datasets/extract", json={
        "timeframe": "1h", "start_date": "2023-01-01", "end_date": "2023-01-15",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["datasets"], list) and isinstance(body["errors"], list)
    assert any(e["symbol"] == "ZZZWATCH" for e in body["errors"]), (
        "unreachable watchlist symbol must be reported as an error entry")

    # 3. GET /api/datasets
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    datasets = resp.json()["datasets"]
    assert len(datasets) >= 1

    # 4. GET /api/datasets/{dataset_id}
    resp = client.get(f"/api/datasets/{dataset_id}")
    assert resp.status_code == 200
    preview = resp.json()
    assert preview["id"] == dataset_id

    # 5. DELETE /api/datasets/{dataset_id}
    resp = client.delete(f"/api/datasets/{dataset_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # 6. Inverted dates are a clean 400.
    resp = client.post("/api/datasets/extract", json={
        "symbol": "BTCUSD", "timeframe": "1h",
        "start_date": "2024-01-01", "end_date": "2023-01-01",
    })
    assert resp.status_code == 400
