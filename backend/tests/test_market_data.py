import asyncio

import pytest

import app.services.market_data as md
from app.services.market_data import MarketDataService

FIXED = [{"time": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}]


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def clear_cache():
    md._cache.clear()
    md._cache_provider.clear()
    yield
    md._cache.clear()
    md._cache_provider.clear()


def test_forex_uses_mt5_first_then_yahoo(monkeypatch):
    svc = MarketDataService()
    order = []

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return FIXED, "mt5"

    async def fake_yahoo(self, symbol, timeframe, limit):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("EURUSD", "4h", 10))
    assert order == ["mt5"]
    assert out["provider"] == "mt5"
    assert out["candles"] == FIXED


def test_forex_falls_back_to_yahoo_when_mt5_empty(monkeypatch):
    svc = MarketDataService()
    order = []

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return [], "mt5"

    async def fake_yahoo(self, symbol, timeframe, limit):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("XAUUSD", "4h", 10))
    assert order == ["mt5", "yahoo"]
    assert out["provider"] == "yahoo"


def test_crypto_prefers_binance(monkeypatch):
    svc = MarketDataService()
    order = []

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return FIXED, "binance"

    async def fake_yahoo(self, symbol, timeframe, limit):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("BTCUSDT", "4h", 10))
    assert order == ["binance"]
    assert out["provider"] == "binance"


def test_crypto_falls_back_to_yahoo(monkeypatch):
    svc = MarketDataService()
    order = []

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return [], "binance"

    async def fake_yahoo(self, symbol, timeframe, limit):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("ETHUSDT", "1h", 100))
    assert order == ["binance", "yahoo"]
    assert out["provider"] == "yahoo"


def test_asset_classification():
    assert md._is_crypto("BTCUSDT") is True
    assert md._is_crypto("XAUUSD") is False
    assert md._is_crypto("EURUSD") is False
    assert md._is_crypto("SOLUSD") is True


def test_mt5_timeframe_map_covers_chart_labels():
    from app.services.mt5_adapter import TIMEFRAME_MAP

    for label in ("M1", "5m", "15m", "1h", "4h", "1D", "1W", "1M"):
        assert label in TIMEFRAME_MAP, f"missing MT5 timeframe mapping: {label}"


def test_mt5_prefers_mt5_for_metals_commodities(monkeypatch):
    # XAGUSD (silver, a commodity) must route to MT5, not Binance.
    svc = MarketDataService()
    order = []

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return FIXED, "mt5"

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return FIXED, "binance"

    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)
    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)

    out = run(svc.fetch("XAGUSD", "4h", 10))
    assert order == ["mt5"]
    assert out["provider"] == "mt5"