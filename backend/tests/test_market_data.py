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

    async def fake_yahoo(self, symbol, timeframe, limit, asset_class=None):
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

    async def fake_yahoo(self, symbol, timeframe, limit, asset_class=None):
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


def test_crypto_classification_bare_and_clipped_bases():
    # Bare bases and bases that end in a stripped character must NOT be clipped
    # (the old rstrip("USDT") bug turned BTC/LTC/ETC into BT/LT/ET).
    assert md._is_crypto("BTC") is True
    assert md._is_crypto("LTCUSDT") is True
    assert md._is_crypto("ETCUSDT") is True
    assert md._is_crypto("ETH") is True


def test_yahoo_symbol_crypto_base_not_clipped():
    assert md.yahoo_symbol("BTCUSDT") == "BTC-USD"
    assert md.yahoo_symbol("BTCUSD") == "BTC-USD"
    assert md.yahoo_symbol("BTC") == "BTC-USD"
    assert md.yahoo_symbol("LTCUSDT") == "LTC-USD"
    assert md.yahoo_symbol("ETCUSDT") == "ETC-USD"
    assert md.yahoo_symbol("ETHUSDT") == "ETH-USD"


def test_yahoo_symbol_fx_and_metals_keep_x():
    assert md.yahoo_symbol("EURUSD") == "EURUSD=X"
    assert md.yahoo_symbol("GBPJPY") == "GBPJPY=X"
    assert md.yahoo_symbol("XAUUSD") == "XAUUSD=X"


def test_yahoo_symbol_stocks_left_bare():
    # Stocks / ETFs must NOT get the FX "=X" suffix (AAPL=X is a dead pair).
    assert md.yahoo_symbol("AAPL") == "AAPL"
    assert md.yahoo_symbol("SPY") == "SPY"
    assert md.yahoo_symbol("TSLA") == "TSLA"


def test_yahoo_symbol_asset_class_hints():
    assert md.yahoo_symbol("AAPL", "stock") == "AAPL"
    assert md.yahoo_symbol("EURUSD", "forex") == "EURUSD=X"
    assert md.yahoo_symbol("BTCUSDT", "crypto") == "BTC-USD"


def test_binance_symbol_bare_base_defaults_usdt():
    assert md.binance_symbol("BTCUSDT") == "BTCUSDT"
    assert md.binance_symbol("BTC") == "BTCUSDT"
    assert md.binance_symbol("ETH") == "ETHUSDT"
    assert md.binance_symbol("XAUUSD") == "XAUUSD"


def test_binance_only_quotes_separated_from_cfd():
    # BTCUSDT is Binance-only; BTCUSD is a plain-USD CFD served by MT5.
    assert md._binance_only("BTCUSDT") is True
    assert md._binance_only("ETHUSDC") is True
    assert md._binance_only("SOLBUSD") is True
    assert md._binance_only("BTCUSD") is False
    assert md._binance_only("BTC") is False
    assert md._binance_only("XAUUSD") is False


def test_plain_usd_crypto_detection():
    assert md._plain_usd_crypto("BTCUSD") is True
    assert md._plain_usd_crypto("ETHUSD") is True
    assert md._plain_usd_crypto("SOLUSD") is True
    assert md._plain_usd_crypto("BTCUSDT") is False
    assert md._plain_usd_crypto("BTC") is False
    assert md._plain_usd_crypto("XAUUSD") is False  # XAU base not a crypto coin
    assert md._plain_usd_crypto("EURUSD") is False


def test_binance_routed_decision():
    assert md._binance_routed("BTCUSDT") is True
    assert md._binance_routed("ETHUSDC") is True
    assert md._binance_routed("BTC") is True  # bare base → Binance BTCUSDT
    assert md._binance_routed("BTCUSD") is False  # MT5 CFD first
    assert md._binance_routed("XAUUSD") is False
    assert md._binance_routed("BTCUSD", "crypto") is True  # explicit hint wins


def test_btcusd_fetches_mt5_first(monkeypatch):
    """BTCUSD is a broker CFD → the local MT5 terminal must be tried first."""
    svc = MarketDataService()
    order = []

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return FIXED, "mt5"

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return [], "binance"

    async def fake_yahoo(self, symbol, timeframe, limit, asset_class=None):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)
    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("BTCUSD", "15m", 10))
    assert order == ["mt5"]
    assert out["provider"] == "mt5"


def test_btcusd_mt5_empty_falls_back_to_binance_then_yahoo(monkeypatch):
    svc = MarketDataService()
    order = []

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return [], "mt5"

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return FIXED, "binance"

    async def fake_yahoo(self, symbol, timeframe, limit, asset_class=None):
        order.append("yahoo")
        return FIXED, "yahoo"

    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)
    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)
    monkeypatch.setattr(MarketDataService, "_yahoo", fake_yahoo)

    out = run(svc.fetch("BTCUSD", "15m", 10))
    assert order == ["mt5", "binance"]
    assert out["provider"] == "binance"


def test_btcusdt_fetches_binance_never_mt5(monkeypatch):
    """BTCUSDT exists only on Binance — MT5 must never be consulted."""
    svc = MarketDataService()
    order = []

    async def fake_binance(self, symbol, timeframe, limit):
        order.append("binance")
        return FIXED, "binance"

    async def fake_mt5(self, symbol, timeframe, limit):
        order.append("mt5")
        return [], "mt5"

    monkeypatch.setattr(MarketDataService, "_binance", fake_binance)
    monkeypatch.setattr(MarketDataService, "_mt5", fake_mt5)

    out = run(svc.fetch("BTCUSDT", "15m", 10))
    assert order == ["binance"]
    assert "mt5" not in order
    assert out["provider"] == "binance"


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