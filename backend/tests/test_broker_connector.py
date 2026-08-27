"""Phase 10 — Universal Trading Platform connector tests.

Covers the broker registry + factory dispatch, the ccxt-based MEXC adapter
(no live network — mocked exchange), and the copier's single dispatch path
falling back to the normalized OrderResult contract.
"""

from unittest.mock import MagicMock, patch

from app.services.broker_adapters import (
    get_broker_adapter,
    keyring_namespace,
    supported_brokers,
)
from app.services.copier import CopierEngine


def test_registry_lists_mt5_only():
    """MT5-only platform: Binance / MEXC brokers are parked (their PUBLIC Binance
    market-data flow still runs via market_data.py)."""
    brokers = supported_brokers()
    assert brokers == ["MT5"]
    assert "MT5" in brokers
    assert "Binance" not in brokers  # parked for now
    assert "MEXC" not in brokers  # parked for now


def test_keyring_namespace_mapping():
    assert keyring_namespace("MT5") == "mt5"
    assert keyring_namespace("Binance") is None  # parked
    assert keyring_namespace("MEXC") is None  # parked
    assert keyring_namespace("cTrader") is None


def test_factory_dispatch_returns_right_adapter_types():
    from app.services.mt5_adapter import MT5Adapter

    assert isinstance(get_broker_adapter("MT5", "12345"), MT5Adapter)
    assert get_broker_adapter("Binance", "key") is None  # parked
    assert get_broker_adapter("MEXC", "key") is None  # parked
    assert get_broker_adapter("cTrader") is None


def test_mexc_adapter_implements_broker_adapter_protocol():
    from app.services.broker_base import BrokerAdapter
    from app.services.mexc_adapter import MEXCAdapter

    assert BrokerAdapter is not None  # protocol module imports cleanly
    # Structural check: the protocol's required methods exist on the adapter.
    for name in ("connect", "connect_stored", "healthcheck", "fetch_balance",
                 "fetch_equity", "pull_ohlcv", "pull_ohlcv_range", "place_order"):
        assert callable(getattr(MEXCAdapter, name)), f"MEXCAdapter missing {name}"


def test_mexc_no_credentials_returns_safe_defaults():
    with patch("app.services.mexc_adapter.VaultService.get", return_value=None):
        from app.services.mexc_adapter import MEXCAdapter

        adapter = MEXCAdapter(username="nobody")
        assert adapter.exchange is None
        assert adapter.fetch_balance()["total"] is None
        assert adapter.fetch_equity() is None
        res = adapter.place_order("BTC/USDT", "BUY", 0.01)
        assert res == {"ok": False, "order_id": None, "error": "no credentials"}
        assert adapter.pull_ohlcv("BTC/USDT") == []


def test_mexc_balance_and_order_normalization():
    fake_exchange = MagicMock()
    fake_exchange.fetch_balance.return_value = {
        "free": {"USDT": 100.0}, "used": {"USDT": 50.0}, "total": {"USDT": 150.0},
    }
    fake_exchange.create_order.return_value = {"id": "mexc-123", "clientOrderId": "abc"}
    fake_exchange.fetch_time.return_value = 1

    with patch("app.services.mexc_adapter.VaultService.get", side_effect=["api-key", "secret"]), \
         patch("ccxt.mexc", return_value=fake_exchange):
        from app.services.mexc_adapter import MEXCAdapter

        adapter = MEXCAdapter(username="me")
        assert adapter.connect_stored() is True
        bal = adapter.fetch_balance()
        assert bal["total"] == 150.0 and bal["quote"] == "USDT"
        assert adapter.fetch_equity() == 150.0

        res = adapter.place_order("BTC/USDT", "BUY", 0.01, sl=100, tp=200)
        assert res["ok"] is True
        assert res["order_id"] == "mexc-123"
        fake_exchange.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 0.01)


def test_mexc_ohlcv_rows():
    fake_exchange = MagicMock()
    fake_exchange.fetch_ohlcv.return_value = [[1700000000000, 1, 2, 3, 4, 5]]
    with patch("app.services.mexc_adapter.VaultService.get", side_effect=["k", "s"]), \
         patch("ccxt.mexc", return_value=fake_exchange):
        from app.services.mexc_adapter import MEXCAdapter

        adapter = MEXCAdapter(username="me")
        rows = adapter.pull_ohlcv("BTC/USDT", "15m", limit=10)
        assert rows[0]["time"] == 1700000000000
        assert rows[0]["close"] == 4.0


def test_copier_place_unsupported_broker_returns_false(session):
    from app.models.entities import Account

    acc = Account(
        id="acct-mexc-1", broker="cTrader", account_type="slave",
        nickname="x", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="ctrader:x", login="x", enabled=True,
    )
    engine = CopierEngine(session)
    assert engine._place(acc, "XAUUSD", "BUY", 0.1, 100, 99, {"tp1": 102}) is False
    assert "unsupported broker" in engine._last_error


def test_copier_place_normalized_order_result_path(session):
    """MEXC (+ future brokers) place through the universal adapter's
    normalized OrderResult contract: ok + order_id -> True + last_order_id."""
    from app.models.entities import Account

    acc = Account(
        id="acct-mexc-2", broker="MEXC", account_type="slave",
        nickname="mexc slave", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="mexc:key123", login="key123", enabled=True,
    )
    fake_adapter = MagicMock()
    fake_adapter.place_order.return_value = {"ok": True, "order_id": "mexc-777"}
    engine = CopierEngine(session)

    with patch("app.services.copier.get_broker_adapter", return_value=fake_adapter):
        ok = engine._place(acc, "BTC/USDT", "BUY", 0.01, 100, 99, {"tp1": 102})
    assert ok is True
    assert engine._last_order_id == "mexc-777"
