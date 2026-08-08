"""Broker registry + factory (Phase 10 — #14 universal platform connection).

All broker-specific construction lives here. Callers use
:func:`get_broker_adapter` with the ``Account.broker`` string and never touch
adapter internals — adding a broker is one registry entry.

MT5 is deliberately NOT normalised through this factory for *order placement*:
MetaTrader's ``order_send`` returns a rich native result object and the copier
relies on its retcode semantics. The factory still serves MT5 adapters for the
connect/equity/status paths (the exact same instances the copier used), so the
dispatch is single-sourced everywhere; order placement keeps its documented
MT5-specific confirmation handling inside the copier.
"""

from __future__ import annotations

from typing import Any

BROKER_MT5 = "MT5"
BROKER_BINANCE = "Binance"
BROKER_MEXC = "MEXC"

# Normalised broker names accepted by the connect/test routes and stored on
# ``Account.broker``. ``keyring_namespace`` is the OS-keyring vault namespace
# the adapter reads credentials from.
BROKER_REGISTRY: dict[str, dict[str, Any]] = {
    BROKER_MT5: {"adapter": "app.services.mt5_adapter.MT5Adapter", "keyring_namespace": "mt5"},
    BROKER_BINANCE: {"adapter": "app.services.binance_adapter.BinanceAdapter", "keyring_namespace": "binance"},
    BROKER_MEXC: {"adapter": "app.services.mexc_adapter.MEXCAdapter", "keyring_namespace": "mexc"},
}


def supported_brokers() -> list[str]:
    """The broker ids the connector can currently drive."""
    return list(BROKER_REGISTRY.keys())


def keyring_namespace(broker: str) -> str | None:
    """Keyring/vault namespace for a broker, or None when unsupported."""
    entry = BROKER_REGISTRY.get(broker)
    return entry["keyring_namespace"] if entry else None


def get_broker_adapter(broker: str, account_key: str = "default") -> Any:
    """Construct the adapter instance for a broker.

    ``account_key`` is the vault/keyring username or MT5 login the adapter
    resolves stored credentials with. Returns ``None`` when the broker is not
    in the registry.
    """
    entry = BROKER_REGISTRY.get(broker)
    if not entry:
        return None
    from importlib import import_module

    module_path, class_name = entry["adapter"].rsplit(".", 1)
    cls = getattr(import_module(module_path), class_name)
    if broker == BROKER_MT5:
        return cls(login=account_key)
    return cls(username=account_key)
