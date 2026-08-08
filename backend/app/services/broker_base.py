"""Universal broker connector contract (Phase 10 — #14).

Every broker adapter (MT5, Binance, MEXC, future cTrader, ...) implements the
:class:`BrokerAdapter` protocol so the copier, dashboard, risk manager and
routes can talk to ANY broker through ONE dispatch path instead of
``if account.broker == "MT5" ... elif == "Binance"`` chains.

Normalized contracts
--------------------
- ``fetch_balance()`` -> :class:`BalanceInfo` (free / used / total, any
  currency — each adapter documents its quote currency, e.g. USDT).
- ``place_order(...)`` -> :class:`OrderResult` (ok + broker order id + error).
- ``pull_ohlcv`` / ``pull_ohlcv_range`` return the standard OHLCV row dicts
  already used across the app (``time`` epoch seconds, open/high/low/close/
  volume floats).

Adapters are constructed via the factory in :mod:`broker_adapters`
(:func:`~broker_adapters.get_broker_adapter`), which maps the ``Account.broker``
string to a concrete adapter class. Adding a new broker = implement the
protocol + register it in the registry; no call-site changes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ccxt-style normalized balance summary. ``quote`` names the currency the
# figures are denominated in (USDT for Binance/MEXC; MT5 returns a flat
# equity number instead and uses ``quote`` = the account's base currency).
BalanceInfo = dict[str, Any]

# Normalized order result. ``ok`` True only when the broker confirmed the
# placement; ``order_id`` is the broker ticket; ``raw`` carries the native
# response for diagnostics.
OrderResult = dict[str, Any]


@runtime_checkable
class BrokerAdapter(Protocol):
    """The contract every broker adapter must satisfy."""

    # --- connectivity -----------------------------------------------------
    def connect(self, *args: Any, **kwargs: Any) -> bool:  # pragma: no cover
        """Live connect with explicit credentials. Returns True on success."""
        ...

    def connect_stored(self) -> bool:  # pragma: no cover
        """Connect using credentials already stored in the OS keyring."""
        ...

    def healthcheck(self) -> bool:  # pragma: no cover
        """True when the adapter is connected and the account is reachable."""
        ...

    # --- account state ---------------------------------------------------
    def fetch_balance(self) -> BalanceInfo:  # pragma: no cover
        """Normalized balance summary (free/used/total + quote)."""
        ...

    def fetch_equity(self) -> float | None:  # pragma: no cover
        """Live account equity, or None when unavailable."""
        ...

    # --- market data -----------------------------------------------------
    def pull_ohlcv(self, symbol: str, timeframe: str, count: int = 200) -> list[dict]:  # pragma: no cover
        """Most recent ``count`` candles as standard OHLCV row dicts."""
        ...

    def pull_ohlcv_range(self, symbol: str, timeframe: str,
                         date_from: Any, date_to: Any) -> list[dict]:  # pragma: no cover
        """Candles inside an exact date window. Optional; default = []."""
        ...

    # --- order execution -------------------------------------------------
    def place_order(self, symbol: str, action: str, lot: float,
                    price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                    order_type: str = "MARKET") -> OrderResult:  # pragma: no cover
        """Place a market/pending order. Returns a normalized OrderResult."""
        ...
