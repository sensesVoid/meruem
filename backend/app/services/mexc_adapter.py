"""MEXC exchange adapter (Phase 10 — universal broker connector).

Built on ``ccxt`` (already a project dependency, supports MEXC natively), this
adapter implements the :class:`~app.services.broker_base.BrokerAdapter`
protocol so MEXC accounts plug into the copier / dashboard / risk engine with
zero broker-specific call-site code — exactly like Binance.

Credentials live in the OS keyring under the ``mexc`` namespace
(``api_key`` and ``secret``), mirroring the Binance convention.
"""

from __future__ import annotations

from typing import Any

from app.services.broker_base import BalanceInfo, OrderResult
from app.services.vault import VaultService


class MEXCAdapter:
    """MEXC connector. Uses stored API credentials from the OS keyring."""

    def __init__(self, username: str = "default"):
        self.api_key = VaultService.get("mexc", username)
        self.secret = VaultService.get("mexc", f"{username}:secret")
        self.exchange = None
        if self.api_key:
            try:
                import ccxt

                self.exchange = ccxt.mexc({"apiKey": self.api_key, "secret": self.secret})
            except Exception:  # noqa: BLE001 - unavailable ccxt build
                self.exchange = None

    # --- connectivity ----------------------------------------------------

    def connect(self, api_key: str, secret: str) -> bool:
        """Live connect with explicit credentials (used by the broker test)."""
        try:
            import ccxt

            self.exchange = ccxt.mexc({"apiKey": api_key, "secret": secret})
            self.exchange.fetch_time()
            self.exchange.fetch_balance()  # validates the key's permissions
            return True
        except Exception:  # noqa: BLE001
            return False

    def connect_stored(self) -> bool:
        if self.exchange is None:
            return False
        try:
            self.exchange.fetch_time()
            return True
        except Exception:  # noqa: BLE001
            return False

    def healthcheck(self) -> bool:
        try:
            return self.exchange is not None and self.fetch_balance() is not None
        except Exception:  # noqa: BLE001
            return False

    # --- account state ---------------------------------------------------

    def fetch_balance(self) -> BalanceInfo:
        """Normalized balance summary (free / used / total in USDT)."""
        if self.exchange is None:
            return {"free": None, "used": None, "total": None, "quote": "USDT"}
        try:
            bal = self.exchange.fetch_balance()
            return {
                "free": bal.get("free", {}).get("USDT"),
                "used": bal.get("used", {}).get("USDT"),
                "total": bal.get("total", {}).get("USDT"),
                "quote": "USDT",
            }
        except Exception:  # noqa: BLE001
            return {"free": None, "used": None, "total": None, "quote": "USDT"}

    def fetch_equity(self) -> float | None:
        total = self.fetch_balance().get("total")
        return float(total) if total else None

    # --- market data -----------------------------------------------------

    def pull_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> list[dict]:
        if self.exchange is None:
            return []
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return [
                {"time": r[0], "open": r[1], "high": r[2], "low": r[3],
                 "close": r[4], "volume": r[5]}
                for r in raw
            ]
        except Exception:  # noqa: BLE001
            return []

    def pull_ohlcv_range(self, symbol: str, timeframe: str,
                         date_from: Any, date_to: Any) -> list[dict]:
        """Candles inside an exact date window (paged ``since`` / ``until``)."""
        if self.exchange is None:
            return []
        try:
            from datetime import datetime

            start_ms = (
                int(date_from.timestamp() * 1000)
                if isinstance(date_from, datetime)
                else int(date_from) * 1000
            )
            end_ms = (
                int(date_to.timestamp() * 1000)
                if isinstance(date_to, datetime)
                else int(date_to) * 1000
            )
            since = start_ms
            rows: list[dict] = []
            while since < end_ms:
                raw = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=1000)
                if not raw:
                    break
                rows.extend(
                    {"time": r[0], "open": r[1], "high": r[2], "low": r[3],
                     "close": r[4], "volume": r[5]}
                    for r in raw
                )
                since = int(raw[-1][0]) + 1
                if since >= end_ms:
                    break
            return rows
        except Exception:  # noqa: BLE001
            return []

    # --- order execution -------------------------------------------------

    def place_order(self, symbol: str, action: str, lot: float,
                    price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                    order_type: str = "MARKET") -> OrderResult:
        """Place a market order. Returns a normalized OrderResult."""
        if self.exchange is None:
            return {"ok": False, "order_id": None, "error": "no credentials"}
        try:
            side = "buy" if action.upper() in ("BUY", "LONG") else "sell"
            if order_type.upper() == "MARKET":
                order = self.exchange.create_order(
                    symbol, "market", side.lower(), lot)
            else:
                order = self.exchange.create_order(
                    symbol, "limit", side.lower(), lot, price)
            return {
                "ok": True,
                "order_id": str(order.get("id") or order.get("clientOrderId") or ""),
                "raw": order,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "order_id": None, "error": str(exc), "raw": None}
