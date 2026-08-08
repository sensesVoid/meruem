import threading
from typing import Any

from app.services.vault import VaultService

# Chart timeframe labels (MarketDataService) -> MT5 constant names.
TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "M5": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "M15": "TIMEFRAME_M15",
    "1h": "TIMEFRAME_H1",
    "H1": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "H4": "TIMEFRAME_H4",
    "1D": "TIMEFRAME_D1",
    "D1": "TIMEFRAME_D1",
    "1W": "TIMEFRAME_W1",
    "W1": "TIMEFRAME_W1",
    "1M": "TIMEFRAME_MN1",
    "MN1": "TIMEFRAME_MN1",
}

# MT5 binds to a single local terminal and is not safe for concurrent access
# from multiple threads, so all market-data calls share one lock.
_mt5_lock = threading.Lock()
_mt5_initialized = False


def _get_mt5_module():
    """Return the initialized MetaTrader5 module, or None if unavailable."""
    global _mt5_initialized
    if _mt5_initialized:
        import MetaTrader5 as mt5

        return mt5
    with _mt5_lock:
        if _mt5_initialized:
            import MetaTrader5 as mt5

            return mt5
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                return None
            _mt5_initialized = True
            return mt5
        except ImportError:
            return None


def _ensure_mt5_initialized() -> bool:
    """Initialize the local MT5 terminal once (process-wide, thread-safe).

    Market data (OHLCV) is available without any account login; the terminal
    only needs to be running. Returns False if MT5 is unavailable.
    """
    return _get_mt5_module() is not None


class MT5Adapter:
    """MetaTrader5 connector. Uses stored credentials from OS keyring."""

    def __init__(self, login: str | None = None, namespace: str = "mt5"):
        self.namespace = namespace
        self.login = login
        self.mt5 = None

    def _password(self) -> str | None:
        return VaultService.get(self.namespace, self.login or "default")

    def _server(self) -> str | None:
        return VaultService.get(self.namespace, f"{self.login}:server" if self.login else "server")

    def connect(self, login: int, server: str, password: str) -> bool:
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5
            if not mt5.initialize():
                return False
            authorized = mt5.login(login=login, server=server, password=password)
            return bool(authorized)
        except ImportError:
            return False

    def connect_stored(self) -> bool:
        """Connect using stored credentials from the OS keyring."""
        if not self.login:
            return False
        password = VaultService.get(self.namespace, self.login)
        if not password:
            return False
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5
            if not mt5.initialize():
                return False
            server = self._server()
            if server:
                return bool(mt5.login(login=int(self.login), server=server, password=password))
            return bool(mt5.login(login=int(self.login), password=password))
        except (ImportError, TypeError, ValueError):
            return False

    def fetch_equity(self) -> float | None:
        if self.mt5 is None:
            return None
        try:
            info = self.mt5.account_info()
            if info is None:
                return None
            return float(info.equity)
        except Exception:
            return None

    def healthcheck(self) -> bool:
        if self.mt5 is None:
            return False
        try:
            info = self.mt5.account_info()
            return info is not None
        except Exception:
            return False

    def pull_ohlcv(self, symbol: str, timeframe: str, count: int = 200) -> list[dict]:
        if self.mt5 is None:
            self.mt5 = _get_mt5_module()
        if self.mt5 is None:
            return []
        tf_attr = TIMEFRAME_MAP.get(timeframe, "TIMEFRAME_H1")
        tf = getattr(self.mt5, tf_attr)
        with _mt5_lock:
            rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            return []
        df_rows = []
        for row in rates:
            df_rows.append({
                "time": int(row["time"]), "open": float(row["open"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"]), "volume": float(row["tick_volume"] or 0),
            })
        return df_rows

    def pull_ohlcv_range(
        self, symbol: str, timeframe: str,
        date_from, date_to,
    ) -> list[dict]:
        """Pull OHLCV for an EXACT date range via ``copy_rates_range``.

        ``date_from`` / ``date_to`` are datetimes (converted to naive UTC for
        the MT5 call). Returns rows keyed the same as :meth:`pull_ohlcv`.
        """
        if self.mt5 is None:
            self.mt5 = _get_mt5_module()
        if self.mt5 is None:
            return []
        tf_attr = TIMEFRAME_MAP.get(timeframe, "TIMEFRAME_H1")
        tf = getattr(self.mt5, tf_attr)
        try:
            from datetime import datetime, timezone

            start = date_from if isinstance(date_from, datetime) else datetime.fromtimestamp(date_from, tz=timezone.utc)
            end = date_to if isinstance(date_to, datetime) else datetime.fromtimestamp(date_to, tz=timezone.utc)
            # copy_rates_range expects naive datetimes (UTC by convention).
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        except (TypeError, ValueError, OverflowError):
            return []
        with _mt5_lock:
            try:
                rates = self.mt5.copy_rates_range(symbol, tf, start, end)
            except Exception:  # noqa: BLE001 - MT5 can raise on bad params
                rates = None
        if rates is None:
            return []
        df_rows = []
        for row in rates:
            df_rows.append({
                "time": int(row["time"]), "open": float(row["open"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"]), "volume": float(row["tick_volume"] or 0),
            })
        return df_rows

    def place_order(self, symbol: str, action: str, lot: float,
                    price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                    order_type: str = "MARKET") -> Any:
        if self.mt5 is None:
            return None
        if action == "BUY":
            act = self.mt5.ORDER_TYPE_BUY if order_type == "MARKET" else self.mt5.ORDER_TYPE_BUY_STOP
        else:
            act = self.mt5.ORDER_TYPE_SELL if order_type == "MARKET" else self.mt5.ORDER_TYPE_SELL_STOP
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL if order_type == "MARKET" else self.mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": act,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
        }
        return self.mt5.order_send(request)