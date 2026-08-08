import ccxt

from app.services.vault import VaultService


class BinanceAdapter:
    def __init__(self, username: str = "default"):
        self.api_key = VaultService.get("binance", username)
        self.secret = VaultService.get("binance", f"{username}:secret")
        self.exchange = None
        if self.api_key:
            self.exchange = ccxt.binance({"apiKey": self.api_key, "secret": self.secret})

    def healthcheck(self) -> bool:
        try:
            return self.exchange is not None and self.fetch_balance() is not None
        except Exception:
            return False

    def fetch_balance(self) -> dict:
        """Return a normalized balance summary (free, used, total in USDT)."""
        if self.exchange is None:
            return {"free": None, "used": None, "total": None}
        try:
            bal = self.exchange.fetch_balance()
            return {
                "free": bal.get("free", {}).get("USDT"),
                "used": bal.get("used", {}).get("USDT"),
                "total": bal.get("total", {}).get("USDT"),
            }
        except Exception:
            return {"free": None, "used": None, "total": None}

    def pull_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> list[dict]:
        if self.exchange is None:
            return []
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return [
                {"time": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                for r in raw
            ]
        except Exception:
            return []

    def place_order(self, symbol: str, side: str, amount: float,
                    order_type: str = "market", price: float | None = None) -> dict:
        if self.exchange is None:
            return {"ok": False, "error": "no credentials"}
        try:
            ord = self.exchange.create_order(symbol, order_type, side.lower(), amount, price)
            return {"ok": True, "order": ord}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}