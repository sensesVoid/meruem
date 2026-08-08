"""Real OHLCV market data for the chart.

Providers (in order of preference):
  1. MetaTrader5 terminal (local) - forex & metals/commodities (no API key,
     authoritative 4H candles)
  2. Binance public klines       - crypto (no key)
  3. Yahoo Finance chart API     - fallback for everything (no key)

Responses are cached briefly so the dashboard doesn't hammer the providers
on every timeframe/lookback change.
"""

import asyncio
import time

import httpx

# Crypto tickers that Binance lists as BASEUSDT and Yahoo as BASE-USD.
CRYPTO = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "LINK",
    "MATIC", "AVAX", "UNI", "SHIB", "TRX", "XLM", "ATOM", "ETC", "NEAR",
    "FIL", "APT", "ARB", "OP", "INJ", "SUI", "TIA", "SEI", "WIF", "PEPE",
    "BNB", "TON", "XMR", "DASH", "ZEC", "VET", "ALGO", "HBAR", "ICP", "FLOW",
    "FTM", "RUNE", "AAVE", "GRT", "SNX", "MKR", "COMP", "CRV", "ENA", "RNDR",
}

YAHOO_INTERVAL = {
    "M1": "1m", "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": None,
    "1D": "1d", "1W": "1wk", "1M": "1mo",
}
YAHOO_RANGE = {
    "M1": "1d", "1m": "1d", "5m": "5d", "15m": "60d", "1h": "730d",
    "4h": "730d", "1D": "max", "1W": "max", "1M": "max",
}
BINANCE_INTERVAL = {
    "M1": "1m", "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h",
    "1D": "1d", "1W": "1w", "1M": "1M",
}

CACHE_TTL = 60.0  # seconds

_cache: dict[tuple[str, str, int], tuple[float, list[dict]]] = {}
_cache_provider: dict[tuple[str, str, int], str] = {}


def _is_crypto(symbol: str) -> bool:
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix) and symbol[: -len(suffix)] in CRYPTO:
            return True
    return False


def yahoo_symbol(symbol: str) -> str:
    if _is_crypto(symbol):
        base = symbol.rstrip("USDT").rstrip("USDC").rstrip("USD")
        return f"{base}-USD"
    if symbol.endswith("=X"):
        return symbol
    return f"{symbol}=X"  # FX / metals convention used by Yahoo.


def binance_symbol(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix) and symbol[: -len(suffix)] in CRYPTO:
            return f"{symbol[: -len(suffix)]}{suffix}"
    return symbol


# FX / metals that Yahoo only serves via a futures ticker (spot =X is dead).
FUTURES_FALLBACK = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "XPTUSD": "PL=F",
    "XPDUSD": "PA=F",
}

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Shared lazy HTTP client (avoids a new connection pool per instance)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Meruem/1.0"},
        )
    return _client


class MarketDataService:
    def __init__(self) -> None:
        self._client = _get_client()

    async def close(self) -> None:
        # The client is process-wide and shared; nothing to close per-instance.
        pass

    async def fetch(self, symbol: str, timeframe: str = "15m", limit: int = 500,
                    preferred: str | None = None) -> dict:
        key = (symbol, timeframe, limit)
        now = time.time()
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": hit[1],
                "provider": _cache_provider.get(key, ""),
            }

        # Preferred provider (usually the configured master broker) is tried
        # first for non-crypto symbols; crypto still routes to Binance first.
        if preferred and not _is_crypto(symbol):
            if preferred == "Binance":
                candles = await self._binance_public(symbol, timeframe, limit)
                if candles:
                    _cache[key] = (now, candles)
                    _cache_provider[key] = "binance"
                    return {"symbol": symbol, "timeframe": timeframe, "candles": candles, "provider": "binance"}
            elif preferred == "MT5":
                candles, provider = await self._mt5(symbol, timeframe, limit)
                if candles:
                    _cache[key] = (now, candles)
                    _cache_provider[key] = provider
                    return {"symbol": symbol, "timeframe": timeframe, "candles": candles, "provider": provider}

        # Crypto is best served by Binance (authoritative, includes 4h/1M);
        # Yahoo is the fallback. For forex & metals use the local MT5
        # terminal first (authoritative 4H candles); Yahoo is the fallback.
        if _is_crypto(symbol):
            candles, provider = await self._binance(symbol, timeframe, limit)
            if not candles:
                candles, provider = await self._yahoo(symbol, timeframe, limit)
        else:
            candles, provider = await self._mt5(symbol, timeframe, limit)
            if not candles:
                candles, provider = await self._yahoo(symbol, timeframe, limit)

        _cache[key] = (now, candles)
        _cache_provider[key] = provider
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "provider": provider,
        }

    async def fetch_range(self, symbol: str, timeframe: str = "15m",
                          start: int = 0, end: int = 0) -> tuple[list[dict], str]:
        """Candles within an EXACT date window ``[start, end]`` (epoch seconds).

        Provider order mirrors :meth:`fetch`: MT5 range for forex/metals, paged
        Binance klines for crypto, Yahoo ``period1/period2`` as the fallback.
        Returns ``(candles, provider)``.
        """
        if _is_crypto(symbol):
            candles = await self._binance_range(symbol, timeframe, start, end)
            if candles:
                return candles, "binance"
            return await self._yahoo_range(symbol, timeframe, start, end)
        candles, provider = await self._mt5_range(symbol, timeframe, start, end)
        if candles:
            return candles, provider
        return await self._yahoo_range(symbol, timeframe, start, end)

    async def _binance_public(self, symbol: str, timeframe: str, limit: int) -> list[dict]:
        return (await self._binance(symbol, timeframe, limit))[0]

    async def _mt5_range(self, symbol: str, timeframe: str, start: int,
                         end: int) -> tuple[list[dict], str]:
        from datetime import datetime, timezone

        from app.services.mt5_adapter import MT5Adapter

        def _run() -> list[dict]:
            return MT5Adapter().pull_ohlcv_range(
                symbol, timeframe,
                datetime.fromtimestamp(start, tz=timezone.utc),
                datetime.fromtimestamp(end, tz=timezone.utc),
            )

        try:
            candles = await asyncio.to_thread(_run)
        except Exception:  # noqa: BLE001
            candles = []
        if candles:
            return candles, "mt5"
        return [], "mt5"

    async def _binance_range(self, symbol: str, timeframe: str, start: int,
                             end: int) -> list[dict]:
        """Page Binance klines across the exact window (1000 candles/call)."""
        interval = BINANCE_INTERVAL.get(timeframe)
        if interval is None or end <= start:
            return []
        url = "https://api.binance.com/api/v3/klines"
        out: list[dict] = []
        cursor = start * 1000
        end_ms = end * 1000
        while cursor < end_ms:
            params = {
                "symbol": binance_symbol(symbol), "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            }
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                raw = resp.json()
            except Exception:  # noqa: BLE001
                break
            if not raw:
                break
            out.extend([
                {
                    "time": int(r[0] // 1000),
                    "open": float(r[1]), "high": float(r[2]),
                    "low": float(r[3]), "close": float(r[4]),
                    "volume": float(r[5]),
                }
                for r in raw
            ])
            last_ms = int(raw[-1][0])
            if len(raw) < 1000 or last_ms <= cursor:
                break
            cursor = last_ms + 1
        return out

    async def _yahoo_range(self, symbol: str, timeframe: str, start: int,
                           end: int) -> tuple[list[dict], str]:
        interval = YAHOO_INTERVAL.get(timeframe)
        if interval is None or end <= start:
            return [], "yahoo"
        tickers = [yahoo_symbol(symbol)]
        fb = FUTURES_FALLBACK.get(symbol)
        if fb and fb not in tickers:
            tickers.append(fb)
        for ticker in tickers:
            candles = await self._yahoo_ticker(ticker, interval, timeframe, 0,
                                               start=start, end=end)
            if candles:
                return candles, "yahoo"
        return [], "yahoo"

    async def _mt5(self, symbol: str, timeframe: str, limit: int) -> tuple[list[dict], str]:
        """Pull candles from the local MetaTrader5 terminal (sync, off-thread)."""
        from app.services.mt5_adapter import MT5Adapter

        def _run() -> list[dict]:
            return MT5Adapter().pull_ohlcv(symbol, timeframe, count=max(limit, 1))

        try:
            candles = await asyncio.to_thread(_run)
        except Exception:  # noqa: BLE001
            candles = []
        if candles:
            return candles, "mt5"
        return [], "mt5"

    async def _yahoo(self, symbol: str, timeframe: str, limit: int) -> tuple[list[dict], str]:
        interval = YAHOO_INTERVAL.get(timeframe)
        if interval is None:  # 4h is not offered by this endpoint
            return [], "yahoo"

        # Try the mapped ticker first; if the provider rejects it (e.g. dead
        # spot FX ticker), fall back to the futures ticker for that symbol.
        tickers = [yahoo_symbol(symbol)]
        fb = FUTURES_FALLBACK.get(symbol)
        if fb and fb not in tickers:
            tickers.append(fb)

        for ticker in tickers:
            candles = await self._yahoo_ticker(ticker, interval, timeframe, limit)
            if candles:
                return candles, "yahoo"
        return [], "yahoo"

    async def _yahoo_ticker(
        self, ticker: str, interval: str, timeframe: str, limit: int,
        start: int | None = None, end: int | None = None,
    ) -> list[dict]:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
        params: dict = {"interval": interval, "includePrePost": "false"}
        if start is not None and end is not None and end > start:
            # Exact date window (epoch seconds) beats the range presets.
            params["period1"] = int(start)
            params["period2"] = int(end)
        else:
            params["range"] = YAHOO_RANGE.get(timeframe, "max")
        try:
            resp = await self._client.get(url, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
            result = data.get("chart", {}).get("result")
            if not result:
                return []
            ts = result[0].get("timestamp") or []
            quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
            o = quote.get("open") or []
            h = quote.get("high") or []
            l = quote.get("low") or []
            c = quote.get("close") or []
            v = quote.get("volume") or []
            candles: list[dict] = []
            for i, t in enumerate(ts):
                if i >= len(c) or c[i] is None:
                    continue
                candles.append({
                    "time": int(t),
                    "open": float(o[i]),
                    "high": float(h[i]),
                    "low": float(l[i]),
                    "close": float(c[i]),
                    "volume": float(v[i] or 0),
                })
            if limit > 0:
                return candles[-limit:]
            return candles
        except Exception:  # noqa: BLE001
            return []

    async def _binance(self, symbol: str, timeframe: str, limit: int) -> tuple[list[dict], str]:
        interval = BINANCE_INTERVAL.get(timeframe)
        if interval is None:
            return [], "binance"
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_symbol(symbol), "interval": interval, "limit": min(limit, 1000)}
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
            candles = [
                {
                    "time": int(r[0] // 1000),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
                for r in raw
            ]
            return candles, "binance"
        except Exception:  # noqa: BLE001
            return [], "binance"
