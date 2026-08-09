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


# Fiat quote suffixes stripped from crypto bases (order matters: USDT before
# USD so "BTCUSDT" never has its trailing "T" misread as part of "USD").
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")

FX_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
FX_METALS = {"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"}


def _strip_quote(symbol: str) -> str:
    """Remove a trailing fiat quote suffix if present.

    Practically the same as `symbol.rstrip("USDT")` but fixes the classic
    `rstrip` bug that also eats letters from the BASE when the base itself
    ends in a stripped character (e.g. ``BTCUSDT -> BT``, ``LTCUSDT -> LT``).
    """
    sym = symbol.upper()
    for suffix in _QUOTE_SUFFIXES:
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return sym[: -len(suffix)]
    return sym


def _is_crypto(symbol: str) -> bool:
    return _strip_quote(symbol) in CRYPTO


def _crypto_quote(symbol: str) -> str | None:
    """Return the Binance-native quote suffix (``USDT``/``USDC``/``BUSD``) if
    the symbol carries one, else ``None``.

    These quotes only exist as exchange pairs on Binance — the MT5 terminal
    never serves ``BTCUSDT``, so it must be skipped entirely for them.
    """
    sym = symbol.upper()
    for suffix in ("USDT", "USDC", "BUSD"):
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return suffix
    return None


def _binance_only(symbol: str) -> bool:
    """True when the symbol is Binance-only (``USDT``/``USDC`` quoted) and
    should never be attempted on the local MT5 terminal."""
    return _crypto_quote(symbol) is not None


def _plain_usd_crypto(symbol: str) -> bool:
    """True for a crypto base quoted in plain ``USD`` (e.g. ``BTCUSD``,
    ``ETHUSD``) — the broker CFD convention that MT5 serves (as opposed to
    ``BTCUSDT`` which only exists on Binance)."""
    sym = symbol.upper()
    if any(sym.endswith(s) for s in ("USDT", "USDC", "BUSD")):
        return False
    return sym.endswith("USD") and _strip_quote(sym) in CRYPTO


def _binance_routed(symbol: str, asset_class: str | None = None) -> bool:
    """Decide whether Binance should be tried BEFORE the MT5 terminal:

    - a crypto/cripto asset_class hint, or
    - a Binance-native quote (``BTCUSDT`` / ``ETHUSDC``…), or
    - a bare crypto base (``BTC`` → Binance ``BTCUSDT``).

    Plain-``USD`` crypto (``BTCUSD``) is intentionally NOT Binance-first —
    that symbol is a CFD on the local MT5 terminal and must be fetched from
    there first (Binance/Yahoo are the fallbacks).
    """
    cls = (asset_class or "").strip().lower()
    if cls in ("crypto", "cripto", "cryptocurrency", "digital"):
        return True
    if _crypto_quote(symbol) is not None:
        return True
    return _strip_quote(symbol) in CRYPTO and not _plain_usd_crypto(symbol)


def _looks_like_fx(symbol: str) -> bool:
    """Heuristic: 6-letter codes such as EURUSD / GBPJPY and the X-metal codes
    are treated as FX/metals (Yahoo ``=X`` convention); otherwise the symbol is
    left bare (US stocks/ETFs like ``AAPL``, ``SPY``)."""
    sym = symbol.upper()
    if sym in FX_METALS:
        return True
    if len(sym) == 6 and sym.isalpha():
        base, quote = sym[:3], sym[3:]
        return (base.startswith("X") or base in FX_CURRENCIES) and quote in FX_CURRENCIES
    return False


def yahoo_symbol(symbol: str, asset_class: str | None = None) -> str:
    """Map an internal symbol to the Yahoo Finance chart ticker.

    - Crypto           → ``BASE-USD`` (``BTCUSDT``/``BTCUSD``/``BTC`` → ``BTC-USD``)
    - FX / metals      → ``EURUSD=X`` / ``XAUUSD=X``
    - Stocks / ETFs    → bare ticker (``AAPL``, ``SPY``)
    The asset_class hint (forex/crypto/stock/commodities…) disambiguates the
    ambiguous 6-letter codes when the caller knows it.
    """
    cls = (asset_class or "").strip().lower()
    base = _strip_quote(symbol)
    if _is_crypto(symbol) or cls in ("crypto", "cripto", "cryptocurrency", "digital"):
        return f"{base}-USD"
    if symbol.endswith("=X"):
        return symbol.upper()
    if cls in ("stock", "equity", "etf", "index", "share"):
        return base or symbol.upper()
    if _looks_like_fx(symbol) or cls in ("forex", "fx", "currency", "commodities", "commodity", "metal", "metals", "indexes"):
        return f"{symbol.upper()}=X"
    return symbol.upper()


def binance_symbol(symbol: str) -> str:
    """Map a crypto symbol to the Binance kline ticker (BASE + quote).

    ``BTCUSDT`` stays ``BTCUSDT``; ``BTCUSD`` → ``BTCUSD``; a bare base such as
    ``BTC`` / ``ETH`` defaults to the ``USDT`` quote so the default watchlist
    style keeps working on Binance.
    """
    sym = symbol.upper()
    if _is_crypto(sym):
        base = _strip_quote(sym)
        if base != sym:
            return sym
        return f"{base}USDT"
    return sym


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
                    preferred: str | None = None,
                    asset_class: str | None = None) -> dict:
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

        binance_first = _binance_routed(symbol, asset_class)

        # Preferred provider (usually the configured master broker) is tried
        # first for non-Binance-first symbols; Binance-native crypto still
        # routes to Binance.
        if preferred and not binance_first:
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
        # Yahoo is the fallback. For forex, metals & plain-USD crypto CFDs
        # (BTCUSD) use the local MT5 terminal first (authoritative 4H candles);
        # Binance + Yahoo act as fallbacks.
        if binance_first:
            candles, provider = await self._binance(symbol, timeframe, limit)
            if not candles:
                candles, provider = await self._yahoo(symbol, timeframe, limit, asset_class=asset_class)
        else:
            candles, provider = await self._mt5(symbol, timeframe, limit)
            if not candles and _plain_usd_crypto(symbol):
                # MT5 does not serve the CFD right now → Binance is still a
                # valid crypto fallback before Yahoo.
                candles, provider = await self._binance(symbol, timeframe, limit)
            if not candles:
                candles, provider = await self._yahoo(symbol, timeframe, limit, asset_class=asset_class)

        _cache[key] = (now, candles)
        _cache_provider[key] = provider
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "provider": provider,
        }

    async def fetch_range(self, symbol: str, timeframe: str = "15m",
                          start: int = 0, end: int = 0,
                          asset_class: str | None = None) -> tuple[list[dict], str]:
        """Candles within an EXACT date window ``[start, end]`` (epoch seconds).

        Provider order mirrors :meth:`fetch`: MT5 range for forex/metals, paged
        Binance klines for crypto, Yahoo ``period1/period2`` as the fallback.
        Returns ``(candles, provider)``.
        """
        if _binance_routed(symbol, asset_class):
            candles = await self._binance_range(symbol, timeframe, start, end)
            if candles:
                return candles, "binance"
            return await self._yahoo_range(symbol, timeframe, start, end, asset_class=asset_class)
        candles, provider = await self._mt5_range(symbol, timeframe, start, end)
        if candles:
            return candles, provider
        if _plain_usd_crypto(symbol):
            candles = await self._binance_range(symbol, timeframe, start, end)
            if candles:
                return candles, "binance"
        return await self._yahoo_range(symbol, timeframe, start, end, asset_class=asset_class)

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
                           end: int, asset_class: str | None = None) -> tuple[list[dict], str]:
        interval = YAHOO_INTERVAL.get(timeframe)
        if interval is None or end <= start:
            return [], "yahoo"
        tickers = [yahoo_symbol(symbol, asset_class)]
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

    async def _yahoo(self, symbol: str, timeframe: str, limit: int,
                     asset_class: str | None = None) -> tuple[list[dict], str]:
        interval = YAHOO_INTERVAL.get(timeframe)
        if interval is None:  # 4h is not offered by this endpoint
            return [], "yahoo"

        # Try the mapped ticker first; if the provider rejects it (e.g. dead
        # spot FX ticker), fall back to the futures ticker for that symbol.
        tickers = [yahoo_symbol(symbol, asset_class)]
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
