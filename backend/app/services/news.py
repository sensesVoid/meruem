"""News & Economic Calendar aggregator (RSS + optional GDELT / CryptoPanic)."""
import datetime as dt
from typing import Any

import feedparser
import requests

from app.services.vault import VaultService

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Meruem/0.2"}

# Reliable, key-less RSS sources. Dedicated feeds per asset class improve
# watchlist-matching coverage (crypto + forex/commodities).
RSS_SOURCES: dict[str, list[str]] = {
    "fx": [
        "https://www.fxstreet.com/rss/",                              # forex/macro/commodities
        "https://www.investing.com/rss/news_25.rss",                  # forex
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",            # crypto
        "https://cointelegraph.com/rss",                              # crypto
    ],
    "news": ["https://www.investing.com/rss/news_1.rss"],
    "cnbc": ["https://www.cnbc.com/id/10000664/device/rss/rss.html"],  # markets
}

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"


class NewsService:
    """Aggregates macro economic events and market news with graceful per-source failure."""

    def __init__(self) -> None:
        self.crypto_key = VaultService.get("news", "cryptopanic")

    def fetch(
        self,
        limit: int = 40,
        symbol: str | None = None,
        watchlist: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        pool_size = max(limit * 2, 100)  # pull a larger pool, then filter/sort/cap
        items: list[dict[str, Any]] = []
        for _, urls in RSS_SOURCES.items():
            for url in urls:
                items.extend(self._fetch_rss(url, pool_size))
        items.extend(self._fetch_gdelt(symbol))
        items.extend(self._fetch_cryptopanic(symbol))

        # English-only: drop items written in a non-Latin script.
        items = [it for it in items if self._is_english(it)]

        # Filter to watchlist assets when provided (falls back to general news
        # if the watchlist is empty so the panel is never blank).
        symbols = self._symbols(watchlist, symbol)
        if symbols:
            items = [it for it in items if self._matches(it, symbols)]

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for it in sorted(items, key=lambda x: x.get("published") or "", reverse=True):
            key = it.get("url") or it.get("title", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _symbols(watchlist: list[str] | None, symbol: str | None) -> list[str]:
        merged = set()
        if watchlist:
            merged.update(s.strip().upper() for s in watchlist if s and s.strip())
        if symbol:
            merged.add(symbol.strip().upper())
        return sorted(merged)

    @staticmethod
    def _matches(item: dict[str, Any], symbols: list[str]) -> bool:
        """Does the item mention any watchlist symbol or a common alias?"""
        text = f"{item.get('title', '')} {item.get('summary', '')}".upper()
        for sym in symbols:
            for needle in NewsService._aliases(sym):
                if needle in text:
                    return True
        return False

    @staticmethod
    def _aliases(symbol: str) -> list[str]:
        aliases: dict[str, list[str]] = {
            "XAUUSD": ["XAUUSD", "GOLD"],
            "XAGUSD": ["XAGUSD", "SILVER"],
            "EURUSD": ["EURUSD", "EURO"],
            "GBPUSD": ["GBPUSD", "STERLING", "POUND"],
            "USDJPY": ["USDJPY", "YEN"],
            "US30": ["US30", "DOW", "DJIA"],
            "SPX500": ["SPX500", "S&P", "S&P 500", "SP 500"],
            "NAS100": ["NAS100", "NASDAQ"],
            "BTC": ["BTC", "BITCOIN"],
            "ETH": ["ETH", "ETHEREUM"],
            "SOL": ["SOL", "SOLANA"],
            "BNB": ["BNB", "BINANCE COIN"],
            "XRP": ["XRP", "RIPPLE"],
            "ADA": ["ADA", "CARDANO"],
        }
        base = [symbol]
        return base + aliases.get(symbol, [])

    @staticmethod
    def _is_english(item: dict[str, Any]) -> bool:
        """Heuristic: title/summary must contain ASCII letters and no
        characters from major non-Latin scripts."""
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        has_ascii_letter = any(
            c.isalpha() and ord(c) < 128 for c in text
        )
        if not has_ascii_letter:
            return False
        for ch in text:
            cp = ord(ch)
            if cp > 127 and (
                (0x0400 <= cp <= 0x04FF)   # Cyrillic
                or (0x0590 <= cp <= 0x05FF)  # Hebrew
                or (0x0600 <= cp <= 0x06FF)  # Arabic
                or (0x0900 <= cp <= 0x097F)  # Devanagari
                or (0x0E00 <= cp <= 0x0E7F)  # Thai
                or (0x3040 <= cp <= 0x30FF)  # Hiragana/Katakana
                or (0x3400 <= cp <= 0x9FFF)  # CJK
                or (0xAC00 <= cp <= 0xD7AF)  # Hangul
            ):
                return False
        return True

    def _fetch_rss(self, url: str, limit: int) -> list[dict[str, Any]]:
        try:
            resp = requests.get(url, headers=UA, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            items = []
            for entry in feed.entries[:limit]:
                published = ""
                parsed = entry.get("published_parsed") or entry.get("updated_parsed")
                if parsed:
                    published = dt.datetime(*parsed[:6]).isoformat()
                items.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "source": NewsService._source_name(url),
                    "published": published,
                    "summary": (entry.get("summary") or "")[:240],
                })
            return items
        except Exception:  # noqa: BLE001 - per-source resilience
            return []

    @staticmethod
    def _source_name(url: str) -> str:
        if "cnbc.com" in url:
            return "cnbc"
        if "coindesk" in url or "cointelegraph" in url:
            return "crypto"
        if "fxstreet" in url:
            return "fx"
        if "investing.com/rss/news_25" in url:
            return "fx"
        return "news"

    def _fetch_gdelt(self, symbol: str | None) -> list[dict[str, Any]]:
        query = " ".join(["markets finance economy", f'("{symbol}")' if symbol else ""]).strip()
        try:
            resp = requests.get(
                GDELT_URL,
                params={"query": query, "mode": "artlist", "format": "json", "maxrecords": 20},
                headers=UA,
                timeout=15,
            )
            if resp.status_code == 429:
                return []  # aggressive rate limit; treat as empty, not an error
            resp.raise_for_status()
            arts = resp.json().get("articles", [])
            return [{
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": "gdelt",
                "published": (a.get("seendate") or "")[:14],
                "summary": (a.get("title") or "") + " — " + (a.get("sourcecountry") or ""),
            } for a in arts]
        except Exception:  # noqa: BLE001
            return []

    def _fetch_cryptopanic(self, symbol: str | None) -> list[dict[str, Any]]:
        if not self.crypto_key:
            return []
        try:
            params = {"auth_token": self.crypto_key, "public": "true"}
            if symbol and symbol.upper() in {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA"}:
                params["currencies"] = symbol.upper()
            resp = requests.get(CRYPTOPANIC_URL, params=params, timeout=15)
            resp.raise_for_status()
            items = []
            for p in resp.json().get("results", []):
                items.append({
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "source": "cryptopanic",
                    "published": p.get("created_at", ""),
                    "summary": " ".join(
                        t.get("title", "") for t in p.get("tags", []) if isinstance(t, dict)
                    ) or p.get("title", ""),
                })
            return items
        except Exception:  # noqa: BLE001
            return []