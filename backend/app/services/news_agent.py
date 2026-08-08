"""News Agent.

Filters market news to the watchlist assets and produces an AI summary +
fundamental analysis per asset, cached for a refresh cycle (15min). The output is
fed to the Validator (and surfaced in the Chart tab News Agent panel) so trade
decisions reference current macro/fundamental context rather than only prices.
"""

import threading
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.models.entities import Asset

_REFRESH_SECONDS = 15 * 60  # 15 minutes

_STATE: dict = {
    "refreshed_at": None,
    "assets": {},       # {symbol: {summary, analysis, bias, block_events, trade_block, ...}}
    "error": None,
}
_LOCK = threading.Lock()

# High-impact macro keywords that can justify a trade block. Deterministic (not
# LLM), so it is a reliable scheduling gate regardless of provider availability.
HIGH_IMPACT_TERMS = (
    "policy rate", "nonfarm payroll", "nfp", "fomc", "interest rate decision",
    "rate decision", "central bank", "fed rate", "cpi", "inflation data",
    "consumer price index", "core cpi", "ppi", "producer price", "gdp", "unemployment rate",
    "employment report", "jobs report", "opec", "earnings", "black swan",
    "flash crash", "bank bailout",
)

# Default block window (minutes before / after the event's publish time) used when
# the DB settings are missing or unreadable.
DEFAULT_BEFORE_MIN = 30
DEFAULT_AFTER_MIN = 45


def _parse_published(value: object) -> datetime | None:
    """Best-effort ISO8601 parse of a news item's publish timestamp."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "," in text:
        text = text.replace(",", "")
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return _parsed_aware(datetime.fromisoformat(text))
    except (ValueError, TypeError):
        return None


def _block_events(items: list[dict]) -> list[dict]:
    """Extract the high-impact events from a batch of news items.

    Each event keeps its publish timestamp (and a raw copy for response display)
    so the active window can be resolved lazily against the live clock.
    """
    out: list[dict] = []
    for it in (items or [])[:12]:
        text = f"{it.get('title', '')} {it.get('summary', '')}".upper()
        terms = [t for t in HIGH_IMPACT_TERMS if t.upper() in text]
        if terms:
            out.append({
                "published": it.get("published") or "",
                "published_at": _parse_published(it.get("published")),
                "terms": terms[:3],
            })
    return out


def _block_window() -> tuple[int, int]:
    """Read the before/after window (minutes) from the live settings."""
    try:
        from app.services.model_config import _read

        before = int(_read("news_block_before_min") or DEFAULT_BEFORE_MIN)
        after = int(_read("news_block_after_min") or DEFAULT_AFTER_MIN)
    except Exception:  # noqa: BLE001 - never block on settings hiccup
        before, after = DEFAULT_BEFORE_MIN, DEFAULT_AFTER_MIN
    return max(10, before), max(10, after)


def _resolve_block(events: list[dict], now: datetime | None = None) -> dict:
    """Resolve whether a news event is inside the active window right now.

    An event blocks trading when ``now`` falls within
    ``[published - before_min, published + after_min]``. Events without a
    parseable publish timestamp are anchored at ``now`` (still current news).
    """
    now = now or datetime.now(timezone.utc)
    before_min, after_min = _block_window()
    for ev in events:
        anchor = ev.get("published_at") or now
        start = anchor - timedelta(minutes=before_min)
        end = anchor + timedelta(minutes=after_min)
        if start <= now <= end:
            return {
                "trade_block": True,
                "block_reason": "High-impact event inbound: " + ", ".join(ev.get("terms") or []),
                "block_start": start.isoformat(),
                "block_end": end.isoformat(),
                "event_at": (anchor.isoformat() if isinstance(anchor, datetime) else ""),
            }
    return {"trade_block": False, "block_reason": "", "block_start": None,
            "block_end": None, "event_at": ""}


def _detect_trade_block(items: list[dict]) -> dict:
    """Return the current block status for a batch of items (for display/cache)."""
    return _resolve_block(_block_events(items))


def active_block(symbol: str) -> dict:
    """Resolve the ACTIVE block for a symbol using the cached events + live clock + settings.
    Returns ``{"trade_block", "block_reason", ...}``; used by the orchestrator at scan time."""
    with _LOCK:
        asset = dict((_STATE.get("assets") or {}).get(symbol) or {})
    return _resolve_block(asset.get("block_events") or [])


def _prompt(symbol: str, raw_items: list[dict]) -> str:
    snippet = []
    for it in (raw_items or [])[:6]:
        snippet.append(f"- {it.get('title', '')} ({it.get('source', '')})")
    body = "\n".join(snippet) if snippet else "No major news found for this asset."
    return (
        f"You are Meruem News Agent. Summarize the news affecting {symbol} for a "
        "trader. DO NOT just list headlines.\n\n"
        "Raw sources:\n"
        f"{body}\n\n"
        "Return STRICT JSON ONLY with keys:\n"
        '"summary": a 2-4 sentence synthesis of what the news means for the asset.\n'
        '"fundamental_bias": "BULLISH"|"BEARISH"|"NEUTRAL".\n'
        '"key_events": [short strings naming the 1-3 most important catalysts].\n'
        '"precision_notes": any minor details (data releases, central-bank, supply/demand).'
    )


def _parse(response: str) -> dict:
    from app.agents.ollama_client import parse_json

    try:
        data = parse_json(response)
    except Exception:  # noqa: BLE001
        data = {}
    return {
        "summary": str(data.get("summary") or "").strip(),
        "fundamental_bias": str(data.get("fundamental_bias") or "NEUTRAL").upper(),
        "key_events": data.get("key_events") or [],
        "precision_notes": str(data.get("precision_notes") or "").strip(),
    }


class NewsAgent:
    """Watchlist-filtered news with AI fundamental summaries (15min cache)."""

    def _raw_items(self, session: Session, symbol: str | None = None) -> list[dict]:
        try:
            from app.services.news import NewsService

            watchlist = [a.symbol for a in session.exec(select(Asset)).all()]
            return NewsService().fetch(limit=12, symbol=symbol, watchlist=watchlist)
        except Exception:  # noqa: BLE001
            return []

    def _summarize(self, symbol: str, items: list[dict]) -> dict:
        """Best-effort AI synthesis; falls back to raw headline digest on failure."""
        try:
            from app.agents.validator import ValidatorChain

            raw, provider = ValidatorChain(agent_id="news").generate(_prompt(symbol, items))
            parsed = _parse(raw)
            if parsed.get("summary"):
                parsed["provider"] = provider
                return parsed
            raise ValueError("empty summary")
        except Exception:  # noqa: BLE001
            return {
                "summary": ("\n".join(f"• {i.get('title', '')}" for i in (items or [])[:5])
                            or "No news items available."),
                "fundamental_bias": "NEUTRAL",
                "key_events": [],
                "precision_notes": "Raw digest only (AI synthesis unavailable).",
                "provider": "fallback",
            }

    def refresh(self, session: Session) -> dict:
        try:
            assets = session.exec(select(Asset)).all()
            out: dict[str, dict] = {}
            for asset in assets:
                items = self._raw_items(session, asset.symbol)
                summary = self._summarize(asset.symbol, items)
                summary["symbol"] = asset.symbol
                summary["asset_class"] = asset.asset_class
                summary["item_count"] = len(items)
                events = _block_events(items)
                summary["block_events"] = events
                summary.update(_resolve_block(events))
                out[asset.symbol] = summary
            with _LOCK:
                _STATE.update({
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                    "assets": out,
                    "error": None,
                })
        except Exception as exc:  # noqa: BLE001
            with _LOCK:
                _STATE["error"] = f"{type(exc).__name__}: {exc}"
        return state()


def state() -> dict:
    """Thread-safe snapshot of the latest news-agent cache.

    Block fields are resolved against the LIVE clock so the UI always reflects
    the current window, even between refreshes.
    """
    with _LOCK:
        assets = {k: dict(v) for k, v in (_STATE["assets"] or {}).items()}
        refreshed_at = _STATE["refreshed_at"]
        error = _STATE["error"]
    for value in assets.values():
        value.update(_resolve_block(value.get("block_events") or []))
    return {"refreshed_at": refreshed_at, "assets": assets, "error": error}


def is_stale() -> bool:
    """True when the cache is empty or older than the 15min refresh cycle."""
    ts = _STATE.get("refreshed_at")
    if not ts:
        return True
    try:
        parsed = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - _parsed_aware(parsed)).total_seconds()
        return age > _REFRESH_SECONDS
    except Exception:  # noqa: BLE001
        return True


def _parsed_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt