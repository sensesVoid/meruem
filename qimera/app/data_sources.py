"""QiMera on-demand market-data sources — yfinance and the local MT5 terminal.

Research ingestion can pull OHLCV on demand from either source directly,
bypassing Meruem's provider chain (MT5-first for FX, Binance-first for crypto,
Yahoo only as fallback):

- ``yfinance`` — the Yahoo Finance library. Internet-only, needs no local
  dependency; covers stocks/ETFs/indexes/crypto/FX/metals; full daily history
  and up to ~60 days of intraday bars. No 4h interval (Yahoo has none). Yahoo
  rate-limits aggressive polling (HTTP 429) — a blocked request returns [].
- ``mt5`` — the local MetaTrader5 terminal. Requires the terminal to be
  running (no credentials for market data); broker-authoritative for FX and
  metals CFDs and the only on-demand source for 4H candles.

Both return the same candle-dict contract as Meruem's ``MarketDataService``
(``time``/``open``/``high``/``low``/``close``/``volume``), so the M1 data-layer
pipeline (regime tagging, quality gates, parquet) is source-agnostic.

Symbol mapping + interval/range tables reuse Meruem's (``yahoo_symbol``,
``FUTURES_FALLBACK``, ``YAHOO_INTERVAL``, ``YAHOO_RANGE``).
"""

from __future__ import annotations

import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Meruem services are imported via the top-level ``app`` name — put the backend
# first on sys.path so the two ``app`` packages (this project's and Meruem's)
# can never shadow each other. Same self-healing pattern as regime_contract.py
# and data_layer.py; keeps standalone invocations working.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.market_data import (  # noqa: E402 - after path setup
    FUTURES_FALLBACK, YAHOO_INTERVAL, YAHOO_RANGE, yahoo_symbol,
)
from app.services.mt5_adapter import MT5Adapter  # noqa: E402

SOURCE_YFINANCE = "yfinance"
SOURCE_MT5 = "mt5"
SOURCES = (SOURCE_YFINANCE, SOURCE_MT5)

# Deep-history paging defaults (MT5 M1). Window-based paging is required:
# date-range queries make the terminal sync old history from the server, while
# position-based calls cannot reach back that far (verified on this setup).
DEEP_CHUNK_DAYS = 60
DEEP_MAX_BARS = 5_500_000      # ~10 years of 24/5 one-minute bars
DEEP_MAX_EMPTY_WINDOWS = 3     # consecutive empty windows = broker history end
_EPOCH_2000 = 946_684_800      # never crawl before this (safety floor)


def fetch_yfinance(symbol: str, timeframe: str = "15m", limit: int | None = None,
                   start: int | None = None, end: int | None = None) -> list[dict]:
    """On-demand OHLCV from Yahoo Finance via the yfinance library (sync).

    ``start``/``end`` are epoch seconds (window, end exclusive) and must be
    given together; when both are omitted the timeframe's default Yahoo range
    is used. Metal spot tickers fall back to their futures ticker
    (``XAUUSD=X`` -> ``GC=F``), mirroring Meruem's provider. Returns [] for
    unsupported intervals / empty results / rate-limited or blocked requests.
    """
    interval = YAHOO_INTERVAL.get(timeframe)
    if interval is None:
        return []
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")

    import yfinance as yf  # lazy: optional dependency

    kwargs: dict = {
        "interval": interval,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
        "multi_level_index": False,  # keep single-level OHLCV columns
    }
    if start and end and end > start:
        kwargs["start"] = datetime.fromtimestamp(start, tz=timezone.utc)
        kwargs["end"] = datetime.fromtimestamp(end, tz=timezone.utc)
    else:
        kwargs["period"] = YAHOO_RANGE.get(timeframe, "max")

    tickers = [yahoo_symbol(symbol)]
    futures = FUTURES_FALLBACK.get(symbol.upper())
    if futures and futures not in tickers:
        tickers.append(futures)

    for ticker in tickers:
        try:
            df = yf.download(ticker, **kwargs)
        except Exception:  # noqa: BLE001 - yfinance raises on 429 / blocks
            continue
        if df is None or df.empty:
            continue
        candles = _frame_to_candles(df)
        if candles:
            return candles[-limit:] if limit and limit > 0 else candles
    return []


def _frame_to_candles(df: pd.DataFrame) -> list[dict]:
    """Convert a yfinance OHLCV frame to the Meruem candle-dict contract.

    Handles the yfinance 1.x quirks: MultiIndex columns, NaN-padded rows
    (dropped), NaN volume (zeroed) and the tz-naive daily index (localized to
    UTC so epochs are exact).
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    if df.index.tz is None:
        # Daily data arrives tz-naive; treat it as UTC, not local machine time.
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    candles: list[dict] = []
    for ts, row in df.iterrows():
        o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
        if any(pd.isna(x) for x in (o, h, l, c)):
            continue  # yfinance pads some rows with NaN
        vol = row.get("Volume")
        candles.append({
            "time": int(ts.timestamp()),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": 0.0 if pd.isna(vol) else float(vol),
        })
    return candles


def candles_to_arrays(candles: list[dict]) -> dict[str, np.ndarray]:
    """Compact numeric view of a candle list (dict of numpy arrays).

    Roughly 5-10x lighter than the same bars as dicts — the "lightweight"
    representation for million-bar research fetches.
    """
    return {
        "time": np.asarray([c["time"] for c in candles], dtype=np.int64),
        "open": np.asarray([c["open"] for c in candles], dtype=np.float64),
        "high": np.asarray([c["high"] for c in candles], dtype=np.float64),
        "low": np.asarray([c["low"] for c in candles], dtype=np.float64),
        "close": np.asarray([c["close"] for c in candles], dtype=np.float64),
        "volume": np.asarray([c["volume"] for c in candles], dtype=np.float64),
    }


def _empty_result(compact: bool):
    """Empty result in the requested shape (reuses the array converter)."""
    return candles_to_arrays([]) if compact else []


def _concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Chronological concatenation of per-window array dicts (oldest first)."""
    if not parts:
        return candles_to_arrays([])
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fetch_mt5_deep(symbol: str, timeframe: str = "M1",
                   start: int | None = None, end: int | None = None,
                   chunk_days: int = DEEP_CHUNK_DAYS,
                   max_bars: int = DEEP_MAX_BARS,
                   max_empty_windows: int = DEEP_MAX_EMPTY_WINDOWS,
                   delay: float = 0.05,
                   compact: bool = False):
    """Page the local MT5 terminal backwards in date windows to pull as much
    intraday history as the broker serves — the only on-demand source that can
    reach back *years* of 1-minute candles (Yahoo caps 1m at ~30 days).

    Paging uses ``copy_rates_range`` date windows (which trigger the terminal
    to sync deep history from the server); position-based paging cannot reach
    that far back on MT5. Windows crawl from ``end`` (default: now) towards
    ``start`` (default: as far back as the broker offers), stopping after
    ``max_empty_windows`` consecutive empty windows or when ``max_bars`` are
    accumulated.

    Returns candle dicts (``compact=False``) or a dict of numpy arrays
    (``compact=True`` — the lightweight form for million-bar fetches).
    """
    adapter = MT5Adapter()
    now = int(_time.time())
    end_epoch = int(end) if end else now
    if end_epoch > now + 2 * 86_400 or end_epoch < _EPOCH_2000:
        raise ValueError(f"end {end_epoch} is outside the supported range")
    if start is not None and start >= end_epoch:
        return _empty_result(compact)

    def _pull(win_start: int, win_end: int) -> list[dict]:
        return adapter.pull_ohlcv_range(
            symbol, timeframe,
            datetime.fromtimestamp(win_start, tz=timezone.utc),
            datetime.fromtimestamp(win_end, tz=timezone.utc),
        )

    windows: list[list[dict]] = []
    empty_streak = 0
    total = 0
    win_end = end_epoch

    while win_end > (start if start is not None else _EPOCH_2000):
        win_start = max(win_end - chunk_days * 86_400,
                        start if start is not None else _EPOCH_2000)
        candles = _pull(win_start, win_end)
        if not candles:
            # One immediate retry of the same window: transient terminal sync
            # failures are common on deep-history crawls, so a lone empty
            # window must not count as the end of broker history.
            candles = _pull(win_start, win_end)
        if candles:
            windows.append(candles)
            empty_streak = 0
            total += len(candles)
            if total >= max_bars:
                break
        else:
            empty_streak += 1
            if empty_streak >= max_empty_windows:
                break
        _time.sleep(delay)
        win_end = win_start

    if compact:
        # Accumulate per-window arrays so peak memory stays ~window-sized even
        # for million-bar fetches — the "lightweight" promise.
        parts = [candles_to_arrays(chunk) for chunk in reversed(windows)]
        merged = _concat_arrays(parts)
        t = merged["time"]
        # copy_rates_range is inclusive of the upper bound: dedupe boundary bars.
        keep = np.ones(len(t), dtype=bool)
        if len(t) > 1:
            keep[1:] = np.diff(t) > 0
        merged = {k: v[keep] for k, v in merged.items()}
        t = merged["time"]
        lo = 0 if start is None else int(np.searchsorted(t, start, side="left"))
        hi = len(t) if end is None else int(np.searchsorted(t, end, side="left"))
        merged = {k: v[lo:hi] for k, v in merged.items()}
        if len(merged["time"]) > max_bars:
            merged = {k: v[-max_bars:] for k, v in merged.items()}
        return merged

    # dict path: merge oldest-first, dedupe boundary bars, then trim.
    merged: list[dict] = []
    last_time: int | None = None
    for chunk in reversed(windows):
        for c in chunk:
            if c["time"] == last_time:
                continue
            merged.append(c)
            last_time = c["time"]
    if start is not None:
        merged = [c for c in merged if c["time"] >= start]
    if end is not None:
        merged = [c for c in merged if c["time"] < end]
    if len(merged) > max_bars:
        merged = merged[-max_bars:]
    return merged


def fetch_mt5(symbol: str, timeframe: str = "15m", limit: int = 500,
              start: int | None = None, end: int | None = None) -> list[dict]:
    """On-demand OHLCV from the local MetaTrader5 terminal (sync, blocking).

    Requires the MT5 terminal to be running (market data needs no login).
    Returns [] when the terminal/package is unavailable or the broker does not
    offer the symbol. With a date window uses ``copy_rates_range``, else the
    last ``limit`` candles.
    """
    adapter = MT5Adapter()
    if start and end and end > start:
        return adapter.pull_ohlcv_range(
            symbol, timeframe,
            datetime.fromtimestamp(start, tz=timezone.utc),
            datetime.fromtimestamp(end, tz=timezone.utc),
        )
    return adapter.pull_ohlcv(symbol, timeframe, count=max(limit or 1, 1))


def fetch_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 500,
                start: int | None = None, end: int | None = None,
                source: str = SOURCE_YFINANCE, deep: bool = False,
                compact: bool = False):
    """On-demand candle fetch from a named source (``yfinance`` | ``mt5``).

    ``deep=True`` (MT5 only — yfinance caps 1m at ~30 days, so deep history is
    meaningless there and rejected) pages the terminal's full history via
    :func:`fetch_mt5_deep`; ``compact=True`` returns numpy arrays instead of
    dicts (the lightweight form for large fetches).
    """
    if deep and source != SOURCE_MT5:
        raise ValueError("deep history is only supported for source='mt5'")
    if source == SOURCE_YFINANCE:
        return fetch_yfinance(symbol, timeframe, limit, start, end)
    if source == SOURCE_MT5:
        if deep:
            return fetch_mt5_deep(symbol, timeframe, start, end, compact=compact)
        return fetch_mt5(symbol, timeframe, limit, start, end)
    raise ValueError(f"unknown data source {source!r} (expected one of {SOURCES})")
