"""DatasetService — Phase 8 Historical Data Extraction & Parquet Storage.

Extracts historical OHLCV dataset from MT5 (or Binance / Yahoo Finance fallback),
formats into standard schema (timestamp, open, high, low, close, volume), and
saves to anchored Parquet files in `backend/data/historical/`.

Extraction is driven by an EXACT date range (``start_date`` / ``end_date``) rather
than a bar count, and can run across the whole watchlist at once (one dataset per
symbol). Supports listing, previewing, and deleting datasets for backtesting &
ML training.
"""

import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.core.config import BACKEND_DIR
from app.services.market_data import MarketDataService
from app.services.mt5_adapter import MT5Adapter

DATASETS_DIR = BACKEND_DIR / "data" / "historical"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def read_dataset_candles(
    symbol: str, timeframe: str,
    bars: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict] | None:
    """Read an extracted Parquet dataset back as OHLCV candle dicts.

    The rows share the shape returned by ``MarketDataService`` (``time`` as
    epoch seconds), so backtests and the ML scanner can consume extracted
    history exactly like live data. Slicing is optional: ``bars`` keeps the
    most recent N rows, ``start``/``end`` keep an exact window. Returns
    ``None`` when no dataset exists for ``(symbol, timeframe)`` or it cannot
    be read — callers fall back to the live providers.
    """
    path = DATASETS_DIR / f"{str(symbol).strip().upper()}_{str(timeframe).strip()}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) == 0:
            return None
        df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
        if start is not None:
            df = df[df["dt"] >= start]
        if end is not None:
            df = df[df["dt"] <= end]
        df = df.sort_values("dt")
        if bars is not None and bars > 0:
            df = df.tail(int(bars))
        if len(df) == 0:
            return None
        rows = []
        for r in df.to_dict("records"):
            rows.append({
                "time": int(r["dt"].timestamp()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("volume", 0) or 0),
            })
        return rows
    except Exception:  # noqa: BLE001 - unreadable dataset -> live fallback
        return None

# Safety ceiling for a single extraction window (avoids accidentally pulling
# gigantic ranges into memory / disk). ~1 year of 15m candles.
MAX_BARS = 500_000
DEFAULT_RANGE_DAYS = 730  # 2 years when only one of the dates is given

_TIMEFRAME_SECONDS = {
    "M1": 60, "1m": 60, "M5": 300, "5m": 300, "M15": 900, "15m": 900,
    "H1": 3600, "1h": 3600, "H4": 14400, "4h": 14400, "D1": 86400,
    "1D": 86400, "W1": 604800, "1W": 604800, "MN1": 2592000, "1M": 2592000,
}


class DatasetService:
    """Manages historical OHLCV data extraction and Parquet persistence."""

    def __init__(self, storage_dir: pathlib.Path | None = None):
        self.storage_dir = storage_dir or DATASETS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_date(value: str | None, default: datetime, name: str) -> datetime:
        """Parse a user date (YYYY-MM-DD or ISO datetime) into UTC datetime."""
        if value is None or not str(value).strip():
            return default
        s = str(value).strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid {name} '{value}': use YYYY-MM-DD or an ISO datetime."
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _resolve_range(start_date: str | None, end_date: str | None) -> tuple[datetime, datetime]:
        """Resolve the extraction window: defaults fill missing edges, and an
        inverted range is rejected loudly."""
        now = datetime.now(timezone.utc)
        end = DatasetService._parse_date(end_date, now, "end_date")
        default_start = end - timedelta(days=DEFAULT_RANGE_DAYS)
        start = DatasetService._parse_date(start_date, default_start, "start_date")
        if start >= end:
            raise ValueError(
                "start_date must be before end_date "
                f"(got {start_date or start.isoformat()} -> {end_date or end.isoformat()}).")
        return start, end

    def extract_dataset(
        self,
        symbol: str,
        timeframe: str = "15m",
        start_date: str | None = None,
        end_date: str | None = None,
        asset_class: str = "",
    ) -> dict[str, Any]:
        """Pull historical OHLCV data for an EXACT date range, convert to
        Parquet, and return metadata. ``start_date`` / ``end_date`` are
        ``YYYY-MM-DD`` (or ISO) strings; missing edges default sensibly."""
        symbol_clean = symbol.strip().upper()
        tf_clean = timeframe.strip()
        asset_class = (asset_class or "").strip().lower()

        start, end = self._resolve_range(start_date, end_date)

        # 0. Fail fast when the window is obviously over the safety cap (before
        #    hammering the providers with a multi-hundred-page fetch).
        span_seconds = int((end - start).total_seconds())
        est_bars = span_seconds // max(_TIMEFRAME_SECONDS.get(tf_clean, 900), 1)
        if est_bars > MAX_BARS:
            raise ValueError(
                f"Range {start.date()} -> {end.date()} would yield ~{est_bars:,} "
                f"{tf_clean} bars — over the {MAX_BARS:,} safety cap. Narrow the "
                "range or use a higher timeframe.")

        # 1. Fetch OHLCV rows for the exact window (MT5 range or fallback)
        rows = self._fetch_ohlcv(symbol_clean, tf_clean, start, end, asset_class)
        if not rows:
            raise ValueError(
                f"No market data could be fetched for {symbol_clean} ({tf_clean})")

        # 2. Keep only bars inside the window, chronological, de-duplicated.
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        rows = [r for r in rows if start_ts <= int(r.get("time", 0)) <= end_ts]
        rows.sort(key=lambda r: int(r.get("time", 0)))
        seen: set[int] = set()
        unique: list[dict] = []
        for r in rows:
            t = int(r.get("time", 0))
            if t in seen:
                continue
            seen.add(t)
            unique.append(r)
        if len(unique) > MAX_BARS:
            raise ValueError(
                f"Range {start.date()} -> {end.date()} yields {len(unique)} bars — "
                f"over the {MAX_BARS} safety cap. Narrow the range or use a higher "
                "timeframe.")
        rows = unique

        # 2. Build DataFrame
        df = pd.DataFrame(rows)

        # Standardize column names
        cols = {c: c.lower() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

        needed = ["time", "open", "high", "low", "close"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in fetched data")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Cast types
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Convert time integer (epoch) to ISO string / datetime64
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["time"], utc=True)

        df["timestamp"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Sort chronologically & drop duplicate timestamps
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Select standard column order
        export_df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        # 3. Save to Parquet file
        filename = f"{symbol_clean}_{tf_clean}.parquet"
        file_path = self.storage_dir / filename
        export_df.to_parquet(file_path, engine="pyarrow", index=False)

        # 4. Prepare Metadata Summary
        bar_count = len(export_df)
        start_time = export_df["timestamp"].iloc[0] if bar_count > 0 else ""
        end_time = export_df["timestamp"].iloc[-1] if bar_count > 0 else ""
        file_size_bytes = file_path.stat().st_size

        return {
            "id": f"{symbol_clean}_{tf_clean}",
            "symbol": symbol_clean,
            "timeframe": tf_clean,
            "bar_count": bar_count,
            "start_time": start_time,
            "end_time": end_time,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "file_path": str(file_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all available Parquet datasets in the storage directory."""
        datasets = []
        for path in self.storage_dir.glob("*.parquet"):
            try:
                # Read metadata using pyarrow for high performance without loading full file
                import pyarrow.parquet as pq

                meta = pq.read_metadata(path)
                bar_count = meta.num_rows

                # Read first and last row for time range
                table_head = pq.read_table(path, columns=["timestamp"])
                df_ts = table_head.to_pandas()
                start_time = str(df_ts["timestamp"].iloc[0]) if len(df_ts) > 0 else ""
                end_time = str(df_ts["timestamp"].iloc[-1]) if len(df_ts) > 0 else ""

                parts = path.stem.split("_", 1)
                symbol = parts[0]
                timeframe = parts[1] if len(parts) > 1 else ""

                file_size_bytes = path.stat().st_size
                updated_at = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat()

                datasets.append({
                    "id": path.stem,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bar_count": bar_count,
                    "start_time": start_time,
                    "end_time": end_time,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                    "file_path": str(path),
                    "updated_at": updated_at,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[DatasetService] Failed reading dataset {path.name}: {exc}")
        
        datasets.sort(key=lambda d: d["symbol"])
        return datasets

    def preview_dataset(self, dataset_id: str, limit: int = 50) -> dict[str, Any]:
        """Read a preview of the dataset rows + summary statistics."""
        file_path = self.storage_dir / f"{dataset_id}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")

        df = pd.read_parquet(file_path)
        bar_count = len(df)
        sample = df.tail(limit).to_dict(orient="records")

        stats = {
            "min_close": float(df["close"].min()) if bar_count else 0.0,
            "max_close": float(df["close"].max()) if bar_count else 0.0,
            "avg_volume": float(df["volume"].mean()) if bar_count else 0.0,
            "bar_count": bar_count,
            "start_time": str(df["timestamp"].iloc[0]) if bar_count else "",
            "end_time": str(df["timestamp"].iloc[-1]) if bar_count else "",
        }

        return {
            "id": dataset_id,
            "stats": stats,
            "sample": sample,
        }

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset Parquet file."""
        file_path = self.storage_dir / f"{dataset_id}.parquet"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    # ---------------------------------------------------------------- helpers

    def extract_watchlist(
        self,
        timeframe: str = "15m",
        start_date: str | None = None,
        end_date: str | None = None,
        asset_class: str = "",
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract a dataset for EVERY watchlist symbol in one call.

        Per-symbol failures are collected (not raised) so one bad symbol never
        aborts the rest of the batch.
        """
        if not symbols:
            raise ValueError(
                "The watchlist is empty — add assets to the watchlist first "
                "so extraction knows which symbols to pull.")
        datasets: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for sym in symbols:
            try:
                datasets.append(self.extract_dataset(
                    sym, timeframe, start_date, end_date, asset_class))
            except Exception as exc:  # noqa: BLE001 - collect per-symbol failure
                errors.append({"symbol": sym, "error": str(exc)})
        return {"datasets": datasets, "errors": errors}

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str,
        start: datetime, end: datetime, asset_class: str,
    ) -> list[dict]:
        """Attempt fetching OHLCV for the exact window from MT5 first, falling
        back to MarketDataService (paged Binance / Yahoo period range)."""
        adapter = MT5Adapter()
        # Attempt MT5 range pull first for forex/commodities/metals. The pull
        # window is padded by 1 day on the start side because MT5 interprets
        # naive datetimes as its own server-local time on some builds (a 2-3h
        # offset could otherwise under-fetch the boundary); the exact window
        # filter below trims the padding.
        if asset_class not in ("crypto", "cripto"):
            try:
                rates = adapter.pull_ohlcv_range(
                    symbol, timeframe, start - timedelta(days=1), end)
                if rates:
                    return rates
            except Exception:  # noqa: BLE001, S110 - MT5 is optional; fall back
                pass

        # Fallback to MarketDataService (Binance for crypto / Yahoo for stocks/FX)
        import asyncio

        async def _fetch():
            md = MarketDataService()
            try:
                candles, _provider = await md.fetch_range(
                    symbol, timeframe, int(start.timestamp()), int(end.timestamp()))
                # Map keys to standard dict
                return [
                    {
                        "time": c.get("time", 0),
                        "open": c.get("open", 0.0),
                        "high": c.get("high", 0.0),
                        "low": c.get("low", 0.0),
                        "close": c.get("close", 0.0),
                        "volume": c.get("volume", 0.0),
                    }
                    for c in candles
                ]
            finally:
                await md.close()

        try:
            return asyncio.run(_fetch())
        except Exception:  # noqa: BLE001
            return []
