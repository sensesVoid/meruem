"""QiMera Data Layer (QiMera.md §6.1 / M1).

Ingests historical OHLCV data — from Meruem's MarketDataService chain or a
direct on-demand source (yfinance / local MT5 terminal, see
``qimera.app.data_sources``) — runs the regime filter contract, evaluates data
quality gates, tags session boundaries, and saves/loads snappy-compressed
Parquet files under ``qimera/data/``.
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure backend and repository root are in path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.market_data import MarketDataService
from app.core.db import get_session
from app.models.entities import Asset
from sqlmodel import Session, select

from qimera.app.regime_contract import compute_historical_regimes

DATA_ROOT = _REPO_ROOT / "qimera" / "data"


def _timeframe_seconds(timeframe: str) -> int:
    """Returns expected seconds per bar for standard timeframes."""
    tf = timeframe.lower()
    if tf in ("m1", "1m"):
        return 60
    elif tf in ("m5", "5m"):
        return 300
    elif tf in ("m15", "15m"):
        return 900
    elif tf in ("h1", "1h"):
        return 3600
    elif tf in ("h4", "4h"):
        return 14400
    elif tf in ("d1", "1d"):
        return 86400
    elif tf in ("w1", "1w"):
        return 604800
    return 900  # Default to 15m


def tag_session_boundaries(df: pd.DataFrame, timeframe: str) -> pd.Series:
    """Tags session boundaries (e.g. start of a new trading day/session).
    
    Returns a boolean Series.
    """
    timestamps = pd.to_datetime(df["time"], unit="s", utc=True)
    
    # Session boundary heuristic:
    # 1. Day changes
    # 2. Time difference is much larger than expected (weekend gap)
    day_changed = timestamps.dt.date != timestamps.dt.date.shift(1)
    
    # Or gap is larger than expected timeframe seconds (e.g. weekend gap)
    expected_sec = _timeframe_seconds(timeframe)
    gap_detected = df["time"].diff() > (expected_sec * 2)
    
    # Mark index 0 as boundary
    boundary = day_changed | gap_detected
    boundary.iloc[0] = True
    return boundary


class DataQualityGateError(ValueError):
    pass


def run_quality_gates(df: pd.DataFrame, timeframe: str, min_bars: int = 240) -> dict:
    """Validates the dataset against required quality checks.
    
    Raises DataQualityGateError if checks fail. Returns statistics dict.
    """
    n_bars = len(df)
    if n_bars < min_bars:
        raise DataQualityGateError(
            f"Insufficient history: dataset has {n_bars} bars, "
            f"but minimum of {min_bars} is required."
        )
        
    # Check for NaN values in OHLCV
    for col in ["open", "high", "low", "close"]:
        nans = df[col].isna().sum()
        if nans > 0:
            raise DataQualityGateError(f"Data contains {nans} NaN values in column '{col}'")
            
    # Check for negative/zero prices
    if (df["close"] <= 0).any():
        raise DataQualityGateError("Dataset contains non-positive close prices.")

    # Gap detection
    expected_sec = _timeframe_seconds(timeframe)
    time_diffs = df["time"].diff().dropna()
    gaps = time_diffs[time_diffs > (expected_sec * 1.5)]
    
    gaps_count = len(gaps)
    total_gap_sec = gaps.sum() if gaps_count > 0 else 0
    
    # Tag warnings if large gaps exist (e.g., missing data during active hours, not weekends)
    # We warn but don't fail for weekend gaps
    active_gaps = time_diffs[time_diffs > (expected_sec * 5)]
    active_gaps_count = len(active_gaps)
    
    return {
        "total_bars": n_bars,
        "gaps_count": gaps_count,
        "active_gaps_count": active_gaps_count,
        "total_gap_duration_sec": int(total_gap_sec),
        "min_time": int(df["time"].min()),
        "max_time": int(df["time"].max()),
    }


async def ingest_market_data(symbol: str, timeframe: str = "15m", limit: int = 5000,
                             preferred: str | None = None,
                             source: str | None = None,
                             persist: bool = True) -> dict:
    """Fetches market data, applies regime tag and quality gates, saves to Parquet.

    ``source`` selects a direct on-demand source — ``"yfinance"`` or ``"mt5"``
    (see ``qimera.app.data_sources``). When ``None``, Meruem's MarketDataService
    provider chain is used and ``preferred`` still applies.

    ``persist=False`` runs the full transform (regime tagging + quality gates)
    but returns the frame in memory without writing anything to the parquet
    store — the lightweight research path. Note the return shapes differ:
    persist=True returns the ``stats`` dict (with ``file_path``/``persisted``),
    persist=False returns ``{symbol, timeframe, bars, persisted, df, stats}``
    with the transformed DataFrame in ``df``.
    """
    symbol = symbol.upper().strip()
    timeframe = timeframe.strip()

    # 1. Fetch candles (direct on-demand source, or Meruem's provider chain)
    if source is not None:
        from qimera.app.data_sources import SOURCE_MT5, SOURCE_YFINANCE, fetch_ohlcv

        if source not in (SOURCE_YFINANCE, SOURCE_MT5):
            raise ValueError(f"unknown data source {source!r}")
        candles = await asyncio.to_thread(
            fetch_ohlcv, symbol, timeframe, limit, None, None, source)
    else:
        mds = MarketDataService()
        try:
            data = await mds.fetch(symbol=symbol, timeframe=timeframe, limit=limit, preferred=preferred)
        finally:
            await mds.close()
        candles = data.get("candles") or []
    if not candles:
        raise ValueError(f"No market data returned for symbol {symbol} ({timeframe})")
        
    # Sort by time ascending
    candles = sorted(candles, key=lambda x: x["time"])
    
    # 2. Compute historical regimes
    regimes = compute_historical_regimes(candles)
    
    # Map back by timestamp
    regime_map = {r["timestamp"]: r for r in regimes}
    
    # 3. Build DataFrame
    rows = []
    for c in candles:
        t = c["time"]
        r = regime_map.get(t, {"regime_label": "UNDEFINED", "confidence_score": 0.5})
        rows.append({
            "time": t,
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
            "regime_label": str(r["regime_label"]),
            "confidence_score": float(r["confidence_score"]),
        })
        
    df = pd.DataFrame(rows)
    
    # 4. Tag boundaries and run quality gates
    df["session_boundary"] = tag_session_boundaries(df, timeframe)
    stats = run_quality_gates(df, timeframe)

    # 5a. Lightweight path: transform in memory, never touch the parquet store.
    if not persist:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": int(len(df)),
            "persisted": False,
            "df": df,
            "stats": stats,
        }

    # 5b. Save to snappy-compressed parquet
    dir_path = DATA_ROOT / symbol / timeframe
    dir_path.mkdir(parents=True, exist_ok=True)
    parquet_file = dir_path / "data.parquet"

    df.to_parquet(parquet_file, index=False, compression="snappy")

    stats["file_path"] = str(parquet_file)
    stats["symbol"] = symbol
    stats["timeframe"] = timeframe
    stats["persisted"] = True
    return stats


def load_market_data(symbol: str, timeframe: str = "15m") -> pd.DataFrame:
    """Loads the cached Parquet dataset for a symbol + timeframe."""
    symbol = symbol.upper().strip()
    timeframe = timeframe.strip()
    
    parquet_file = DATA_ROOT / symbol / timeframe / "data.parquet"
    if not parquet_file.is_file():
        raise FileNotFoundError(f"No cached parquet file found at {parquet_file}")
        
    return pd.read_parquet(parquet_file)


def list_ingested_assets() -> list[dict]:
    """Scans DATA_ROOT and returns a list of ingested symbol/timeframe configurations."""
    out = []
    if not DATA_ROOT.is_dir():
        return out
        
    for symbol_dir in sorted(DATA_ROOT.iterdir()):
        if not symbol_dir.is_dir():
            continue
        for tf_dir in sorted(symbol_dir.iterdir()):
            if not tf_dir.is_dir():
                continue
            parquet_file = tf_dir / "data.parquet"
            if parquet_file.is_file():
                try:
                    df = pd.read_parquet(parquet_file)
                    out.append({
                        "symbol": symbol_dir.name,
                        "timeframe": tf_dir.name,
                        "bars": len(df),
                        "start_time": int(df["time"].min()),
                        "end_time": int(df["time"].max()),
                        "file_size": parquet_file.stat().st_size,
                    })
                except Exception:  # noqa: BLE001
                    pass
    return out
