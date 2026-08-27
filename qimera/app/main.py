"""QiMera API — offline research/optimization extension, opened from Meruem.

Serves the QiMera web UI (browser tab) and exposes the export artifact store.
The one cross-system action is ``POST /api/inject`` which pushes a symbol's
latest export artifact into Meruem's ML-artifact context store so the Meruem
scanner consumes it as context on the next scan of that symbol.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import requests
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Explicit package paths: QiMera ships as its own ``app`` package but lives in
# the same repo as Meruem's ``backend/app`` package. ``qimera.app.*`` (this
# project) and ``backend.app.*`` (Meruem services) are imported by full name so
# the two ``app`` packages can never shadow each other on sys.path.
from qimera.app.config import settings
from qimera.app.export_store import export_root, latest_manifest, list_exports, new_manifest

API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    export_root()
    yield


app = FastAPI(title=settings.app_name, version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All routes live under /api (same contract as Meruem's backend and the Vite
# dev proxy), health/status included.
api_router = APIRouter()


class ExportRequest(BaseModel):
    asset: str
    regime: str | None = None
    note: str = ""


class InjectRequest(BaseModel):
    asset: str


class BenchmarkRequest(BaseModel):
    bars: int = 20000
    lookback: int = 200
    runs: int = 3


class InterpretRequest(BaseModel):
    concept: str


class SynthesizeRequest(BaseModel):
    n_proposals: int = 12
    k: int = 10


class ClassifyRequest(BaseModel):
    n_windows: int = 3


class ChampionPromoteRequest(BaseModel):
    asset: str
    regime: str = "ANY"
    source: str = "gp"
    strategy: str = ""
    params: list | None = None
    run_id: str = ""
    metrics: dict | None = None


class ChampionRollbackRequest(BaseModel):
    asset: str
    regime: str = "ANY"
    target_id: int | None = None


@api_router.get("/health")
def health():
    return {"ok": True, "app": settings.app_name, "version": API_VERSION}


@api_router.get("/status")
def status():
    return {
        "app": settings.app_name,
        "version": API_VERSION,
        "export_dir": str(export_root()),
        "exports": list_exports(),
        "meruem": {"api": settings.meruem_api},
    }


@api_router.get("/export")
def list_export():
    return {"exports": list_exports()}


@api_router.get("/export/{asset}")
def get_export(asset: str):
    manifest = latest_manifest(asset)
    if manifest is None:
        raise HTTPException(404, "no exported artifact for this asset")
    return manifest


@api_router.post("/export")
def create_export(payload: ExportRequest):
    asset = payload.asset.strip().upper()
    if not asset:
        raise HTTPException(400, "asset is required")
    manifest = new_manifest(asset, regime=payload.regime, note=payload.note)
    return {"manifest": manifest}


@api_router.post("/inject")
def inject(payload: InjectRequest):
    """Push the latest artifact for ``asset`` into Meruem's ML-artifact store."""
    asset = payload.asset.strip()
    manifest = latest_manifest(asset)
    if manifest is None:
        raise HTTPException(404, "no exported artifact for this asset")
    version = manifest.get("registry_id") or manifest.get("artifact_id", "")[:10]
    filename = f"qimera_{manifest['asset']}_{version}.json"
    if not settings.meruem_api:
        raise HTTPException(503, "meruem_api is not configured")
    try:
        resp = requests.post(
            f"{settings.meruem_api}/strategies/ml-artifacts",
            params={
                "filename": filename,
                "symbol": manifest["asset"],
                "content": json.dumps(manifest, indent=2),
            },
            timeout=5.0,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Meruem unreachable: {exc}") from exc
    if not resp.ok:
        raise HTTPException(502, f"Meruem rejected injection: {resp.status_code} {resp.text}")
    return {"injected": True, "filename": filename, "meruem_status": resp.status_code}


# ---------------------------------------------------------------------------
# Julia compute layer (§6.4 / M2) — always gracefully degradable
# ---------------------------------------------------------------------------


@api_router.get("/julia/status")
def julia_status():
    from qimera.app.julia_bridge import status as bridge_status

    return bridge_status()


@api_router.get("/data/ohlcv")
async def data_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 500,
                     start: int = 0, end: int = 0,
                     source: str = "yfinance", deep: bool = False,
                     compact: bool = False) -> dict:
    """On-demand OHLCV from ``yfinance`` or the local ``mt5`` terminal.

    Returns the same candle-dict contract as Meruem's ``/candles`` (time/open/
    high/low/close/volume + provider) so QiMera research and the UI can fetch
    any symbol/timeframe/date-window directly from either source.

    ``deep=true`` (MT5 only) pages the terminal's full intraday history in date
    windows — the only path to years of 1-minute candles (Yahoo caps 1m at
    ~30 days). ``compact=true`` returns flat arrays (``time``/``open``/...)
    instead of a list of dicts — the lightweight form for large fetches.
    """
    from qimera.app.data_sources import SOURCES, fetch_ohlcv

    if source not in SOURCES:
        raise HTTPException(400, f"unknown source {source!r}; expected one of {SOURCES}")
    if deep and source != "mt5":
        raise HTTPException(400, "deep history is only supported for source='mt5'")
    try:
        candles = await asyncio.to_thread(
            fetch_ohlcv, symbol, timeframe, limit,
            start or None, end or None, source, deep, compact,
        )
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "timeframe": timeframe, "candles": [],
                "provider": source, "error": str(exc)}
    if compact and isinstance(candles, dict):
        candles = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in candles.items()
        }
    return {"symbol": symbol, "timeframe": timeframe, "candles": candles,
            "provider": source, "compact": compact}


@api_router.post("/benchmark")
def benchmark(payload: BenchmarkRequest):
    """Julia vs pure-Python benchmark on the O(n·lookback) sweep feature.

    Returns parity checks (both implementations must agree) and the speed-up, or
    503 when the Julia runtime/juliacall is not installed.
    """
    from qimera.app.julia_bridge import JuliaUnavailable
    from qimera.app.julia_bridge import benchmark as run_benchmark

    try:
        result = run_benchmark(bars=payload.bars, lookback=payload.lookback,
                               runs=payload.runs)
    except JuliaUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return result


@api_router.post("/interp/interpret")
def interpret(payload: InterpretRequest):
    """M6: interpret a descriptive concept into a typed strategy blueprint.

    Runs the bounded Indicator Interpretation Agent (LLM concept -> M5 schema,
    optional web search with provenance logging, sandboxed 2s backtest guard).
    Degrades to deterministic keyword mapping + template code offline.
    """
    concept = payload.concept.strip()
    if not concept:
        raise HTTPException(400, "concept is required")
    from qimera.app.interp_agent import InterpAgent

    return InterpAgent().interpret(concept)


@api_router.post("/synthesis/run")
def synthesis_run(payload: SynthesizeRequest):
    """M7: agentic feature synthesis over the metadata registry.

    Evolution agent proposes typed feature mutations/composites, the coder agent
    emits vectorized expressions validated by the sandboxed guard, and the pool
    scores them on a synthetic dataset. Returns the top-k candidates.
    """
    if not 1 <= payload.n_proposals <= 200:
        raise HTTPException(400, "n_proposals must be in [1, 200]")
    from qimera.app.interp_agent import _synthetic_df
    from qimera.app.synthesis import run_synthesis

    return run_synthesis(_synthetic_df(), n_proposals=payload.n_proposals,
                         k=payload.k)


@api_router.post("/classifier/validate")
def classifier_validate(payload: ClassifyRequest):
    """M8: rolling-window XGBoost validation on synthetic data.

    Trains the regime-aware classifier per window, gates low-confidence calls,
    and returns the per-window OOS accuracy + feature-drift report. 503 when
    xgboost is not installed.
    """
    if not 2 <= payload.n_windows <= 8:
        raise HTTPException(400, "n_windows must be in [2, 8]")
    from qimera.app.classifier import available as xgb_available

    if not xgb_available():
        raise HTTPException(503, "xgboost is not installed")
    from qimera.app.classifier import rolling_validate
    from qimera.app.interp_agent import _synthetic_df

    return rolling_validate(_synthetic_df(), n_windows=payload.n_windows)


@api_router.get("/champions")
def champions_list(asset: str | None = None, regime: str | None = None,
                   promotion: str | None = None):
    """M9: list champions from the SQLite registry (filterable)."""
    from qimera.app.champions import ChampionsRegistry

    reg = ChampionsRegistry()
    try:
        return {"champions": reg.list_champions(asset=asset, regime=regime,
                                                promotion=promotion)}
    finally:
        reg.close()


@api_router.get("/champions/current")
def champions_current(asset: str, regime: str = "ANY"):
    """M9: the currently promoted champion for an asset+regime."""
    from qimera.app.champions import ChampionsRegistry

    reg = ChampionsRegistry()
    try:
        champ = reg.current(asset, regime)
        if champ is None:
            raise HTTPException(404, "no promoted champion for this asset/regime")
        return champ
    finally:
        reg.close()


@api_router.post("/champions/promote")
def champions_promote(payload: ChampionPromoteRequest):
    """M9: register a champion (auto-supersedes the previous for asset+regime)."""
    from qimera.app.champions import ChampionsRegistry

    reg = ChampionsRegistry()
    try:
        return reg.promote(
            asset=payload.asset, regime=payload.regime, source=payload.source,
            strategy=payload.strategy, params=payload.params,
            run_id=payload.run_id, metrics=payload.metrics)
    finally:
        reg.close()


@api_router.post("/champions/rollback")
def champions_rollback(payload: ChampionRollbackRequest):
    """M9: restore a superseded champion (rollback utility)."""
    from qimera.app.champions import ChampionsRegistry

    reg = ChampionsRegistry()
    try:
        champ = reg.rollback(payload.asset, payload.regime, payload.target_id)
        if champ is None:
            raise HTTPException(404, "no superseded champion to roll back to")
        return champ
    finally:
        reg.close()

app.include_router(api_router, prefix="/api")
