import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.db import init_db
from app.services.scheduler import start_scheduler, stop_scheduler


def _run_ollama_detection() -> None:
    """Best-effort Ollama endpoint + model self-configuration at launch.

    Runs on a daemon thread so a slow/unreachable Ollama never delays startup.
    Retries briefly in case Ollama is still coming up; the orchestrator/status
    paths probe live anyway, so a miss here is never fatal.
    """
    import app.services.ollama_discovery as discovery

    for attempt in range(1, 4):
        report = discovery.autoconfigure()
        if not report.get("error"):
            print(f"[ollama] detected {report['url']}: "
                  f"{len(report['models'])} model(s), "
                  f"re-pointed {len(report['fixed'])} agent(s)")
            break
        if attempt < 3:
            print(f"[ollama] detection attempt {attempt}/3: "
                  f"{report.get('error')} — retrying")
            time.sleep(3)
    else:
        print(f"[ollama] detection failed at launch: {report.get('error')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    if settings.ollama_detect_on_launch:
        threading.Thread(target=_run_ollama_detection, daemon=True).start()
    yield
    stop_scheduler()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
