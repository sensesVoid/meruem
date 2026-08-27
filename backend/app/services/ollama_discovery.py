"""Ollama endpoint + model auto-detection.

Runs at launch (non-blocking) so the local LLM stack never hard-codes a dead
URL or a model Ollama hasn't pulled. We probe candidate endpoints, persist the
first reachable one as the ``ollama_url`` setting, cache the installed model
list, and re-point the LOCAL agents (scanner / regime) at a model that actually
exists. Cloud-assigned agents are left untouched — they route through the
universal chain which already falls back to an installed Ollama model.

Everything network-facing is best-effort and short-timeout, and results are
TTL-cached so an unreachable or slow Ollama never stalls the app.
"""

import os
import threading
import time

import requests

from app.core.config import settings as cfg

_PROBE_TIMEOUT = 1.5        # seconds per /api/tags probe
_DETECT_TTL = 60.0          # cache endpoint detection for this long
_MODELS_TTL = 60.0          # cache the installed model list for this long

# Order-preference when the configured model is not installed.
_PREFERENCE = [
    "qwen3:4b",
    "qwen3:1.7b",
    "phi4-mini:latest",
    "gemma3:4b",
    "qwen2.5-coder:3b-instruct-q4_K_M",
]

_LOCK = threading.Lock()
_detect: dict[str, object] = {}  # {"url": str, "ts": float}
_models: dict[str, tuple[float, list[str]]] = {}
_report: dict = {}


def _normalize(host: str) -> str:
    """Coerce an Ollama host/env value into a URL ('127.0.0.1:11434' -> http://…)."""
    host = (host or "").strip().rstrip("/")
    if not host:
        return ""
    return host if "://" in host else f"http://{host}"


def candidate_urls() -> list[str]:
    """Ordered, deduped endpoints to try: configured -> config default ->
    OLLAMA_HOST env -> common localhost variants."""
    from app.services.model_config import get as model_get

    urls: list[str] = []
    configured = str(model_get("ollama_url") or "").strip().rstrip("/")
    if configured:
        urls.append(configured)
    if cfg.ollama_url and cfg.ollama_url.rstrip("/") not in urls:
        urls.append(cfg.ollama_url.rstrip("/"))
    env = _normalize(os.environ.get("OLLAMA_HOST", ""))
    if env and env not in urls:
        urls.append(env)
    for extra in ("http://localhost:11434", "http://127.0.0.1:11434",
                  "http://localhost:11435", "http://127.0.0.1:11435"):
        if extra not in urls:
            urls.append(extra)
    return urls


def _probe(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/api/tags", timeout=_PROBE_TIMEOUT)
        return bool(resp.ok)
    except Exception:  # noqa: BLE001 - any failure = not reachable
        return False


def detect_url(force: bool = False) -> str:
    """First reachable Ollama endpoint (TTL-cached); '' when unreachable.

    ``force=True`` bypasses the cache — used by the launch reconciler and the
    manual discover endpoint so they always probe fresh.
    """
    if not force:
        with _LOCK:
            hit = dict(_detect)
        if "ts" in hit and time.time() - float(hit.get("ts", 0)) < _DETECT_TTL:
            return str(hit.get("url") or "")
    for u in candidate_urls():
        if _probe(u):
            with _LOCK:
                _detect.update({"url": u, "ts": time.time()})
            return u
    with _LOCK:
        _detect.update({"url": "", "ts": time.time()})
    return ""


def list_installed(url: str | None = None) -> list[str]:
    """Installed model names from ``/api/tags`` (TTL-cached); [] on failure."""
    url = url or detect_url()
    if not url:
        return []
    now = time.time()
    with _LOCK:
        hit = _models.get(url)
        if hit and now - hit[0] < _MODELS_TTL:
            return hit[1]
    try:
        resp = requests.get(f"{url}/api/tags", timeout=_PROBE_TIMEOUT)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        names = [n for n in names if n]
    except Exception:  # noqa: BLE001
        names = []
    with _LOCK:
        _models[url] = (now, names)
    return names


def _pick(installed: list[str]) -> str:
    """Best chat-capable model from an installed list."""
    for m in _PREFERENCE:
        if m in installed:
            return m
    for m in installed:
        if "embed" not in m and "nomic" not in m:
            return m
    return installed[0] if installed else ""


def best_installed(url: str | None = None) -> str:
    return _pick(list_installed(url))


def resolve_model(preferred: str, url: str | None = None) -> str:
    """``preferred`` model if it is actually installed, else the best installed one.

    Returns ``preferred`` unchanged when Ollama is unreachable so callers still
    surface a configurable name (status carries the connection error)."""
    installed = list_installed(url)
    if not installed:
        return preferred
    if preferred and preferred in installed:
        return preferred
    return _pick(installed)


def autoconfigure() -> dict:
    """Probe Ollama, persist the reachable endpoint, re-point local agents.

    Non-destructive: cloud model assignments are preserved. Returns a report
    for logging and the UI. Safe to call at launch and on demand."""
    import json

    from sqlmodel import Session, select

    from app.core.db import engine
    from app.models.entities import Setting
    from app.services.model_config import get as model_get
    from app.services.settings_service import SettingsService

    report = {
        "url_before": str(model_get("ollama_url") or ""),
        "url": "",
        "models": [],
        "fixed": {},
        "error": "",
    }
    url = detect_url(force=True)
    report["url"] = url or report["url_before"]
    if not url:
        report["error"] = "no reachable Ollama endpoint"
        _store_report(report)
        return report

    installed = list_installed(url)
    report["models"] = installed

    # Persist the resolved endpoint so every consumer (scanner, universal
    # chain, status) agrees on the same host.
    with Session(engine) as s:
        SettingsService(s).set("ollama_url", url)

    # Re-point the LOCAL agents at an installed model when their current
    # assignment is missing. Cloud assignments ride along untouched.
    fixed: dict[str, dict] = {}
    with Session(engine) as s:
        row = s.exec(select(Setting).where(Setting.key == "agent_models")).first()
        data: dict = {}
        if row is not None:
            try:
                parsed = json.loads(row.value_json)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:  # noqa: BLE001
                data = {}
        best = _pick(installed)
        for agent_id in ("scanner", "regime"):
            current = str(data.get(agent_id) or "")
            resolved = current if current in installed else best
            if best and resolved and resolved != current:
                data[agent_id] = resolved
                fixed[agent_id] = {"from": current or "(catalog default)", "to": resolved}
        if fixed:
            SettingsService(s).set("agent_models", data)

    report["fixed"] = fixed
    _store_report(report)
    return report


def latest_report() -> dict:
    """The most recent discovery result (empty until autoconfigure runs)."""
    with _LOCK:
        return dict(_report)


def _store_report(report: dict) -> None:
    with _LOCK:
        _report.clear()
        _report.update(report)