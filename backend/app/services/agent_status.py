"""Live status of AI agents and the models they resolve to.

Surfaces what the workspace is actually running so the Vault UI can show
green/red readiness: the local Ollama connector, the ordered cloud providers,
and the resolved model for each agent role (Scanner / Validator / QA /
Refinement).
"""

import time

from app.services.model_config import get as model_get
from app.services.model_config import get_agent_catalog, get_agent_model, get_providers
from app.services.vault import VaultService


def _ollama_status() -> dict:
    from app.agents.ollama_client import OllamaClient

    url = model_get("ollama_url")
    models: list[str] = []
    error = ""
    start = time.time()
    try:
        models = OllamaClient(base_url=url).list_models()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {
        "url": url,
        "reachable": not error,
        "models": models,
        "latency_ms": int((time.time() - start) * 1000),
        "error": error,
        "scanner_model": model_get("ollama_scanner_model"),
        "validator_model": model_get("ollama_validator_model"),
    }


def _providers_status() -> list[dict]:
    out: list[dict] = []
    for p in get_providers():
        ref = p.get("key_ref", "")
        if ":" in ref:
            ns, user = ref.split(":", 1)
        else:
            ns, user = "llm", ref
        key_set = bool(VaultService.get(ns, user))
        out.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "base_url": p.get("base_url"),
            "model": p.get("model"),
            "enabled": bool(p.get("enabled", True)),
            "key_set": key_set,
            "ready": bool(p.get("enabled", True)) and key_set,
        })
    return out


def agent_status() -> dict:
    from app.services.ollama_discovery import latest_report

    ollama = _ollama_status()
    providers = _providers_status()
    cloud_ready = any(p["ready"] for p in providers)
    ollama_ready = ollama["reachable"] and bool(ollama["scanner_model"])

    first_cloud = next((p for p in providers if p["ready"]), None)
    fallback_model = ollama["validator_model"]
    catalog = {a["id"]: a for a in get_agent_catalog()}

    def _agent(agent_id, resolved_model, ready, note, engine_label, name, role) -> dict:
        return {
            "id": agent_id,
            "name": name,
            "role": role,
            "engine": engine_label,
            "model": resolved_model,
            "ready": ready,
            "status": "running" if ready else "blocked",
            "note": note,
        }

    def _cloud_engine() -> str:
        return "Cloud chain → Ollama fallback"

    agents = []
    for agent_id in ("scanner", "validator", "risk_manager", "qa", "news", "regime"):
        entry = catalog.get(agent_id, {})
        name = entry.get("name", agent_id)
        role = entry.get("role", "")

        if agent_id == "scanner":
            model = get_agent_model("scanner") or ollama["scanner_model"]
            ready = ollama_ready
            note = "Ollama unreachable" if not ollama["reachable"] else (
                "no scanner model selected" if not ollama["scanner_model"] else "local LLM scanner (raw chat)")
            label = "Ollama (local LLM)"
        elif agent_id == "regime":
            model = get_agent_model("regime") or "indicator-based"
            ready = True
            note = "deterministic indicators, scheduler refreshes every 5 minutes"
            label = "Deterministic indicators (5-min cycle)"
        else:
            model = get_agent_model(agent_id) or (first_cloud["model"] if first_cloud else fallback_model)
            ready = cloud_ready or ollama["reachable"]
            note = "add a cloud key for the online chain" if not cloud_ready else "ready"
            label = _cloud_engine()
        agents.append(_agent(agent_id, model, ready, note, label, name, role))

    return {
        "ollama": ollama,
        "providers": providers,
        "agents": agents,
        "cloud_ready": cloud_ready,
        "ollama_discovery": latest_report(),
    }
