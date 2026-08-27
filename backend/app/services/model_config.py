"""Runtime LLM / model configuration.

The user picks models in the Vault UI. To make the pipeline use that choice
(cached briefly) without threading DB sessions through every agent, this
module reads the relevant Settings rows with a short TTL and falls back to
config defaults.
"""

import json
import time

from sqlmodel import Session, select

from app.core.config import settings as cfg
from app.core.db import engine
from app.models.entities import Setting

_TTL = 15.0  # seconds
_cache: dict[str, tuple[float, object | None]] = {}

# Setting key -> config default used as fallback.
_MODEL_KEYS: dict[str, object] = {
    "ollama_url": cfg.ollama_url,
    "ollama_scanner_model": cfg.ollama_scanner_model,
    "ollama_validator_model": "qwen3:4b",
}

# Default provider list used when no `llm_providers` setting row exists.
DEFAULT_PROVIDERS: list[dict] = [
    {
        "id": "gemini",
        "name": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "enabled": True,
        "key_ref": "llm:gemini",
    },
    {
        "id": "groq",
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "enabled": True,
        "key_ref": "llm:groq",
    },
]


def get_providers() -> list[dict]:
    """Return the ordered list of enabled LLM providers (or defaults)."""
    value = _read("llm_providers")
    if isinstance(value, list) and value and all(isinstance(p, dict) and p.get("id") for p in value):
        return value
    return DEFAULT_PROVIDERS


def _read(key: str) -> object | None:
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    value: object | None = None
    try:
        with Session(engine) as session:
            row = session.exec(select(Setting).where(Setting.key == key)).first()
            if row is not None:
                value = json.loads(row.value_json)
    except Exception:  # noqa: BLE001
        value = None
    _cache[key] = (now, value)
    return value


def get(key: str) -> str:
    value = _read(key)
    if value is None:
        value = _MODEL_KEYS.get(key)
    return str(value) if value is not None else ""


# --------------------------------------------------------------------------
# Agent catalog (task #12)
#
# Every AI "agent" is described once so the Vault UI can assign a model to each
# and so the pipeline can resolve the model an agent should use. An agent's
# engine is either "local" (Ollama) or "cloud" (the universal provider chain).
# Refresh times follow the task spec (scanner/regime 5m, news 15m, ...).
# --------------------------------------------------------------------------

AGENT_CATALOG: list[dict] = [
    {
        "id": "scanner",
        "name": "Scanner Agent",
        "role": "Scans the market for potential entry setups.",
        "engine": "local",
        "refresh": "5 min / on click",
        "default_model": "qwen3:4b",
        "recommended_model": "qwen3:4b",
        "tip": "Fast, disciplined setup detector. A compact local model (qwen3:4b) is plenty; it only reads indicators + rules.",
    },
    {
        "id": "validator",
        "name": "Master Trader / Validator Agent",
        "role": "Technical analysis referencing the Regime and News agents to validate entries and exit signals.",
        "engine": "cloud",
        "refresh": "on-demand",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "Does the deeper cross-market analysis; a strong cloud model (Gemini 3.5 or Groq 70b) improves judgement.",
    },
    {
        "id": "regime",
        "name": "Regime Agent",
        "role": "Filters market regime: trend direction, trend strength, volatility, momentum/exhaustion, breadth.",
        "engine": "local",
        "refresh": "5 min",
        "default_model": "qwen3:4b",
        "recommended_model": "qwen3:4b",
        "tip": "Currently indicator-driven, not LLM. Model only used if enrolled in the LLM chain in a future phase.",
    },
    {
        "id": "news",
        "name": "News Agent",
        "role": "Filters news to watchlist assets, produces AI summaries + fundamental analysis (no raw headlines).",
        "engine": "cloud",
        "refresh": "15 min",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "Summarization is cost-sensitive at 15-min cadence; a fast flash-class model is a good default.",
    },
    {
        "id": "ingestion",
        "name": "Ingestion Agent",
        "role": "Aids RAG for Synthesizer and Master Trader by chunking trading concepts and info.",
        "engine": "local",
        "refresh": "on-demand",
        "default_model": "all-MiniLM-L6-v2",
        "recommended_model": "all-MiniLM-L6-v2",
        "tip": "Embedding model used for RAG chunking; not an LLM chat model.",
    },
    {
        "id": "synthesizer",
        "name": "Synthesizer Agent",
        "role": "Synthesizes the strategy concepts library into the best trade setup grounded on ingested concepts.",
        "engine": "cloud",
        "refresh": "on-demand",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "Concept grounding is reasoning-heavy; a strong cloud model yields more faithful setups.",
    },
    {
        "id": "ltr",
        "name": "LoseTradeReview Agent",
        "role": "Reviews the most recent losing trades, writing a journal of error and a lesson fed to the synthesizer so strategies improve.",
        "engine": "cloud",
        "refresh": "on click",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "On-demand review of losses; a flash-class model keeps each click cheap.",
    },
    {
        "id": "qa",
        "name": "Strategy QA",
        "role": "Proofread + anti-RAG-contamination filter for new strategies.",
        "engine": "cloud",
        "refresh": "on-demand",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "Short deterministic checks; a fast model is ideal.",
    },
    {
        "id": "risk_manager",
        "name": "Risk Manager Agent",
        "role": "Analyses account states, outputs risk plan and lot sizing, enforces RiskInterceptor metrics, clearance gate for trades.",
        "engine": "cloud",
        "refresh": "on-demand / pre-trade",
        "default_model": "gemini-3.5-flash-lite",
        "recommended_model": "gemini-3.5-flash-lite",
        "tip": "Evaluates risk guardrails deterministically and generates an AI narrative risk plan advisory.",
    },
]

_AGENT_MODELS_DEFAULT: dict[str, str] = {
    a["id"]: a["default_model"] for a in AGENT_CATALOG
}


def _agent_models() -> dict:
    value = _read("agent_models")
    if isinstance(value, dict):
        return value
    return dict(_AGENT_MODELS_DEFAULT)


# Legacy setting key consulted when no per-agent override exists, so a training
# model the user set via the old per-provider model fields keeps being honoured.
_LEGACY_MODEL_KEY: dict[str, str] = {
    "scanner": "ollama_scanner_model",
    "validator": "ollama_validator_model",
    "qa": "ollama_validator_model",
    "news": "ollama_validator_model",
    "regime": "ollama_scanner_model",
    "synthesizer": "ollama_validator_model",
    "ltr": "ollama_validator_model",
    "risk_manager": "ollama_validator_model",
    "ingestion": "",
}


def get_agent_model(agent_id: str) -> str:
    """Resolve the model an LLM agent should use.

    Priority: explicit per-agent override (``agent_models``) -> legacy Ollama
    setting the agent used historically -> agent catalog default -> empty.
    """
    assigned = _agent_models().get(agent_id)
    if assigned:
        return str(assigned)
    legacy_key = _LEGACY_MODEL_KEY.get(agent_id, "")
    if legacy_key:
        legacy = _read(legacy_key)
        if legacy:
            return str(legacy)
    return _AGENT_MODELS_DEFAULT.get(agent_id, "")


def get_agent_catalog() -> list[dict]:
    """Catalog with the currently assigned model per agent merged in."""
    assigned = _agent_models()
    out = []
    for a in AGENT_CATALOG:
        out.append({
            **a,
            "assigned_model": assigned.get(a["id"]) or a["default_model"],
        })
    return out


def list_provider_models(provider: dict) -> list[str]:
    """List available models from an OpenAI-compatible provider's /models.

    Requires a stored API key (read from the Vault via ``key_ref``). Returns an
    empty list on any failure so the Vault UI can fall back to typing the name.
    """
    import requests

    from app.services.vault import VaultService

    ref = provider.get("key_ref", "")
    if not ref:
        return []
    if ":" in ref:
        namespace, username = ref.split(":", 1)
    else:
        namespace, username = "llm", ref
    key = VaultService.get(namespace, username)
    if not key:
        return []
    base = str(provider.get("base_url", "")).rstrip("/")
    if not base:
        return []
    url = f"{base}/models"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        names = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
        return sorted(str(n) for n in names)
    except Exception:  # noqa: BLE001 - model discovery is best-effort
        return []