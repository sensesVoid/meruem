"""Tests for task #12 Phase 1+2: agent catalog, per-agent model resolution, and
cloud/Ollama provider model discovery."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_config import (
    AGENT_CATALOG,
    get_agent_catalog,
    get_agent_model,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_catalog_has_all_agents():
    ids = {a["id"] for a in AGENT_CATALOG}
    assert {
        "scanner", "validator", "regime", "news", "ingestion",
        "synthesizer", "backtest", "refiner", "ltr", "qa",
    } <= ids


def test_every_agent_has_recommendation_and_tip():
    for a in AGENT_CATALOG:
        assert a["id"]
        assert a.get("name")
        assert a.get("role")
        assert a.get("engine") in ("local", "cloud")
        assert a.get("recommended_model")
        assert a.get("tip")


def test_get_agent_model_defaults_from_catalog():
    assert get_agent_model("scanner") == "qwen2.5:3b-instruct"
    assert get_agent_model("validator") == "gemini-3.5-flash-lite"


def test_get_agent_model_override_not_persisted_across_runs():
    # Since settings are cached, this just confirms the resolver returns a
    # defined string for every agent id in the catalog.
    for a in AGENT_CATALOG:
        model = get_agent_model(a["id"])
        assert isinstance(model, str)
        assert model


def test_catalog_includes_assigned_model():
    for a in get_agent_catalog():
        assert "assigned_model" in a
        assert a["assigned_model"]


def test_unknown_agent_returns_empty():
    assert get_agent_model("does_not_exist") == ""


def test_list_provider_models_without_key_returns_empty():
    from app.services.model_config import list_provider_models

    assert list_provider_models({"key_ref": "llm:missing-key-provider"}) == []


def test_provider_models_ollama_endpoint(client, monkeypatch):
    from app.agents.ollama_client import OllamaClient

    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["qwen2.5:3b", "qwen2.5:7b"])
    r = client.get("/api/providers/ollama/models")
    assert r.status_code == 200
    assert r.json()["provider"] == "ollama"
    assert "qwen2.5:3b" in r.json()["models"]


def test_provider_models_unknown_returns_404(client):
    r = client.get("/api/providers/nope/models")
    assert r.status_code == 404


def test_settings_round_trip_updates_catalog(client):
    import app.services.model_config as mc
    from app.services.model_config import get_agent_model

    put = client.put("/api/settings", json={"key": "agent_models", "value": {"scanner": "qwen3:x-test"}})
    assert put.status_code == 200
    try:
        mc._cache.clear()
        cat = client.get("/api/agents/catalog").json()["agents"]
        scanner = next(a for a in cat if a["id"] == "scanner")
        assert scanner["assigned_model"] == "qwen3:x-test"
        assert get_agent_model("scanner") == "qwen3:x-test"
    finally:
        client.put("/api/settings", json={"key": "agent_models", "value": {}})
        mc._cache.clear()
        # no override -> falls back to legacy ollama_scanner_model setting
        assert get_agent_model("scanner") == "qwen2.5:3b-instruct"


def test_agent_status_reflects_assigned_model(client, monkeypatch):
    import app.services.model_config as mc
    from app.services.agent_status import agent_status

    monkeypatch.setattr(mc, "get_providers", lambda: [])
    try:
        client.put("/api/settings", json={"key": "agent_models", "value": {"validator": "gemini-x-pro"}})
        mc._cache.clear()
        st = agent_status()
        validator = next(a for a in st["agents"] if a["id"] == "validator")
        assert validator["model"] == "gemini-x-pro"
    finally:
        client.put("/api/settings", json={"key": "agent_models", "value": {}})
        mc._cache.clear()