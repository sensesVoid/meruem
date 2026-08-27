"""Tests for launch-time Ollama auto-detection (ollama_discovery)."""

import pytest
from fastapi.testclient import TestClient

from app.main import _run_ollama_detection, app
from app.services import ollama_discovery
from app.services.model_config import get_agent_model
from app.services.settings_service import SettingsService


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_autoconfigure_persists_reachable_endpoint_and_repoints_agents(
    session, monkeypatch
):
    """Reachable Ollama -> ollama_url persisted, local agents re-pointed.

    Cloud agents must be left untouched (non-destructive).
    """
    url = "http://127.0.0.1:9999"
    monkeypatch.setattr(ollama_discovery, "detect_url", lambda force=False: url)
    monkeypatch.setattr(ollama_discovery, "list_installed", lambda _url=None: ["qwen3:4b"])

    report = ollama_discovery.autoconfigure()

    assert report["url"] == url
    assert report["models"] == ["qwen3:4b"]
    assert "qwen3:4b" in {f["to"] for f in report["fixed"].values()}

    svc = SettingsService(session)
    assert svc.get("ollama_url") == url
    # scanner / regime resolve to the detected model via the agent_models row.
    assert get_agent_model("scanner") == "qwen3:4b"
    assert get_agent_model("regime") == "qwen3:4b"
    assert get_agent_model("synthesizer") == "gemini-3.5-flash-lite"  # untouched


def test_autoconfigure_unreachable_is_noop(session, monkeypatch):
    """No reachable endpoint -> error report, no settings written."""
    monkeypatch.setattr(ollama_discovery, "detect_url", lambda force=False: "")

    report = ollama_discovery.autoconfigure()

    assert report["error"]
    assert SettingsService(session).get("ollama_url") is None
    # the launch reconciler must keep the default resolution intact
    assert get_agent_model("scanner") == "qwen3:4b"


def test_launch_reconciler_hits_autoconfigure_once_on_success(monkeypatch):
    calls = {"n": 0}

    def fake_autoconfigure():
        calls["n"] += 1
        return {"url": "http://127.0.0.1:11434", "models": ["qwen3:4b"],
                "fixed": {}, "error": ""}

    monkeypatch.setattr(ollama_discovery, "autoconfigure", fake_autoconfigure)
    _run_ollama_detection()

    assert calls["n"] == 1


def test_launch_reconciler_retries_then_gives_up(monkeypatch):
    calls = {"n": 0}
    import app.main as main_mod

    def fake_autoconfigure():
        calls["n"] += 1
        return {"url": "", "models": [], "fixed": {}, "error": "no reachable Ollama"}

    monkeypatch.setattr(ollama_discovery, "autoconfigure", fake_autoconfigure)
    monkeypatch.setattr(main_mod.time, "sleep", lambda _s: None)

    _run_ollama_detection()

    assert calls["n"] == 3  # bounded retries, then give up silently


def test_agents_status_includes_discovery_report(client, monkeypatch):
    from app.services import agent_status

    def fake_autoconfigure():
        return {"url": "http://127.0.0.1:11434", "models": ["qwen3:4b"],
                "fixed": {}, "error": ""}

    monkeypatch.setattr(ollama_discovery, "latest_report", lambda: fake_autoconfigure())
    monkeypatch.setattr(agent_status, "_ollama_status", lambda: {
        "url": "http://127.0.0.1:11434", "reachable": True, "models": ["qwen3:4b"],
        "latency_ms": 5, "error": "", "scanner_model": "qwen3:4b",
        "validator_model": "qwen3:4b",
    })
    monkeypatch.setattr(agent_status, "_providers_status", list)

    r = client.get("/api/agents/status")

    assert r.status_code == 200
    data = r.json()
    assert "ollama_discovery" in data
    assert data["ollama_discovery"]["url"] == "http://127.0.0.1:11434"
    assert "agents" in data