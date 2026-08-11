"""Phase: broker connection revamp — unified connect, single master, status reason."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _mk(session, account_type="master", broker="MT5", login="acc1"):
    from app.models.entities import Account

    acc = Account(
        id=f"a-{login}", broker=broker, account_type=account_type,
        nickname=login, risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref=f"{broker.lower()}:{login}", login=login, server="Srv",
        enabled=True,
    )
    session.add(acc)
    session.commit()
    session.refresh(acc)
    return acc


def test_connect_requires_valid_broker(session, client):
    r = client.post("/api/brokers/connect", json={
        "broker": "Nope", "account_type": "slave", "login": "1",
    })
    assert r.status_code == 400


def test_connect_persists_account_and_single_master(session, client):
    with patch("app.api.routes._test_broker", return_value=(True, "ok")), \
         patch("app.services.copier.CopierEngine._account_status", return_value=(True, "connected")):
        r1 = client.post("/api/brokers/connect", json={
            "broker": "MT5", "account_type": "master", "login": "111",
            "server": "Srv", "password": "pw",
        })
        r2 = client.post("/api/brokers/connect", json={
            "broker": "Binance", "account_type": "master", "api_key": "k2",
            "secret": "s2", "login": "acc2",
        })
    assert r1.status_code == 200
    assert r1.json()["ok"] is True
    assert r2.json()["ok"] is True

    # only one master remains among all accounts
    from app.models.entities import Account
    from sqlmodel import select

    masters = session.exec(select(Account).where(Account.account_type == "master")).all()
    assert len(masters) == 1
    assert masters[0].id == r2.json()["account"]["id"]

    from app.services.settings_service import SettingsService
    assert SettingsService(session).get("selected_master_account_id") == masters[0].id


def test_connect_failed_does_not_create_account(session, client):
    with patch("app.api.routes._test_broker", return_value=(False, "bad creds")):
        r = client.post("/api/brokers/connect", json={
            "broker": "MT5", "account_type": "master", "login": "777",
            "server": "Srv", "password": "nope",
        })
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["account"] is None


def test_status_reports_reason_for_unconfigured(session, client):
    from app.services.copier import CopierEngine

    acc = _mk(session, login="s1")
    connected, reason = CopierEngine(session)._account_status(acc)
    assert connected is False
    assert "no MT5 credentials" in reason


def test_brokers_status_returns_accounts_with_reason(session, client):
    _mk(session, broker="MT5", login="s2")
    r = client.get("/api/brokers/status")
    assert r.status_code == 200
    data = r.json()
    assert "accounts" in data
    assert any(a["account_id"] == "a-s2" for a in data["accounts"])
    entry = next(a for a in data["accounts"] if a["account_id"] == "a-s2")
    assert entry["connected"] is False
    assert entry["reason"]


def test_agents_status_reports_active_agents(client):
    with patch("app.services.agent_status._ollama_status", return_value={
        "url": "http://127.0.0.1:11434", "reachable": True, "models": ["qwen2.5:3b"],
        "latency_ms": 5, "error": "", "scanner_model": "qwen2.5:3b",
        "validator_model": "qwen2.5:7b",
    }), patch("app.services.agent_status._providers_status", return_value=[]), \
            patch("app.services.agent_status.get_agent_model", side_effect=lambda a: {
                "scanner": "qwen2.5:3b", "validator": "qwen2.5:7b",
                "qa": "qwen2.5:7b", "synthesizer": "qwen2.5:7b",
                "regime": "indicator-based",
                "news": "qwen2.5:7b",
            }.get(a, "")):
        r = client.get("/api/agents/status")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    ids = {a["id"] for a in data["agents"]}
    assert {"scanner", "validator", "qa", "risk_manager"} <= ids
    scanner = next(a for a in data["agents"] if a["id"] == "scanner")
    assert scanner["ready"] is True
    assert scanner["model"] == "qwen2.5:3b"
    assert data["ollama"]["reachable"] is True