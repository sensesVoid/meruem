import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _seed(client):
    client.post("/api/accounts", json={
        "id": "dash-master", "broker": "MT5", "account_type": "master",
        "nickname": "Main", "risk_scaling_mode": "proportional",
        "scale_factor": 1.0, "encrypted_ref": "mt5:main", "balance": 10000,
        "enabled": True,
    })
    client.post("/api/accounts", json={
        "id": "dash-slave", "broker": "MT5", "account_type": "slave",
        "nickname": "Slave", "risk_scaling_mode": "fixed",
        "scale_factor": 0.5, "encrypted_ref": "mt5:slave", "balance": 5000,
        "enabled": True,
    })
    client.post("/api/accounts", json={
        "id": "dash-disabled", "broker": "Binance", "account_type": "slave",
        "nickname": "Off", "risk_scaling_mode": "fixed",
        "scale_factor": 0.5, "encrypted_ref": "binance:off", "balance": 9000,
        "enabled": False,
    })
    client.put("/api/settings", json={"key": "selected_master_account_id", "value": "dash-master"})
    # one open + one closed (with pnl) on the master
    client.post("/api/trades", json={
        "account_id": "dash-master", "direction": "BUY", "entry_price": 2650,
        "sl": 2630, "tp1": 2670, "lot_size": 0.1, "mode": "auto",
    })
    t = client.post("/api/trades", json={
        "account_id": "dash-master", "direction": "SELL", "entry_price": 2700,
        "sl": 2720, "tp1": 2680, "lot_size": 0.1, "mode": "manual",
    }).json()
    client.put(f"/api/trades/{t['id']}/close")
    # realized pnl is not written by close() (needs broker mark-to-market),
    # so we set it directly to verify aggregation.
    from sqlalchemy import text

    from app.core.db import engine

    with engine.begin() as conn:
        conn.execute(text("UPDATE trades SET pnl = 85.5 WHERE id = :id"), {"id": t["id"]})


def test_dashboard_returns_master_and_slaves(client):
    _seed(client)
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert data["master_id"] == "dash-master"
    assert data["master"]["account_id"] == "dash-master"
    assert data["master"]["is_master"] is True
    assert data["master"]["funded"] == 10000.0
    assert data["master"]["running_capital"] == 10000.0
    assert data["master"]["gain"] == 0.0
    assert data["master"]["gain_pct"] == 0.0
    # disabled account excluded
    ids = {s["account_id"] for s in data["slaves"]}
    assert "dash-slave" in ids
    assert "dash-disabled" not in ids


def test_dashboard_gain_calculation(client):
    _seed(client)
    from sqlalchemy import text

    from app.core.db import engine

    with engine.begin() as conn:
        conn.execute(text("UPDATE accounts SET balance = 10000 WHERE id = 'dash-master'"))
    data = client.get("/api/dashboard").json()
    # master has funded 10000, running falls back to balance -> gain 0
    master = data["master"]
    assert master["funded"] == 10000.0
    assert master["gain_pct"] == 0.0


def test_dashboard_falls_back_to_first_enabled_master(client):
    client.post("/api/accounts", json={
        "id": "fallback-master", "broker": "MT5", "account_type": "master",
        "nickname": "Fallback", "risk_scaling_mode": "proportional",
        "scale_factor": 1.0, "encrypted_ref": "mt5:fallback", "balance": 20000,
        "enabled": True,
    })
    data = client.get("/api/dashboard").json()
    assert data["master"]["account_id"] == "fallback-master"
    assert data["master"]["funded"] == 20000.0


def test_dashboard_returns_performance_curve_and_stats(client):
    from sqlalchemy import text

    from app.core.db import engine

    _seed(client)
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE trades SET pnl = 85.5, closed_at = '2026-08-07T10:00:00+00:00' "
            "WHERE id = (SELECT id FROM trades WHERE account_id = 'dash-master' AND pnl IS NOT NULL LIMIT 1)"
        ))
    data = client.get("/api/dashboard").json()
    assert data["master_id"] == "dash-master"
    perf = data["performance"]
    assert perf is not None
    stats = perf["stats"]
    assert stats["baseline"] == 10000.0
    # one closed winner -> curve starts at baseline, ends baseline+pnl
    assert perf["curve"][0]["equity"] == 10000.0
    assert perf["curve"][-1]["equity"] == pytest.approx(10085.5, abs=0.01)
    assert stats["total_closed"] == 1
    assert stats["wins"] == 1
    assert stats["win_rate"] == 100.0
    assert stats["best"] == pytest.approx(85.5, abs=0.01)

    # create a second (losing) closed trade and re-check stats aggregation.
    t2 = client.post("/api/trades", json={
        "account_id": "dash-master", "direction": "BUY", "entry_price": 1.0,
        "sl": 0.5, "tp1": None, "lot_size": 0.1, "mode": "manual", "symbol": "EURUSD",
    }).json()
    client.put(f"/api/trades/{t2['id']}/close")
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE trades SET pnl = -50, closed_at = '2026-08-07T12:00:00+00:00' WHERE id = :id"
        ), {"id": t2["id"]})
    data = client.get("/api/dashboard").json()
    stats = data["performance"]["stats"]
    assert stats["total_closed"] == 2
    assert stats["wins"] == 1 and stats["losses"] == 1
    assert stats["win_rate"] == 50.0
    assert stats["profit_factor"] == pytest.approx(1.71, abs=0.02)
    assert stats["avg_loss"] == pytest.approx(50.0, abs=0.01)
    assert stats["max_drawdown_pct"] > 0.0
