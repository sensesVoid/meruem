from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.entities import Account, Asset, Trade
from app.services.orchestrator import Orchestrator
from app.services.risk_manager import RiskManager
from app.services.settings_service import SettingsService


def test_risk_manager_plan(session: Session):
    # Setup accounts
    master = Account(
        id="acct-master-1",
        broker="MT5",
        account_type="master",
        nickname="Master 1",
        risk_scaling_mode="proportional",
        scale_factor=1.0,
        balance=10000.0,
        enabled=True,
        encrypted_ref="mt5:123",
    )
    slave = Account(
        id="acct-slave-1",
        broker="MT5",
        account_type="slave",
        nickname="Slave 1",
        balance=5000.0,
        risk_scaling_mode="proportional",
        scale_factor=1.0,
        enabled=True,
        encrypted_ref="mt5:456",
    )
    session.add(master)
    session.add(slave)
    session.commit()

    rm = RiskManager(session)
    plan = rm.plan()

    assert "generated_at" in plan
    assert "decision" in plan
    assert "exposure" in plan
    assert plan["equity"] == 10000.0
    assert len(plan["accounts"]) == 2
    assert len(plan["lots"]) == 2
    assert isinstance(plan["advisory"], str)


def test_risk_manager_clearance_allowed(session: Session):
    svc = SettingsService(session)
    svc.set("auto_execution_enabled", True)
    svc.set("master_risk_pct", 1.0)
    svc.set("master_lot_sizing_mode", "risk_pct")

    master = Account(
        id="acct-master-clear",
        broker="MT5",
        account_type="master",
        nickname="Master Clear",
        risk_scaling_mode="proportional",
        scale_factor=1.0,
        balance=10000.0,
        enabled=True,
        encrypted_ref="mt5:789",
    )
    session.add(master)
    session.commit()

    rm = RiskManager(session)
    res = rm.clearance(
        account_id=master.id,
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        sl=2640.0,
    )

    assert res["allowed"] is True
    assert res["reason"] == "All risk guardrails passed"
    assert "lots" in res["detail"]
    assert res["detail"]["sl_distance"] == 10.0
    lots = res["detail"]["lots"]
    assert len(lots) == 1
    assert lots[0]["lot"] == 1.0  # (10000 * 0.01) / (10 * 10) = 1.0


def test_risk_manager_clearance_blocked(session: Session):
    svc = SettingsService(session)
    svc.set("auto_execution_enabled", False)

    rm = RiskManager(session)
    res = rm.clearance(symbol="XAUUSD", direction="BUY", entry_price=2650.0, sl=2640.0)

    assert res["allowed"] is False
    assert "disabled" in res["reason"]


def test_risk_manager_stop_scan_reason(session: Session):
    svc = SettingsService(session)
    svc.set("auto_execution_enabled", True)
    svc.set("max_concurrent_auto_trades", 1)

    trade = Trade(
        id="trade-open-auto",
        account_id="acct-1",
        entry_price=2650.0,
        sl=2640.0,
        direction="BUY",
        lot_size=0.1,
        mode="auto",
    )
    session.add(trade)
    session.commit()

    rm = RiskManager(session)
    stop_reason = rm.stop_scan_reason()

    assert "max concurrent auto-trades reached" in stop_reason.lower()


def test_orchestrator_stop_scan_gate(session: Session):
    svc = SettingsService(session)
    svc.set("auto_execution_enabled", True)
    svc.set("max_concurrent_auto_trades", 1)

    trade = Trade(
        id="trade-open-auto-gate",
        account_id="acct-1",
        entry_price=2650.0,
        sl=2640.0,
        direction="BUY",
        lot_size=0.1,
        mode="auto",
    )
    asset = Asset(id="ast-gold", symbol="XAUUSD", asset_class="forex", timeframe_watchlist="15m")
    session.add(trade)
    session.add(asset)
    session.commit()

    orch = Orchestrator(session)
    result = orch.run_scan(asset)

    assert result["risk_blocked"] is True
    assert "stop_reason" in result
    assert "max concurrent auto-trades reached" in result["stop_reason"].lower()


def test_risk_api_endpoints(session: Session):
    client = TestClient(app)

    # GET /api/risk/plan
    resp = client.get("/api/risk/plan")
    assert resp.status_code == 200
    data = resp.json()
    assert "decision" in data
    assert "lots" in data

    # POST /api/risk/clear
    resp = client.post(
        "/api/risk/clear",
        json={"symbol": "XAUUSD", "direction": "BUY", "entry_price": 2650.0, "sl": 2640.0},
    )
    assert resp.status_code == 200
    clear_data = resp.json()
    assert "allowed" in clear_data

    # GET /api/risk/check
    resp = client.get("/api/risk/check")
    assert resp.status_code == 200
    assert "allowed" in resp.json()
