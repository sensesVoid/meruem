"""Phase 2: account copier — equity resolution, selected master, account update."""

from app.services.copier import CopierEngine
from app.services.orchestrator import Orchestrator
from app.services.settings_service import SettingsService


def _mk_account(session, account_type="master", broker="MT5", enabled=True, balance=None, login="acc1"):
    from app.models.entities import Account

    acc = Account(
        id=f"a-{login}", broker=broker, account_type=account_type,
        nickname=login, risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref=f"mt5:{login}", login=login, balance=balance, enabled=enabled,
    )
    session.add(acc)
    session.commit()
    session.refresh(acc)
    return acc


def test_equity_uses_account_balance_when_no_live(session):
    acc = _mk_account(session, balance=5000.0)
    engine = CopierEngine(session)
    assert engine._account_equity(acc) == 5000.0


def test_equity_falls_back_to_default(session):
    acc = _mk_account(session, balance=None)
    engine = CopierEngine(session)
    assert engine._account_equity(acc) == SettingsService(session).get("default_account_equity", 10000.0)


def test_equity_ignores_disabled_accounts_for_live(session):
    acc = _mk_account(session, enabled=False, balance=9999.0)
    engine = CopierEngine(session)
    # Disabled accounts never resolve live equity; fall back to stored balance.
    assert engine._account_equity(acc) == 9999.0
    # Disabled accounts are excluded from fan-out entirely.
    from app.services.settings_service import SettingsService
    master = _mk_account(session, login="master", balance=10000.0)
    plans = engine.fan_out(
        master_account=master, master_lot=0.1, master_equity=10000.0,
        sl_distance_pips=50, pip_value=10, entry_price=1.0, sl=0.95,
        tp={}, direction="BUY", scan_id=None, dry_run=True,
    )
    assert len(plans) == 1  # only master, slave is disabled


def test_orchestrator_selects_user_selected_master(session):
    a = _mk_account(session, login="m1")
    _mk_account(session, login="m2", balance=1234.0)
    svc = SettingsService(session)
    svc.set("selected_master_account_id", "a-m2")
    orch = Orchestrator(session)
    master = orch._select_master()
    assert master is not None and master.id == "a-m2"
    assert master.balance == 1234.0


def test_orchestrator_falls_back_to_first_enabled_master(session):
    _mk_account(session, login="m1", balance=100.0)
    _mk_account(session, login="m2", balance=200.0)
    orch = Orchestrator(session)
    master = orch._select_master()
    assert master is not None and master.id == "a-m1"


def test_orchestrator_skips_disabled_master(session):
    _mk_account(session, login="off", enabled=False)
    _mk_account(session, login="on")
    orch = Orchestrator(session)
    master = orch._select_master()
    assert master is not None and master.id == "a-on"
