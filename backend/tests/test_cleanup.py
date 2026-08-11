from datetime import datetime, timedelta, timezone

from app.models.entities import Scan, Trade, Validation
from app.services.cleanup import DataCleanupService


def _mk_scan(session, id_: str, days_old: int):
    s = Scan(id=id_, asset_id=None, timestamp=datetime.now(timezone.utc) - timedelta(days=days_old),
             scanner_output_json="{}", strategy_refs="[]")
    session.add(s)
    return s


def _mk_account(session, id_: str):
    from app.models.entities import Account

    a = Account(id=id_, broker="MT5", account_type="slave", nickname=id_,
                risk_scaling_mode="fixed", scale_factor=0.5, encrypted_ref=f"mt5:{id_}",
                balance=5000, enabled=True)
    session.add(a)
    session.commit()
    return a


def test_cleanup_prunes_old_scans_and_keeps_recent(session):
    _mk_scan(session, "old-scan", days_old=200)
    _mk_scan(session, "new-scan", days_old=5)
    session.commit()
    result = DataCleanupService(session, retention_days=90).cleanup(dry_run=True)
    assert result["deleted"]["scans"] == 1
    session.expunge_all()
    assert session.get(Scan, "new-scan") is not None


def test_cleanup_prunes_old_closed_trades_keeps_open(session):
    acc = _mk_account(session, "acc-closed")
    old_closed = Trade(id="old", account_id=acc.id, entry_price=1, sl=0.5, direction="BUY",
                       lot_size=0.1, closed_at=datetime.now(timezone.utc) - timedelta(days=200), pnl=5.0)
    open_trade = Trade(id="open", account_id=acc.id, entry_price=1, sl=0.5, direction="BUY",
                       lot_size=0.1, closed_at=None, pnl=None)
    session.add(old_closed)
    session.add(open_trade)
    session.commit()
    result = DataCleanupService(session, retention_days=90).cleanup()
    assert result["deleted"]["closed_trades"] == 1
    assert session.get(Trade, "open") is not None


def test_cleanup_retention_zero_disables(session):
    _mk_scan(session, "s", days_old=200)
    session.commit()
    result = DataCleanupService(session, retention_days=0).cleanup()
    assert result["enabled"] is False
    assert session.get(Scan, "s") is not None


def test_cleanup_cascades_validations_with_scans(session):
    _mk_scan(session, "old-scan-v", days_old=200)
    session.add(Validation(id="v1", scan_id="old-scan-v", verdict="ACCEPTABLE",
                           rationale="x", suggested_levels_json="{}", provider_used="ollama"))
    session.commit()
    result = DataCleanupService(session, retention_days=90).cleanup()
    assert result["deleted"]["scans"] == 1
    session.expunge_all()
    assert session.get(Validation, "v1") is None


def test_equity_cache_deduplicates_live_calls():
    from app.services.copier import _EquityCache

    cache = _EquityCache(ttl=100.0)
    assert cache.get("mt5:x") is None
    cache.put("mt5:x", 1234.5)
    assert cache.get("mt5:x") == 1234.5