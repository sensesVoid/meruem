"""Task [adhoc]: scan covers the whole watchlist, and scheduler honours the
user's persisted auto-scan prefs on startup."""

from unittest.mock import patch

import pytest

from app.models.entities import Asset


@pytest.fixture
def assets(session):
    rows = [
        Asset(id="a-a", symbol="AAA", asset_class="forex"),
        Asset(id="a-b", symbol="BBB", asset_class="crypto"),
        Asset(id="a-c", symbol="CCC", asset_class="forex"),
    ]
    for r in rows:
        session.add(r)
    session.commit()
    for r in rows:
        session.refresh(r)
    return rows


def test_scan_all_assets_iterates_every_watchlist_symbol(session, assets):
    from app.services.orchestrator import scan_all_assets

    scanned = []

    class _Orch:
        def __init__(self, session):
            pass

        def run_scan(self, asset):
            scanned.append(asset.symbol)
            return {"symbol": asset.symbol, "candidate": {}}

    with patch("app.services.orchestrator.Orchestrator", _Orch):
        results = scan_all_assets()

    assert sorted(scanned) == ["AAA", "BBB", "CCC"]
    assert set(results.keys()) == {"AAA", "BBB", "CCC"}


def test_scan_all_assets_continues_after_one_failure(session, assets):
    from app.services.orchestrator import scan_all_assets

    def _run_scan(self, asset):
        if asset.symbol == "BBB":
            raise RuntimeError("boom")
        return {"symbol": asset.symbol, "candidate": {}}

    class _Orch:
        def __init__(self, session):
            pass

        run_scan = _run_scan

    with patch("app.services.orchestrator.Orchestrator", _Orch):
        results = scan_all_assets()

    # Still scanned all three; one carries its error, others are intact.
    assert set(results.keys()) == {"AAA", "BBB", "CCC"}
    assert "error" in results["BBB"]
    assert "candidate" in results["AAA"]


def test_start_scheduler_honours_persisted_disabled(session):
    import app.services.model_config as mc
    from sqlmodel import Session

    from app.core.db import engine
    from app.services import scheduler as sch
    from app.services.settings_service import SettingsService

    with Session(engine) as s:
        SettingsService(s).set("auto_scan_enabled", False)
        SettingsService(s).set("scan_interval_seconds", 45)
    mc._cache.clear()  # invalidate the 15s model_config TTL before reading

    sch.stop_scheduler()
    try:
        sched = sch.start_scheduler()
        job = sched.get_job("scan_cycle")
        assert sch._scan_enabled is False
        assert sch._scan_interval == 45
        assert job.next_run_time is None  # paused because disabled
    finally:
        sch.stop_scheduler()
        with Session(engine) as s:
            SettingsService(s).set("auto_scan_enabled", True)
            SettingsService(s).set("scan_interval_seconds", 300)
        mc._cache.clear()