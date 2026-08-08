from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine

_scheduler: BackgroundScheduler | None = None
_scan_interval: int = settings.scan_interval_seconds
_scan_enabled: bool = True


def _run_cleanup() -> None:
    """Daily data cleanup cycle: prune rows past the retention window."""
    from app.services.cleanup import DataCleanupService

    try:
        with Session(engine) as session:
            result = DataCleanupService(session).cleanup()
        print(f"[scheduler] data cleanup: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] cleanup error: {exc}")


def _run_scan_cycle() -> None:
    from app.services.orchestrator import scan_all_assets

    try:
        results = scan_all_assets()
        trades = sum(1 for r in results.values() if (r.get("validation") or {}).get("verdict") == "ACCEPTABLE")
        print(f"[scheduler] scan cycle done: {len(results)} asset(s), {trades} signal(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] scan cycle error: {exc}")


def _run_regime() -> None:
    """Refresh market regime classification every 5 minutes."""
    from app.services.regime import RegimeAgent

    try:
        with Session(engine) as session:
            RegimeAgent().refresh(session)
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] regime refresh error: {exc}")


def _run_news_agent() -> None:
    """Refresh AI news summaries every 15 minutes."""
    from app.services.news_agent import NewsAgent

    try:
        with Session(engine) as session:
            NewsAgent().refresh(session)
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] news agent refresh error: {exc}")


def _run_ltr() -> None:
    """Automated LoseTradeReview: review once ltr_review_threshold losses pending.

    Idempotent: only unreviewed losing trades are consumed and each is snapshotted
    into the ltr_records reference DB; reviewed losses are never re-reviewed.
    """
    from app.agents.ltr import LTRAgent

    try:
        with Session(engine) as session:
            res = LTRAgent(session).auto_review(threshold=settings.ltr_review_threshold)
        if res["triggered"]:
            print(f"[scheduler] LTR: reviewed {res['count']} losing trade(s) "
                  f"({res['pending']} pending)")
        else:
            print(f"[scheduler] LTR: {res['pending']} pending, "
                  f"threshold {res['threshold']} not reached")
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] LTR error: {exc}")


def _run_ml_retrain() -> None:
    """Weekly XGBoost ML scanner retrain.

    Automatically retrains the classifier every 7 days using the latest
    green-gate strategies + accepted BacktestRuns so the model stays current
    as new strategies are added to the library.
    """
    from app.services.ml_scanner import train as ml_train

    try:
        with Session(engine) as session:
            result = ml_train(session)
        if result.get("ok"):
            print(f"[scheduler] ML retrain: {result.get('n_samples')} samples, "
                  f"train_acc={result.get('train_accuracy'):.3f}, "
                  f"test_acc={result.get('test_accuracy'):.3f}")
        else:
            print(f"[scheduler] ML retrain skipped: {result.get('error')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] ML retrain error: {exc}")


def _load_persisted_scan_settings() -> tuple[int, bool]:
    """Read the user's persisted automation prefs (or defaults).

    The Signals tab's automation toggle writes ``auto_scan_enabled`` and
    ``scan_interval_seconds`` to the DB. On restart we must honour those instead
    of silently resetting to enabled/300s from config defaults.
    """
    try:
        from app.services.model_config import _read

        enabled = _read("auto_scan_enabled")
        interval = _read("scan_interval_seconds")
    except Exception:  # noqa: BLE001 - never block startup on a read hiccup
        enabled = interval = None
    enabled = bool(enabled) if enabled is not None else True
    interval = int(interval) if interval is not None else settings.scan_interval_seconds
    return max(10, interval), enabled


def start_scheduler() -> BackgroundScheduler:
    global _scheduler, _scan_interval, _scan_enabled
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scan_interval, _scan_enabled = _load_persisted_scan_settings()
    job = _scheduler.add_job(
        _run_scan_cycle,
        trigger=IntervalTrigger(seconds=_scan_interval),
        id="scan_cycle",
        replace_existing=True,
    )
    if not _scan_enabled:
        job.pause()
    _scheduler.add_job(
        _run_cleanup,
        trigger=IntervalTrigger(days=1),
        id="cleanup_daily",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_regime,
        trigger=IntervalTrigger(minutes=5),
        id="regime_5min",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_news_agent,
        trigger=IntervalTrigger(minutes=15),
        id="news_agent_15m",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_ltr,
        trigger=IntervalTrigger(seconds=settings.ltr_check_interval_seconds),
        id="ltr_auto",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_ml_retrain,
        trigger=IntervalTrigger(weeks=1),
        id="ml_retrain_weekly",
        replace_existing=True,
    )
    _scheduler.start()
    return _scheduler


def set_scan_enabled(enabled: bool) -> None:
    global _scheduler, _scan_enabled
    _scan_enabled = bool(enabled)
    if _scheduler is None or not _scheduler.running:
        return
    job = _scheduler.get_job("scan_cycle")
    if job is None:
        _scheduler.add_job(
            _run_scan_cycle,
            trigger=IntervalTrigger(seconds=_scan_interval),
            id="scan_cycle",
            replace_existing=True,
        )
    elif _scan_enabled:
        job.resume()
    else:
        job.pause()


def set_scan_interval(seconds: int) -> None:
    global _scheduler, _scan_interval
    _scan_interval = max(10, int(seconds))
    if _scheduler is not None and _scheduler.running:
        _scheduler.modify_job(
            "scan_cycle",
            trigger=IntervalTrigger(seconds=_scan_interval),
        )


def scan_state() -> dict:
    return {
        "enabled": _scan_enabled,
        "interval_seconds": _scan_interval,
    }


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None