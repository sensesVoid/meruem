"""Data cleanup cycle: prune old non-essential rows after a retention window.

Live/operational data (accounts, assets, strategies, open trades, settings,
and closed trades are retained unless configured otherwise) is preserved; bulk
historical artifacts (scans, validations, backtest runs) and long-closed trades
are eligible for pruning once they pass the retention age.
"""

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.models.entities import BacktestRun, Scan, Trade, Validation
from app.services.settings_service import SettingsService


class DataCleanupService:
    """Prune old records beyond a configurable retention threshold.

    ``retention_days`` may be provided explicitly or read from the
    ``data_retention_days`` setting (``0`` disables pruning).
    """

    def __init__(self, session: Session, retention_days: int | None = None):
        self.session = session
        self.settings = SettingsService(session)
        if retention_days is None:
            retention_days = int(self.settings.get("data_retention_days", 0) or 0)
        self.retention_days = retention_days

    def cleanup(self, dry_run: bool = False) -> dict:
        if self.retention_days <= 0:
            return {"enabled": False, "retention_days": self.retention_days, "deleted": {}}

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        deleted: dict[str, int] = {}

        # Delete validations whose parent scan is being pruned (cascade).
        old_scans = self.session.exec(select(Scan).where(Scan.timestamp < cutoff)).all()
        scan_ids = [s.id for s in old_scans]
        if scan_ids:
            # Drop any orphaned validations referencing pruned scans.
            self._delete_by_column(Validation, "scan_id", scan_ids)
            for s in old_scans:
                self.session.delete(s)
            deleted["scans"] = len(old_scans)

        # Old closed trades (only those already closed long ago); never open trades.
        old_closed = self.session.exec(
            select(Trade).where(
                Trade.closed_at.is_not(None),
                Trade.closed_at < cutoff,
            )
        ).all()
        for t in old_closed:
            self.session.delete(t)
        if old_closed:
            deleted["closed_trades"] = len(old_closed)

        # Old backtest runs keep the library lean.
        old_backtests = self.session.exec(
            select(BacktestRun).where(BacktestRun.created_at < cutoff)
        ).all()
        for b in old_backtests:
            self.session.delete(b)
        if old_backtests:
            deleted["backtest_runs"] = len(old_backtests)

        if not dry_run:
            self.session.commit()

        return {
            "enabled": True,
            "retention_days": self.retention_days,
            "cutoff": cutoff.isoformat(),
            "deleted": deleted,
        }

    def _delete_by_column(self, model, column_name: str, values: list) -> None:
        if not values:
            return
        col = getattr(model, column_name)
        for obj in self.session.exec(select(model).where(col.in_(values))).all():
            self.session.delete(obj)