"""Benchmark: dashboard + signals — BEFORE (N+1) vs AFTER (grouped) the refactor.

Runs the ACTUAL current route functions (``dashboard`` / ``list_signals``)
against a faithful reconstruction of the pre-refactor query patterns, on the
same seeded SQLite database, measuring wall time AND real SQL statement count
(via a ``before_cursor_execute`` event hook).

Reconstruction notes (from the refactor history):
  * BEFORE dashboard: per-account ``select(Trade).where(account_id==..)`` inside
    ``_account_performance`` (N+1 over accounts), then a SECOND query for the
    master's trades to build the performance curve.
  * BEFORE signals: per-asset ``order_by(timestamp.desc()).first()`` for the
    latest scan (N+1 over assets), then per-scan validation + executed queries
    inside ``_signal_from_scan`` (2N more). The current ``_signal_from_scan``
    keeps that per-scan code path when called with 2 args, so we reuse it.

Run:  cd backend && python scripts/bench_refactor.py
"""
import os
import sys
import tempfile
import time

# Make the backend package importable regardless of the working directory.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

_tmp = tempfile.mkdtemp(prefix="meruem_bench_")
os.environ["MERUEM_DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'bench.db')}"

from datetime import datetime, timedelta, timezone  # noqa: E402

from sqlalchemy import event as sa_event  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from app.api.routes import _performance_curve, _signal_from_scan, dashboard, list_signals  # noqa: E402
from app.core.db import engine, init_db  # noqa: E402
from app.models.entities import Account, Asset, Scan, Trade, Validation  # noqa: E402
from app.services.copier import CopierEngine  # noqa: E402
from app.services.settings_service import SettingsService  # noqa: E402

init_db()

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Query counter
# --------------------------------------------------------------------------
class QueryCounter:
    def __init__(self) -> None:
        self.count = 0

    def _before(self, _conn, _cursor, _stmt, _params, _ctx, _many) -> None:  # type: ignore[no-untyped-def]
        self.count += 1


def timed(fn):
    """Return (wall_seconds, sql_statement_count) for one invocation of fn."""
    counter = QueryCounter()
    sa_event.listen(engine, "before_cursor_execute", counter._before)
    try:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
    finally:
        sa_event.remove(engine, "before_cursor_execute", counter._before)
    return dt, counter.count


def best_of(fn, runs: int = 3) -> tuple[float, int]:
    """Best wall time over `runs` + the query count of that run."""
    results = [timed(fn) for _ in range(runs)]
    return min(results, key=lambda r: r[0])


# --------------------------------------------------------------------------
# Seed data
# --------------------------------------------------------------------------
def seed(accounts_n: int, trades_per: int, assets_n: int, scans_per: int) -> None:
    with Session(engine) as s:
        for t in (Account, Asset, Scan, Validation, Trade):
            s.exec(select(t)).all()  # touch nothing; rows deleted below
        from sqlalchemy import text

        for table in ("trades", "validations", "scans", "assets", "accounts"):
            s.exec(text(f"DELETE FROM {table}"))
        s.commit()

        master = Account(
            id="acct-master", broker="MT5", account_type="master",
            nickname="Bench Master", risk_scaling_mode="proportional",
            encrypted_ref="mt5:bench", balance=10_000.0, enabled=True,
        )
        accounts = [master]
        for i in range(accounts_n - 1):
            accounts.append(Account(
                id=f"acct-{i}", broker="MT5", account_type="slave",
                nickname=f"Slave {i}", risk_scaling_mode="fixed",
                encrypted_ref=f"mt5:bench{i}", balance=5_000.0, enabled=True,
            ))
        s.add_all(accounts)

        assets: list[Asset] = []
        for i in range(assets_n):
            assets.append(Asset(
                id=f"asset-{i}", symbol=f"SYM{i}", asset_class="forex",
            ))
        s.add_all(assets)

        scans: list[Scan] = []
        validations: list[Validation] = []
        now = datetime.now(timezone.utc)
        for i, a in enumerate(assets):
            for j in range(scans_per):
                scans.append(Scan(
                    id=f"scan-{a.id}-{j}",
                    asset_id=a.id,
                    timestamp=now - timedelta(minutes=(scans_per - j)),
                    scanner_output_json='{"direction": "BUY", "entry_zone": [1.1, 1.2]}',
                    strategy_refs="[]",
                ))
                # validate a third of the scans (per-scan lookups still fire for all)
                if j % 3 == 0:
                    validations.append(Validation(
                        id=f"val-{a.id}-{j}", scan_id=f"scan-{a.id}-{j}",
                        verdict="ACCEPTABLE", rationale="bench",
                        suggested_levels_json='{"tp1": 1.3}', provider_used="ollama",
                    ))
        s.add_all(scans)
        s.add_all(validations)

        trades: list[Trade] = []
        k = 0
        for i, acc in enumerate(accounts):
            for j in range(trades_per):
                opened = T0 + timedelta(minutes=k)
                closed = opened + timedelta(minutes=30) if j % 2 else None
                trades.append(Trade(
                    id=f"trade-{k}", account_id=acc.id,
                    scan_id=f"scan-asset-{j % assets_n}-0" if j % 7 == 0 else None,
                    entry_price=1.1, sl=1.09, direction="BUY", lot_size=0.1,
                    opened_at=opened, closed_at=closed,
                    pnl=12.5 if closed else None, mode="auto",
                    symbol=f"SYM{j % assets_n}",
                ))
                k += 1
        s.add_all(trades)
        s.commit()
        s.exec(text("UPDATE scans SET timestamp = datetime('now', '-1 minute')"))


# --------------------------------------------------------------------------
# BEFORE reconstructions (faithful to the pre-refactor query patterns)
# --------------------------------------------------------------------------
def before_dashboard(session: Session) -> dict:
    """Pre-refactor dashboard: per-account trade query (N+1) + master re-query."""
    copier = CopierEngine(session)
    settings = SettingsService(session)
    selected = settings.get("selected_master_account_id", None)
    accounts = session.exec(select(Account)).all()

    master_id = None
    if selected:
        master = next((a for a in accounts if a.id == selected), None)
        if master is not None:
            master_id = master.id
    if master_id is None:
        first = next((a for a in accounts if a.account_type == "master" and a.enabled), None)
        if first:
            master_id = first.id

    rows = []
    for a in accounts:
        if not a.enabled:
            continue
        running = copier._account_equity(a)
        funded = float(a.balance) if a.balance is not None else None
        connected = copier._account_connected(a)
        reason = copier._account_status(a)[1]
        # OLD N+1: one query per account
        trades = session.exec(select(Trade).where(Trade.account_id == a.id)).all()
        closed = [t for t in trades if t.closed_at is not None and t.pnl is not None]
        rows.append({
            "account_id": a.id, "running_capital": running, "funded": funded,
            "connected": connected, "reason": reason,
            "open_trades": sum(1 for t in trades if t.closed_at is None),
            "closed_trades": len(closed),
            "realized_pnl": round(sum(t.pnl for t in closed), 2) if closed else 0.0,
            "is_master": a.id == master_id,
        })

    master = next((r for r in rows if r["is_master"]), None)
    performance = None
    if master is not None and master_id is not None:
        baseline = master["funded"] if master["funded"] is not None else 0.0
        # OLD: separate re-query for the master's trades (curve)
        trades = session.exec(select(Trade).where(Trade.account_id == master_id)).all()
        performance = _performance_curve(trades, baseline)
    return {"master": master, "slaves": rows, "performance": performance}


def before_signals(session: Session) -> dict:
    """Pre-refactor signals: per-asset latest-scan query + per-scan lookups."""
    from app.services.scheduler import scan_state

    assets = session.exec(select(Asset)).all()
    out = []
    for a in assets:
        # OLD N+1: latest scan per asset
        latest = session.exec(
            select(Scan).where(Scan.asset_id == a.id).order_by(Scan.timestamp.desc())
        ).first()
        if latest is None:
            continue
        # OLD per-scan validation + executed queries (2-arg call = old path)
        out.append(_signal_from_scan(session, latest))
    out.sort(key=lambda s: s["timestamp"], reverse=True)
    return {
        "signals": out,
        "automation": scan_state(),
        "auto_execution_enabled": SettingsService(session).get("auto_execution_enabled", False),
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
SCALES = [
    ("small  (3 acct x 300 tr | 8 assets x 40 scans)", 3, 300, 8, 40),
    ("medium (8 acct x 1.5k tr | 20 assets x 150 scans)", 8, 1500, 20, 150),
    ("large  (20 acct x 4k tr | 50 assets x 400 scans)", 20, 4000, 50, 400),
]


def main() -> None:
    print(f"{'dataset':<48} {'variant':<10} {'wall (ms)':>10} {'SQL stmts':>10} {'speedup':>9}")
    print("-" * 92)
    for label, acct_n, tr_per, asset_n, scan_per in SCALES:
        seed(acct_n, tr_per, asset_n, scan_per)
        with Session(engine) as s:
            db_ms, db_q = best_of(lambda: before_dashboard(s))
        with Session(engine) as s:
            da_ms, da_q = best_of(lambda: dashboard(s))
        with Session(engine) as s:
            sb_ms, sb_q = best_of(lambda: before_signals(s))
        with Session(engine) as s:
            sa_ms, sa_q = best_of(lambda: list_signals(s))

        print(f"\n{label}")
        print(f"{'':<48} {'BEFORE':<10} {db_ms * 1000:>10.2f} {db_q:>10} {'':>9}")
        print(f"{'':<48} {'AFTER':<10} {da_ms * 1000:>10.2f} {da_q:>10} {db_ms / da_ms:>8.1f}x")
        print(f"{'':<48} {'BEFORE':<10} {sb_ms * 1000:>10.2f} {sb_q:>10} {'':>9}")
        print(f"{'':<48} {'AFTER':<10} {sa_ms * 1000:>10.2f} {sa_q:>10} {sb_ms / sa_ms:>8.1f}x")

    print("\nbest-of-3 warm runs; SQL count via before_cursor_execute hook.")


if __name__ == "__main__":
    main()
