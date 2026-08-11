from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

_db_path = Path(settings.database_url.replace("sqlite:///", ""))
_db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    from app.models import entities  # noqa: F401  (register tables)

    SQLModel.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """Lightweight column migrations for tables created before schema changes.

    ``create_all`` never alters existing tables, so new columns added to
    models are back-filled here with ALTER TABLE when missing.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if insp.has_table("accounts"):
        existing = {c["name"] for c in insp.get_columns("accounts")}
        additions: dict[str, str] = {
            "login": "VARCHAR DEFAULT ''",
            "server": "VARCHAR DEFAULT ''",
            "base_currency": "VARCHAR DEFAULT 'USD'",
            "balance": "REAL",
            "enabled": "BOOLEAN DEFAULT 1",
        }
        with engine.begin() as conn:
            for name, ddl in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE accounts ADD COLUMN {name} {ddl}"))

    if insp.has_table("trades"):
        existing = {c["name"] for c in insp.get_columns("trades")}
        additions: dict[str, str] = {
            "symbol": "VARCHAR DEFAULT ''",
            "broker": "VARCHAR DEFAULT ''",
            "broker_order_id": "VARCHAR",
            "broker_error": "VARCHAR DEFAULT ''",
            "exit_price": "REAL",
            "close_reason": "VARCHAR DEFAULT ''",
            "loss_reason": "VARCHAR DEFAULT ''",
            "ltr_review": "VARCHAR DEFAULT ''",
            "ltr_reviewed": "INTEGER DEFAULT 0",
        }
        with engine.begin() as conn:
            for name, ddl in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE trades ADD COLUMN {name} {ddl}"))

    if insp.has_table("strategies"):
        existing = {c["name"] for c in insp.get_columns("strategies")}
        with engine.begin() as conn:
            if "params_json" not in existing:
                conn.execute(text("ALTER TABLE strategies ADD COLUMN params_json VARCHAR DEFAULT ''"))
            if "qa_notes" not in existing:
                conn.execute(text("ALTER TABLE strategies ADD COLUMN qa_notes VARCHAR DEFAULT ''"))

    _drop_obsolete(insp)


def _drop_obsolete(insp) -> None:
    """Retire tables/columns removed by the slimming refactor (backtest runs,
    GP evolution, strategy provenance/code). Idempotent so a fresh DB is fine."""
    from sqlalchemy import text

    with engine.begin() as conn:
        for table in ("backtest_runs", "evolution_runs", "evolution_genome_results"):
            if insp.has_table(table):
                conn.execute(text(f"DROP TABLE {table}"))

    if insp.has_table("strategies"):
        existing = {c["name"] for c in insp.get_columns("strategies")}
        with engine.begin() as conn:
            for col in ("asset_class", "timeframe", "strategy_code", "ir_blueprint_json"):
                if col in existing:
                    conn.execute(text(f"ALTER TABLE strategies DROP COLUMN {col}"))


def get_session():
    with Session(engine) as session:
        yield session
