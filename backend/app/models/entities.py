from datetime import datetime, timezone

from sqlalchemy import event
from sqlmodel import Field, SQLModel


class Account(SQLModel, table=True):
    """Master or Slave broker account configuration."""

    __tablename__ = "accounts"

    id: str = Field(primary_key=True)
    broker: str  # MT5 | Binance | cTrader
    account_type: str  # master | slave
    nickname: str
    risk_scaling_mode: str  # proportional | fixed | risk_pct
    scale_factor: float = 1.0
    encrypted_ref: str  # key storage reference in OS keyring
    login: str = ""  # broker login / API user identifier
    server: str = ""  # MT5 broker server (e.g. ICMarkets-Live02)
    base_currency: str = "USD"
    balance: float | None = None  # user-entered funding (pre-live) or cached live
    enabled: bool = True


class Asset(SQLModel, table=True):
    __tablename__ = "assets"

    id: str = Field(primary_key=True)
    symbol: str = Field(unique=True, index=True)
    asset_class: str  # forex | crypto | commodities
    timeframe_watchlist: str = "15m,1h"


class Strategy(SQLModel, table=True):
    __tablename__ = "strategies"

    id: str = Field(primary_key=True)
    name: str
    concept_markdown: str = ""  # general trading concept (ingested as context, preserved as-is)
    body_markdown: str  # entry criteria: concrete, executable entry/exit/risk basis
    asset_class: str = ""
    timeframe: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_id: str | None = None
    status: str = "active"  # active | draft | archived | failed
    params_json: str = ""  # best backtest spec (JSON) applied on approval
    strategy_code: str = ""  # AI-generated executable strategy code (faithful backtest)
    qa_notes: str = ""  # QA/rejection notes persisted for the user (edited on fix)


class Scan(SQLModel, table=True):
    __tablename__ = "scans"

    id: str = Field(primary_key=True)
    asset_id: str | None = Field(default=None, foreign_key="assets.id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scanner_output_json: str
    strategy_refs: str = "[]"  # JSON array


class Validation(SQLModel, table=True):
    __tablename__ = "validations"

    id: str = Field(primary_key=True)
    scan_id: str = Field(foreign_key="scans.id")
    verdict: str  # ACCEPTABLE | NOT_ACCEPTABLE
    rationale: str
    suggested_levels_json: str  # TP1..TP4, SL
    provider_used: str  # gemini | groq | deepseek | ollama


class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id: str = Field(primary_key=True)
    account_id: str = Field(foreign_key="accounts.id")
    scan_id: str | None = Field(default=None, foreign_key="scans.id")
    entry_price: float
    sl: float
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    tp4: float | None = None
    direction: str  # BUY | SELL
    lot_size: float = 0.0
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    pnl: float | None = None
    mode: str = "manual"  # manual | auto
    symbol: str = ""  # broker symbol the order was placed on
    broker: str = ""  # MT5 | Binance
    broker_order_id: str | None = None  # ticket/id returned by the broker
    broker_error: str = ""  # non-empty when live placement was attempted & failed
    exit_price: float | None = None  # actual fill price when the position closed (SL/TP/manual/broker)
    close_reason: str = ""  # why the trade closed: sl_hit | tp_hit | manual | broker | trail
    loss_reason: str = ""  # journal of error: why this losing trade lost (LTR)
    ltr_review: str = ""  # AI review of the loss (root cause + lesson, LTR)
    ltr_reviewed: int = 0  # 1 once the LTR agent has reviewed this losing trade


class LTRRecord(SQLModel, table=True):
    """Losing-trade reference DB (LTR): one snapshot per AI-reviewed loss.

    Backend reference only — agents (Synthesizer, retraining) read this to learn
    from past failures. Stores BOTH the journal of error (why it went wrong) and
    the correction (how it should have been done), the raw inputs, and the outcome
    so the record doubles as structured retraining data when the same loss pattern
    recurs. Written exactly once per losing trade (``trade_id`` unique), never
    re-written, so reviewed losses can not be reverted into the review queue.
    """

    __tablename__ = "ltr_records"

    id: str = Field(primary_key=True)
    trade_id: str = Field(index=True)
    account_id: str = Field(index=True)
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    sl: float = 0.0
    tp1: float | None = None
    exit_price: float | None = None
    pnl: float | None = None
    close_reason: str = ""
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    loss_reason: str = ""  # WHY it went wrong (journal of error)
    correction: str = ""  # HOW it should have been done (retraining target)
    review: str = ""  # root-cause analysis + actionable lesson
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider_used: str = ""


class BacktestRun(SQLModel, table=True):
    __tablename__ = "backtest_runs"

    id: str = Field(primary_key=True)
    strategy_id: str = Field(foreign_key="strategies.id")
    symbol: str
    timeframe: str
    iteration: int = 0
    spec_json: str
    metrics_json: str  # {"train": {...}, "test": {...}}
    rationale: str = ""
    accepted: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Setting(SQLModel, table=True):
    __tablename__ = "settings"

    id: str = Field(primary_key=True)
    key: str = Field(unique=True, index=True)
    value_json: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@event.listens_for(SQLModel.metadata, "before_create")
def _on_create(*_args, **_kwargs):
    pass