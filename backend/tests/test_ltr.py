"""LoseTradeReview (LTR) agent tests (#8.4).

Covers: selecting only unreviewed losing trades, writing the journal + review +
correction, snapshotting into the ltr_records reference DB, skipping previously
reviewed losses, offline heuristic fallback, the automated threshold trigger, and
the SynthesizerAgent learning-context feed.
"""

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.agents.synthesizer import SynthesizerAgent
from app.models.entities import LTRRecord, Trade


def _trade(session: Session, pnl: float, reviewed: int = 0, symbol: str = "EURUSD",
           closed: bool = True, direction: str = "BUY") -> Trade:
    t = Trade(
        id=f"t-{abs(hash((pnl, reviewed, symbol)))}",
        account_id="acct-trade",
        entry_price=1.1000,
        sl=1.0900,
        tp1=1.1200,
        direction=direction,
        lot_size=0.1,
        pnl=pnl,
        symbol=symbol,
        ltr_reviewed=reviewed,
        closed_at=datetime.now(timezone.utc) - timedelta(days=abs(pnl)),
    )
    session.add(t)
    if closed:
        t.closed_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(t)
    return t


def test_review_only_unreviewed_losing_trades(session, monkeypatch):
    from app.agents.ltr import LTRAgent
    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(
        ValidatorChain, "generate",
        lambda self, prompt: (
            ('{"loss_reason": "entered against strong trend", '
             '"correction": "only enter with the trend after a confirmed retest.", '
             '"review": "wait for trend confirmation before entry."}'),
            "test",
        ),
    )
    _trade(session, pnl=-50, reviewed=0, symbol="EURUSD")
    _trade(session, pnl=-30, reviewed=1, symbol="XAUUSD")  # already reviewed
    _trade(session, pnl=40, reviewed=0, symbol="BTCUSDT")  # winning, skip

    out = LTRAgent(session).review(limit=10)
    assert out["count"] == 1
    assert out["reviewed"][0]["symbol"] == "EURUSD"
    assert "trend" in out["reviewed"][0]["loss_reason"]
    assert "retest" in out["reviewed"][0]["correction"]
    assert out["lessons"]
    assert "LTR" in out["lessons"]

    trades = session.exec(select(Trade)).all()
    by_symbol = {t.symbol: t for t in trades}
    assert by_symbol["EURUSD"].ltr_reviewed == 1
    assert by_symbol["XAUUSD"].ltr_reviewed == 1  # untouched
    assert by_symbol["BTCUSDT"].ltr_reviewed == 0  # untouched (winning)

    records = session.exec(select(LTRRecord)).all()
    assert len(records) == 1
    assert records[0].trade_id == by_symbol["EURUSD"].id
    assert records[0].correction == "only enter with the trend after a confirmed retest."
    assert records[0].loss_reason == "entered against strong trend"
    assert records[0].symbol == "EURUSD"
    assert records[0].pnl == -50


def test_review_heuristic_fallback_when_llm_unavailable(session, monkeypatch):
    from app.agents.ltr import LTRAgent
    from app.agents.validator import ValidatorChain

    def boom(self, prompt):
        raise RuntimeError("offline")

    monkeypatch.setattr(ValidatorChain, "generate", boom)
    _trade(session, pnl=-25.0, reviewed=0, symbol="US30")

    result = LTRAgent(session).review(limit=5)
    assert result["count"] == 1
    assert result["reviewed"][0]["loss_reason"]
    assert "No LLM" in result["reviewed"][0]["review"]


def test_synthesizer_lessons_include_reviewed_loss(session, monkeypatch):
    from app.agents.ltr import LTRAgent
    from app.agents.validator import ValidatorChain

    monkeypatch.setattr(
        ValidatorChain, "generate",
        lambda self, prompt: (
            ('{"loss_reason": "chased the move", '
             '"correction": "wait for a pullback to a discounted price instead.", '
             '"review": "do not enter above the 20-bar high without pullback."}'),
            "test",
        ),
    )
    _trade(session, pnl=-12.0, reviewed=0, symbol="GBPUSD")
    LTRAgent(session).review(limit=5)

    lessons = SynthesizerAgent(session)._ltr_lessons()
    assert "chased the move" in lessons
    assert "pullback" in lessons


def test_no_review_when_no_losing_trades(session):
    from app.agents.ltr import LTRAgent

    _trade(session, pnl=15.0, reviewed=0, symbol="EURUSD")
    result = LTRAgent(session).review(limit=10)
    assert result["count"] == 0
    assert result["lessons"] == ""


def test_auto_review_triggers_at_threshold_and_reviews_once(session, monkeypatch):
    """LTR automation (#8.4): review runs only once the pending-loss threshold is
    reached, consumes the newest batch, and never re-reviews a reviewed loss."""
    from app.agents.ltr import LTRAgent
    from app.agents.validator import ValidatorChain

    def fake(self, prompt):
        return (
            ('{"loss_reason": "stopped out", '
             '"correction": "widen the stop below structure.", '
             '"review": "the stop was inside the noise."}'),
            "test",
        )

    monkeypatch.setattr(ValidatorChain, "generate", fake)

    # 9 losses -> below threshold of 10, must NOT trigger.
    for i in range(9):
        _trade(session, pnl=-(i + 1), reviewed=0, symbol=f"X{i}")
    agent = LTRAgent(session)
    res = agent.auto_review(threshold=10, limit=10)
    assert res["triggered"] is False
    assert res["pending"] == 9
    assert res["count"] == 0
    assert not session.exec(select(LTRRecord)).all()

    # 10th loss arrives -> auto-trigger consumes the newest 10 and marks them done.
    _trade(session, pnl=-100, reviewed=0, symbol="EURUSD")
    res = agent.auto_review(threshold=10, limit=10)
    assert res["triggered"] is True
    assert res["count"] == 10
    assert agent.pending_count() == 0

    records = session.exec(select(LTRRecord)).all()
    assert len(records) == 10
    assert all(r.correction for r in records)
    assert {r.trade_id for r in records} == {t.id for t in session.exec(select(Trade)).all()}

    # Re-running after the batch reviews nothing new (already reviewed).
    res = agent.auto_review(threshold=10, limit=10)
    assert res["triggered"] is False
    assert res["count"] == 0
    assert len(session.exec(select(LTRRecord)).all()) == 10


def test_auto_review_limit_batches_oldest_first(session, monkeypatch):
    """Once a review already ran, newer losses queue for the next threshold hit."""
    from app.agents.ltr import LTRAgent
    from app.agents.validator import ValidatorChain

    def fake(self, prompt):
        return (
            ('{"loss_reason": "boom", "correction": "skip", '
             '"review": "avoid."}'),
            "test",
        )

    monkeypatch.setattr(ValidatorChain, "generate", fake)
    for i in range(10):
        _trade(session, pnl=-(i + 1), reviewed=0, symbol=f"Y{i}")
    agent = LTRAgent(session)
    assert agent.auto_review(threshold=10, limit=10)["count"] == 10
    # 11th loss is unreviewed and now the only pending one.
    _trade(session, pnl=-200, reviewed=0, symbol="EURUSD")
    res = agent.auto_review(threshold=1, limit=10)
    assert res["triggered"] is True
    assert res["count"] == 1
    assert res["reviewed"][0]["symbol"] == "EURUSD"
