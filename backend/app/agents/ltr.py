"""LoseTradeReview (LTR) Agent.

Phase 8.4: reviews the most recent losing trades (closed, pnl < 0, not yet
reviewed) and writes, for each, a *journal of error* — why the trade lost — plus
an actionable AI review capturing the root cause and the lesson. Each reviewed
loss is snapshotted into the ``ltr_records`` reference DB (``LTRRecord``) so
other agents / retraining can reference the WHY and the HOW-IT-SHOULD-HAVE-BEEN-
DONE correction without touching live trade rows.

Run is AUTOMATIC: ``auto_review()`` triggers once ``ltr_review_threshold`` losing
trades are pending (default 10), invoked by the scheduler. Previously reviewed
losing trades are skipped (``ltr_reviewed`` == 1), so a loss is never re-reviewed.
"""

from sqlmodel import Session, select

from app.agents.ollama_client import parse_json
from app.agents.validator import ValidatorChain
from app.models.entities import LTRRecord, Trade

LTR_PROMPT = """\
You are Meruem LoseTradeReview (LTR) Agent. A losing trade is shown below. Write a
concise JOURNAL OF ERROR (why the trade lost) and an AI REVIEW (root cause + the
process lesson a synthesizer should learn so the strategy is improved). Then give
the CORRECTION: exactly how the trade SHOULD have been done differently so the
retraining data can turn this loss into an explicit rule change.

Trade data:
  symbol: {symbol}
  direction: {direction}
  entry: {entry_price}   SL: {sl}   TP1: {tp1}   TP2: {tp2}
  lot size: {lot_size}
  opened: {opened_at}   closed: {closed_at}
  PnL: {pnl}
  exit price: {exit_price}

Return STRICT JSON only:
{{
  "loss_reason": "one-two sentence journal of the concrete error behind this loss "
                 "(e.g. entered on weak momentum / stop too tight / no confluence)",
  "correction": "one-two sentence HOW it should have been done: the concrete rule "
                "change (entry trigger, stop distance, size, wait condition) that "
                "would have avoided or reversed this specific loss",
  "review": "2-4 sentence root-cause analyse + a specific lesson: what to avoid or \\
             change in the strategy entry/risk rules next time"
}}
"""


class LTRAgent:
    """LTR: review unreviewed losing trades and produce error journals + lessons."""

    def __init__(self, session: Session):
        self.session = session
        self.chain = ValidatorChain(agent_id="ltr")

    def pending_count(self) -> int:
        """Number of losing trades still waiting for a review."""
        return len(self._unreviewed_losses(limit=10**6))

    def auto_review(self, threshold: int = 10, limit: int = 10) -> dict:
        """Automated LTR entry point (scheduler): review once enough losses queue up.

        Returns ``triggered`` false while pending losses are below ``threshold``;
        otherwise reviews the newest ``limit`` and marks them done so they are
        never re-reviewed. ``lessons`` is the learning-context block for the
        Synthesizer Agent.
        """
        pending = self.pending_count()
        if pending < threshold:
            return {
                "triggered": False,
                "pending": pending,
                "threshold": threshold,
                "count": 0,
                "reviewed": [],
                "lessons": "",
            }
        out = self.review(limit=limit)
        out["triggered"] = True
        out["pending"] = pending
        out["threshold"] = threshold
        return out

    def review(self, limit: int = 10) -> dict:
        """Review the most recent unreviewed LOSSING trades.

        Returns a summary dict with ``reviewed`` (list of per-trade reviews) plus
        ``lessons`` (the combined learning-context block for the synthesizer).
        Each reviewed loss is also snapshotted into ``ltr_records``.
        """
        losing = self._unreviewed_losses(limit)
        reviewed = []
        for trade in losing:
            review = self._review_one(trade)
            trade.loss_reason = review["loss_reason"]
            trade.ltr_review = review["review"]
            trade.ltr_reviewed = 1
            self.session.add(trade)
            self._snapshot(trade, review)
            reviewed.append({
                "trade_id": trade.id,
                "symbol": trade.symbol or trade.account_id,
                "direction": trade.direction,
                "pnl": trade.pnl,
                "loss_reason": review["loss_reason"],
                "correction": review["correction"],
                "review": review["review"],
            })
        self.session.commit()

        lessons = self._lessons_block(reviewed)
        return {
            "reviewed": reviewed,
            "count": len(reviewed),
            "lessons": lessons,
        }

    def _snapshot(self, trade: Trade, review: dict) -> None:
        """Persist the reviewed loss into the ltr_records reference DB.

        One snapshot per trade: if the row already exists (e.g. re-run), update it
        in place instead of duplicating — the trade itself is never re-reviewed.
        """
        record = self.session.exec(
            select(LTRRecord).where(LTRRecord.trade_id == trade.id)
        ).first()
        if record is None:
            record = LTRRecord(
                id=f"ltr-{trade.id}",
                trade_id=trade.id,
                account_id=trade.account_id,
            )
            self.session.add(record)
        record.symbol = trade.symbol or trade.account_id
        record.direction = trade.direction
        record.entry_price = trade.entry_price
        record.sl = trade.sl
        record.tp1 = trade.tp1
        record.exit_price = trade.exit_price
        record.pnl = trade.pnl
        record.close_reason = trade.close_reason
        record.opened_at = trade.opened_at
        record.closed_at = trade.closed_at
        record.loss_reason = review["loss_reason"]
        record.correction = review["correction"]
        record.review = review["review"]
        record.provider_used = "ltr"

    def _unreviewed_losses(self, limit: int) -> list[Trade]:
        return self.session.exec(
            select(Trade).where(
                Trade.closed_at.is_not(None),
                Trade.pnl.is_not(None),
                Trade.pnl < 0,
                Trade.ltr_reviewed == 0,
            ).order_by(Trade.closed_at.desc()).limit(limit)
        ).all()

    def _review_one(self, trade: Trade) -> dict:
        context = {
            "symbol": trade.symbol or trade.account_id,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "sl": trade.sl,
            "tp1": trade.tp1,
            "tp2": trade.tp2,
            "lot_size": trade.lot_size,
            "opened_at": trade.opened_at,
            "closed_at": trade.closed_at,
            "pnl": trade.pnl,
            "exit_price": trade.exit_price,
        }
        try:
            prompt = LTR_PROMPT.format(**context)
            raw, _provider = self.chain.generate(prompt)
            data = parse_json(raw)
            reason = str(data.get("loss_reason") or "").strip()
            correction = str(data.get("correction") or "").strip()
            review = str(data.get("review") or "").strip()
            if reason and review:
                return {
                    "loss_reason": reason,
                    "correction": correction or reason,
                    "review": review,
                }
        except Exception:  # noqa: BLE001, S110 - offline fallback below
            pass
        return self._heuristic(trade)

    @staticmethod
    def _heuristic(trade: Trade) -> dict:
        reason = (f"{trade.direction} trade on {trade.symbol or trade.account_id} "
                  f"closed at PnL {trade.pnl} — likely stopped out.")
        if trade.tp1 is not None and trade.pnl and trade.pnl <= 0:
            reason = (f"{trade.direction} trade on {trade.symbol or trade.account_id} "
                      f"closed at PnL {trade.pnl}: never reached TP1 "
                      f"({trade.tp1}) and exited at {trade.sl}.")
        return {
            "loss_reason": reason,
            "correction": (f"Verify the {trade.symbol or 'asset'} entry had a real "
                           f"trigger and the stop ({trade.sl}) was beyond the noise "
                           f"before sizing {trade.lot_size} lots; otherwise skip."),
            "review": ("No LLM at the moment — mark the entry, the stop distance and "
                       "the size. Verify the entry was pinned to a real trigger and the "
                       "stop was outside the noise, then re-test the setup."),
        }

    @staticmethod
    def _lessons_block(reviewed: list[dict]) -> str:
        if not reviewed:
            return ""
        lines = ["# Learning from recent losing trades (LTR)", ""]
        for r in reversed(reviewed):
            lines.append(f"## {r['symbol']} {r['direction']} (PnL {r['pnl']})")
            lines.append(f"- Error: {r['loss_reason']}")
            lines.append(f"- Fix: {r['correction']}")
            lines.append(f"- Lesson: {r['review']}")
            lines.append("")
        return "\n".join(lines)
