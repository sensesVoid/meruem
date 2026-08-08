"""RiskManager — Phase 7 Risk Management Agent (code-first, LLM advisory).

Risk planning is DETERMINISTIC on the hot path (same precedent as RegimeAgent:
no LLM as the source of truth for safety). The ``risk_manager`` agent only
writes the narrative advisory for the panel; every go/no-go and sizing decision
is computed here and reused by the executor, the scheduler and the risk UI.

Composes :class:`RiskInterceptor` for the guardrail verdict (auto-gate, max
concurrent, daily loss %, drawdown %) and adds exposure, per-account equity and
the shared risk-pct / proportional / fixed lot formulas so the clearance and the
plan are the single authoritative sizing source.
"""

from datetime import datetime, timezone

from sqlmodel import Session, func, select

from app.models.entities import Account, Trade
from app.services.copier import CopierEngine
from app.services.risk_interceptor import RiskInterceptor
from app.services.settings_service import SettingsService

PIP_VALUE = 10.0  # value per 0.0001 move for a 1.0 lot (orchestrator-consistent)


class RiskManager:
    """Authoritative risk plan + clearance + lot sizing (code-first)."""

    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)
        self.interceptor = RiskInterceptor(session)

    # ---------------------------------------------------------------- plan

    def plan(self, symbol: str | None = None) -> dict:
        """Full risk plan: guardrail verdict, exposure, accounts, lot sizing."""
        cfg = self.settings.get_all()
        decision = self.interceptor.evaluate()
        accounts = self._enabled_accounts()
        exposure = self._exposure()
        equity = self._current_equity()

        plan = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision.to_dict(),
            "exposure": exposure,
            "equity": round(equity, 2),
            "risk_pct": float(cfg.get("master_risk_pct", 1.0)),
            "lot_mode": cfg.get("master_lot_sizing_mode", "risk_pct"),
            "accounts": accounts,
            "lots": self._size_default(accounts),
        }
        plan["advisory"] = self._advisory(plan)
        return plan

    def clearance(self, account_id: str | None = None,
                  symbol: str = "", direction: str = "BUY",
                  entry_price: float = 0.0, sl: float = 0.0) -> dict:
        """Go/No-go for one concrete proposed trade + recommended sizing.

        Deterministic: the same RiskInterceptor guard the auto path uses, then a
        suggested lot per enabled account for THIS trade's SL distance.
        """
        cfg = self.settings.get_all()
        if not cfg.get("auto_execution_enabled"):
            return self._clear_response(False, "Auto-execution is disabled")
        decision = self.interceptor.evaluate(account_id=account_id or None,
                                             require_auto=True)
        if not decision.allowed:
            return self._clear_response(False, decision.reason, decision.detail)

        accounts = self._enabled_accounts()
        sl_distance = abs(float(entry_price) - float(sl)) if entry_price and sl else 0.0
        lots = self._size_for_distance(accounts, sl_distance)
        return self._clear_response(
            True, "All risk guardrails passed",
            {"lots": lots, "sl_distance": round(sl_distance, 6)})

    # ------------------------------------------------------------ short-cut

    def stop_scan_reason(self) -> str:
        """Non-empty message when scanning should short-circuit (no LLM wasted).

        Returns the reason when auto-execution is disabled OR the global gate is
        closed OR max concurrent auto-trades is already reached — in any of
        which a scan's only outcome would be a blocked trade.
        """
        cfg = self.settings.get_all()
        if not cfg.get("auto_execution_enabled"):
            return ""
        if cfg.get("system_status") == "PAUSED_LLM_LIMIT":
            return "system PAUSED_LLM_LIMIT"
        decision = self.interceptor.evaluate(require_auto=True)
        if not decision.allowed:
            return decision.reason
        open_auto = self._open_auto()
        max_concurrent = int(cfg.get("max_concurrent_auto_trades", 2))
        if open_auto >= max_concurrent:
            return (f"max concurrent auto-trades reached "
                    f"({open_auto}/{max_concurrent})")
        return ""

    # ---------------------------------------------------------------- account

    def _enabled_accounts(self) -> list[dict]:
        rows = self.session.exec(
            select(Account).where(Account.enabled.is_(True))
            .order_by(Account.account_type, Account.id)
        ).all()
        out = []
        for acc in rows:
            out.append({
                "id": acc.id,
                "type": acc.account_type,
                "nickname": acc.nickname,
                "broker": acc.broker,
                "mode": acc.risk_scaling_mode,
                "scale": acc.scale_factor,
                "equity": round(self._account_equity(acc), 2),
            })
        return out

    def _exposure(self) -> dict:
        open_auto = self._open_auto()
        max_concurrent = int(self.settings.get("max_concurrent_auto_trades", 2))
        open_all = int(self.session.exec(
            select(func.count(Trade.id)).where(Trade.closed_at.is_(None))
        ).one())
        return {
            "open_trades": open_all,
            "open_auto": open_auto,
            "slots_free": max(0, max_concurrent - open_auto),
            "max_concurrent": max_concurrent,
        }

    def _open_auto(self) -> int:
        return int(self.session.exec(
            select(func.count(Trade.id)).where(
                Trade.mode == "auto", Trade.closed_at.is_(None))
        ).one())

    def _account_equity(self, account: Account) -> float:
        try:
            equity = CopierEngine(self.session)._account_equity(account)
            if equity and equity > 0:
                return float(equity)
        except Exception:  # noqa: BLE001, S110 - fall back to stored balance
            pass
        if account.balance and account.balance > 0:
            return float(account.balance)
        return float(self.settings.get("default_account_equity", 10000.0) or 10000.0)

    def _current_equity(self) -> float:
        master = self.session.exec(
            select(Account).where(
                Account.account_type == "master", Account.enabled.is_(True))
        ).first()
        if master is not None:
            return self._account_equity(master)
        return float(self.settings.get("default_account_equity", 10000.0) or 10000.0)

    # ------------------------------------------------------------ lot sizing

    def _size_for_distance(self, accounts: list[dict], sl_distance: float) -> list[dict]:
        """Per-account lot at a concrete SL distance (shared math with copier)."""
        cfg = self.settings.get_all()
        mode = cfg.get("master_lot_sizing_mode", "risk_pct")
        risk_pct = float(cfg.get("master_risk_pct", 1.0)) / 100.0
        manual_lot = float(cfg.get("master_manual_lot", 0.10))
        master_equity = self._current_equity()

        out = []
        for acc in accounts:
            eq = acc.get("equity") or master_equity
            if eq <= 0 or sl_distance <= 0:
                out.append({"account_id": acc["id"], "type": acc["type"],
                            "lot": 0.0, "detail": "no equity/distance"})
                continue
            if mode == "manual" and acc["type"] == "master":
                lot, detail = manual_lot, f"manual: {manual_lot}"
            elif mode == "manual":
                lot, detail = self._slave_lot(acc, sl_distance, master_equity)
            elif acc["type"] == "master":
                lot = (eq * risk_pct) / (sl_distance * PIP_VALUE)
                detail = f"risk_pct: ({eq:.0f}*{risk_pct:.4f})/({sl_distance}*{PIP_VALUE})"
            else:
                lot, detail = self._slave_lot(acc, sl_distance, master_equity)
            out.append({"account_id": acc["id"], "type": acc["type"],
                        "lot": round(max(0.0, lot), 2), "detail": detail})
        return out

    def _size_default(self, accounts: list[dict] | None = None) -> list[dict]:
        """Reference sizing at the last auto-trade's SL distance (or ~0 for none)."""
        if accounts is None:
            accounts = self._enabled_accounts()
        last = self.session.exec(
            select(Trade).where(
                Trade.mode == "auto", Trade.closed_at.is_(None),
                Trade.pnl.is_not(None))
            .order_by(Trade.opened_at.desc())
        ).first()
        distance = 0.0
        if last is not None and last.entry_price and last.sl:
            distance = abs(last.entry_price - last.sl)
        return self._size_for_distance(accounts, distance)

    def _slave_lot(self, acc: dict, sl_distance: float,
                   master_equity: float) -> tuple[float, str]:
        mode = acc.get("mode") or "proportional"
        risk_pct = float(self.settings.get("master_risk_pct", 1.0)) / 100.0
        slave_equity = acc.get("equity") or 0.0
        if slave_equity <= 0 or sl_distance <= 0:
            return 0.0, f"{mode}: no equity/distance"
        if mode == "risk_pct":
            lot = (slave_equity * risk_pct) / (sl_distance * PIP_VALUE)
            detail = f"risk_pct: ({slave_equity:.0f}*{risk_pct:.4f})/({sl_distance}*{PIP_VALUE})"
        elif mode == "fixed":
            lot = float(self.settings.get("master_manual_lot", 0.10)) * (acc.get("scale") or 1.0)
            detail = f"fixed: manual*{acc.get('scale')}"
        else:  # proportional
            manual = float(self.settings.get("master_manual_lot", 0.10))
            lot = manual * (slave_equity / master_equity) * (acc.get("scale") or 1.0)
            detail = f"proportional: {manual}*({slave_equity}/{master_equity:.0f})*{acc.get('scale')}"
        return round(max(0.0, lot), 2), detail

    # -------------------------------------------------------------- advisory

    def _advisory(self, plan: dict) -> str:
        """LLM narrative of the risk plan (off-risk-safe, best-effort).

        The plan returns a deterministic fallback when the LLM is unavailable;
        the go/no-go itself never blocks on an LLM call.
        """
        base = (f"Guardrail: {plan['decision']['reason']}. "
                f"Exposure {plan['exposure']['open_auto']}/{plan['exposure']['max_concurrent']} "
                f"auto slot(s), equity {plan['equity']:,.0f}.")
        try:
            from app.agents.validator import ValidatorChain

            raw, _provider = ValidatorChain(agent_id="risk_manager").generate(
                "You are Meruem RiskManager. Summarize this risk plan in 2-3 crisp "
                "sentences, highlighting anything that blocks trading today and "
                "one concrete recommendation:\n" + str(plan)
            )
            out = str(raw).strip()
            if len(out) > 8:
                return out
        except Exception:  # noqa: BLE001, S110 - advisory is best-effort
            pass
        return base

    # ------------------------------------------------------------ responses

    def _clear_response(self, allowed: bool, reason: str,
                        detail: dict | None = None) -> dict:
        return {
            "allowed": allowed,
            "reason": reason,
            "detail": detail or {},
        }