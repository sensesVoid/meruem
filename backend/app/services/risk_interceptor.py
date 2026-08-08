from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.entities import Trade
from app.services.settings_service import SettingsService


class RiskDecision:
    def __init__(self, allowed: bool, reason: str, detail: dict | None = None):
        self.allowed = allowed
        self.reason = reason
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason, "detail": self.detail}


class RiskInterceptor:
    """Dynamically evaluates user-configured risk caps from SQLite before
    every order placement. Settings are re-fetched per evaluation so changes
    made in the UI apply immediately without restart."""

    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)

    def evaluate(self, account_id: str | None = None, require_auto: bool = True) -> RiskDecision:
        cfg = self.settings.get_all()

        if cfg.get("system_status") == "PAUSED_LLM_LIMIT":
            return RiskDecision(False, "System PAUSED_LLM_LIMIT: manual acknowledgement required")
        if require_auto and not cfg.get("auto_execution_enabled"):
            return RiskDecision(False, "Auto-execution is disabled")

        max_concurrent = int(cfg.get("max_concurrent_auto_trades", 2))
        open_auto, day_pnl, dd = self._guardrail_metrics()
        if open_auto >= max_concurrent:
            return RiskDecision(
                False,
                f"Max concurrent auto-trades reached ({open_auto}/{max_concurrent})",
                {"open_auto": open_auto, "max_concurrent": max_concurrent},
            )

        daily_cap = float(cfg.get("daily_loss_cap_pct", -3.0))
        if day_pnl <= daily_cap:
            return RiskDecision(
                False,
                f"Daily loss cap breached ({day_pnl:.2f}% <= {daily_cap:.2f}%)",
                {"day_pnl_pct": day_pnl, "cap_pct": daily_cap},
            )

        dd_ceiling = float(cfg.get("max_drawdown_pct", -8.0))
        if dd <= dd_ceiling:
            return RiskDecision(
                False,
                f"Max drawdown ceiling breached ({dd:.2f}% <= {dd_ceiling:.2f}%)",
                {"drawdown_pct": dd, "ceiling_pct": dd_ceiling},
            )

        return RiskDecision(True, "All risk guardrails passed")

    def _guardrail_metrics(self) -> tuple[int, float, float]:
        """One batched query computes the three guardrail inputs.

        Returns (open_auto_count, day_pnl_pct, total_drawdown_pct). Previously
        this ran three separate scans of the auto trades per evaluation.
        """
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        rows = self.session.exec(
            select(Trade.closed_at, Trade.pnl).where(Trade.mode == "auto")
        ).all()
        open_auto = 0
        day_total = 0.0
        total = 0.0
        for closed_at, pnl in rows:
            if closed_at is None:
                open_auto += 1
            if pnl is None:
                continue
            total += float(pnl)
            if closed_at is not None and closed_at >= start:
                day_total += float(pnl)
        equity = self._current_equity()
        if not equity:
            return open_auto, 0.0, 0.0
        return open_auto, (day_total / equity * 100.0), (total / equity * 100.0)

    def _current_equity(self) -> float:
        """Current equity for drawdown/daily-loss %.

        Prefers the live equity of the selected master account (the account the
        auto-execution actually trades), falling back to the configured default
        so the guardrails stay consistent with the copier's equity resolution.
        """
        try:
            from sqlmodel import select

            from app.models.entities import Account
            from app.services.copier import CopierEngine

            master_id = self.settings.get("selected_master_account_id", "")
            master = None
            if master_id:
                master = self.session.get(Account, master_id)
            if master is None or not master.enabled or master.account_type != "master":
                master = self.session.exec(
                    select(Account).where(
                        Account.account_type == "master", Account.enabled.is_(True)
                    )
                ).first()
            if master is not None:
                equity = CopierEngine(self.session)._account_equity(master)
                if equity and equity > 0:
                    return float(equity)
        except Exception:  # noqa: BLE001, S110 - roll back to the configured default.
            pass
        return float(self.settings.get("default_account_equity", 10000.0) or 10000.0)
