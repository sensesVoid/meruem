from datetime import datetime, timezone

from sqlmodel import Session

from app.models.entities import Trade
from app.services.settings_service import SettingsService


class PositionManager:
    """Dynamic trade management: multi-TP, trailing stop, auto-breakeven rules."""

    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)

    def evaluate_auto_be(self, trade: Trade, current_price: float) -> bool | None:
        """Returns True if a new SL should be applied (move to entry or TP gain)."""
        cfg = self.settings.get_all()
        if not cfg.get("auto_breakeven_enabled", True):
            return None
        trigger = cfg.get("auto_be_trigger", "tp1")
        if trade.closed_at is not None:
            return None

        if trigger == "tp1" and trade.tp1 is not None:
            hit = (
                current_price >= trade.tp1
                if trade.direction == "BUY"
                else current_price <= trade.tp1
            )
            if hit and abs(trade.sl - trade.entry_price) > 1e-9:
                trade.sl = trade.entry_price
                return True
        elif trigger == "r_multiple":
            r = cfg.get("auto_be_r_multiple", 1.0)
            risk = abs(trade.entry_price - trade.sl)
            if risk > 0:
                target = (
                    trade.entry_price + r * risk
                    if trade.direction == "BUY"
                    else trade.entry_price - r * risk
                )
                hit = (
                    current_price >= target
                    if trade.direction == "BUY"
                    else current_price <= target
                )
                if hit and abs(trade.sl - trade.entry_price) > 1e-9:
                    trade.sl = trade.entry_price
                    return True
        self.session.commit()
        return None

    def apply_trailing_stop(self, trade: Trade, current_price: float) -> float | None:
        cfg = self.settings.get_all()
        if not cfg.get("trailing_stop_enabled", True) or trade.direction != "BUY":
            return None
        act = cfg.get("trailing_stop_activation_r", 1.5)
        step = cfg.get("trailing_stop_step_pips", 10.0)
        risk = abs(trade.entry_price - trade.sl)
        if risk <= 0 or current_price < trade.entry_price:
            return None
        activation = trade.entry_price + act * risk
        if current_price >= activation:
            new_sl = current_price - step
            if new_sl > trade.sl:
                trade.sl = round(new_sl, 5)
                self.session.commit()
                return trade.sl
        return None

    def peek_managed_sl(self, trade: Trade, current_price: float) -> float | None:
        """Non-mutating projection of the SL the trade *would* hold right now.

        Applies auto-breakeven then trailing-stop rules against `current_price`
        without committing anything, so the chart can render the live managed SL
        (move-to-breakeven / ratcheting trailing stop) as an authoritative line.
        Returns the effective SL price (possibly the original ``trade.sl``).
        """
        cfg = self.settings.get_all()
        if trade.closed_at is not None or current_price is None:
            return trade.sl
        eff = trade.sl

        # Auto-breakeven: once price reaches trigger, SL moves to entry.
        auto_be = cfg.get("auto_breakeven_enabled", True)
        if auto_be:
            trigger = cfg.get("auto_be_trigger", "tp1")
            if trigger == "tp1" and trade.tp1 is not None:
                hit = current_price >= trade.tp1 if trade.direction == "BUY" \
                    else current_price <= trade.tp1
                if hit:
                    eff = trade.entry_price
            elif trigger == "r_multiple":
                r = cfg.get("auto_be_r_multiple", 1.0)
                risk = abs(trade.entry_price - trade.sl)
                target = trade.entry_price + r * risk if trade.direction == "BUY" \
                    else trade.entry_price - r * risk
                if risk > 0 and (
                    current_price >= target if trade.direction == "BUY"
                    else current_price <= target
                ):
                    eff = trade.entry_price

        # Trailing stop (BUY only today). Without mutating, project what a step
        # ratchet would produce: SL trails step below the high-water mark.
        if cfg.get("trailing_stop_enabled", True) and trade.direction == "BUY":
            act = cfg.get("trailing_stop_activation_r", 1.5)
            step = cfg.get("trailing_stop_step_pips", 10.0)
            risk = abs(trade.entry_price - trade.sl)
            if risk > 0:
                peak = max(current_price, trade.entry_price)
                activation = trade.entry_price + act * risk
                if peak >= activation:
                    new_sl = peak - step
                    if new_sl > eff:
                        eff = round(new_sl, 5)
        return eff

    def partial_close(self, trade: Trade, fraction: float) -> None:
        """Close a fraction of a position (sizing stub; PnL needs live broker)."""
        trade.lot_size = round(trade.lot_size * (1 - fraction), 2)
        self.session.commit()

    def close(self, trade: Trade, reason: str = "manual",
              exit_price: float | None = None) -> Trade:
        """Close a position, recording why it closed and (when known) the exit
        price so LTR / audit have the context behind the outcome."""
        trade.closed_at = datetime.now(timezone.utc)
        if exit_price is not None:
            trade.exit_price = float(exit_price)
        trade.close_reason = reason or trade.close_reason or "manual"
        self.session.commit()
        return trade