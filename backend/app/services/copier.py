from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.entities import Account, Trade
from app.services.broker_adapters import get_broker_adapter
from app.services.settings_service import SettingsService


@dataclass
class LotPlan:
    account_id: str
    account_type: str
    lot_size: float
    mode: str
    detail: str
    equity: float | None = None
    connected: bool = False


@dataclass
class _EquityCache:
    """Short-TTL cache for live broker equity.

    Live equity resolution logs into the broker (MT5) or hits the exchange
    API (Binance), which is expensive and must not run on every dashboard poll
    or account query. Values are cached for a few seconds.
    """

    values: dict[str, float] = field(default_factory=dict)
    timestamps: dict[str, float] = field(default_factory=dict)
    ttl: float = 5.0

    def get(self, key: str) -> float | None:
        ts = self.timestamps.get(key)
        if ts is None or (self._now() - ts) > self.ttl:
            return None
        return self.values.get(key)

    def put(self, key: str, value: float) -> None:
        self.values[key] = value
        self.timestamps[key] = self._now()

    def _now(self) -> float:
        return datetime.now(timezone.utc).timestamp()


class CopierEngine:
    """Routes master execution to slave accounts with configurable lot sizing.

    Each account's equity is resolved live (when broker credentials are stored
    and connected) and falls back to the account ``balance`` or the
    ``default_account_equity`` setting so the engine still runs pre-live.
    """

    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)
        self._equity_cache: _EquityCache = _EquityCache()
        self._status_cache: _EquityCache = _EquityCache(ttl=10.0)
        self._last_order_id: str | None = None
        self._last_error: str = ""

    def compute_slave_lots(
        self,
        master_lot: float,
        master_equity: float,
        slave_equity: float,
        mode: str,
        scale_factor: float,
        sl_distance_pips: float,
        pip_value: float,
        risk_pct: float,
    ) -> tuple[float, str]:
        mode = mode or "proportional"
        if mode == "proportional":
            lot = master_lot * (slave_equity / master_equity) * scale_factor
            detail = f"proportional: {master_lot}*({slave_equity}/{master_equity})*{scale_factor}"
        elif mode == "fixed":
            lot = master_lot * scale_factor
            detail = f"fixed: {master_lot}*{scale_factor}"
        elif mode == "risk_pct":
            if sl_distance_pips <= 0 or pip_value <= 0:
                return 0.0, "risk_pct: invalid SL distance or pip value"
            lot = (slave_equity * risk_pct) / (sl_distance_pips * pip_value)
            detail = f"risk_pct: ({slave_equity}*{risk_pct})/({sl_distance_pips}*{pip_value})"
        else:
            return 0.0, f"unknown mode: {mode}"
        lot = round(max(0.0, lot), 2)
        return lot, detail

    def fan_out(
        self,
        master_account: Account,
        master_lot: float,
        master_equity: float,
        sl_distance_pips: float,
        pip_value: float,
        entry_price: float,
        sl: float,
        tp: dict,
        direction: str,
        scan_id: str | None,
        mode: str = "manual",
        dry_run: bool = False,
        symbol: str = "",
    ) -> list[LotPlan]:
        """Build lot plans and (when not dry-run) place live orders.

        Every account that is asked to execute attempts a REAL broker order via the
        MT5 / Binance adapter. A Trade row is only persisted when the broker
        confirmed placement (``broker_order_id`` set); otherwise the row carries the
        ``broker_error`` so the dashboard never shows a phantom executed trade.
        """
        plans: list[LotPlan] = []
        symbol = symbol or getattr(master_account, "login", "") or ""
        placed = not dry_run and self._place(
            master_account, symbol, direction, master_lot,
            entry_price, sl, tp,
        )
        master_plan = LotPlan(
            account_id=master_account.id,
            account_type="master",
            lot_size=master_lot,
            mode="manual_input",
            detail="master order",
            equity=master_equity,
            connected=bool(master_account.enabled),
        )
        plans.append(master_plan)

        if not dry_run and placed:
            self.session.add(
                Trade(
                    id=f"{master_account.id}-m-{datetime.now(timezone.utc).timestamp():.0f}",
                    account_id=master_account.id,
                    scan_id=scan_id,
                    entry_price=entry_price,
                    sl=sl,
                    tp1=tp.get("tp1"),
                    tp2=tp.get("tp2"),
                    tp3=tp.get("tp3"),
                    tp4=tp.get("tp4"),
                    direction=direction,
                    lot_size=master_lot,
                    mode="master",
                    symbol=symbol,
                    broker=master_account.broker,
                    broker_order_id=self._last_order_id,
                )
            )

        slaves = self.session.exec(
            select(Account).where(Account.account_type == "slave")
        ).all()
        cfg = self.settings.get_all()

        for slave in slaves:
            if not slave.enabled:
                continue
            slave_equity = self._account_equity(slave)
            scaling_mode = slave.risk_scaling_mode
            scale = slave.scale_factor
            if scaling_mode == "risk_pct":
                scale = cfg.get("master_risk_pct", 1.0) / 100.0
            lot, detail = self.compute_slave_lots(
                master_lot=master_lot,
                master_equity=master_equity,
                slave_equity=slave_equity,
                mode=scaling_mode,
                scale_factor=scale,
                sl_distance_pips=sl_distance_pips,
                pip_value=pip_value,
                risk_pct=scale,
            )
            connected = self._account_connected(slave)
            plans.append(
                LotPlan(slave.id, "slave", lot, scaling_mode, detail,
                        equity=slave_equity, connected=connected)
            )

            if not dry_run and lot > 0:
                placed = self._place(
                    slave, symbol, direction, lot, entry_price, sl, tp,
                )
                if placed:
                    self.session.add(
                        Trade(
                            id=f"{slave.id}-{datetime.now(timezone.utc).timestamp():.0f}",
                            account_id=slave.id,
                            scan_id=scan_id,
                            entry_price=entry_price,
                            sl=sl,
                            tp1=tp.get("tp1"),
                            tp2=tp.get("tp2"),
                            tp3=tp.get("tp3"),
                            tp4=tp.get("tp4"),
                            direction=direction,
                            lot_size=lot,
                            mode=scaling_mode,
                            symbol=symbol,
                            broker=slave.broker,
                            broker_order_id=self._last_order_id,
                        )
                    )

        if not dry_run:
            self.session.commit()
        return plans

    def _place(self, account: Account, symbol: str, direction: str, lot: float,
               entry_price: float, sl: float, tp: dict) -> bool:
        """Place a live broker order for one account. Returns True on confirmation.

        Uses the universal broker connector (registry factory). MT5 keeps its
        documented native-retcode confirmation semantics; every other broker
        (Binance, MEXC, ...) returns the normalized OrderResult.
        """
        self._last_order_id = None
        if lot <= 0:
            return False
        key = self._account_key(account)
        try:
            adapter = get_broker_adapter(account.broker, account_key=key)
            if adapter is None:
                self._last_error = f"unsupported broker: {account.broker}"
                return False
            if account.broker == "MT5":
                if not adapter.connect_stored():
                    return False
                result = adapter.place_order(
                    symbol=symbol, action=direction, lot=lot, sl=sl, tp=tp.get("tp1"),
                )
                if result is None:
                    return False
                ok = bool(getattr(result, "retcode", None) == 10009)  # TRADE_RETCODE_DONE
                self._last_order_id = (
                    str(getattr(result, "order", 0) or 0) if ok else None)
                if not ok:
                    self._last_error = (
                        f"MT5 retcode {getattr(result, 'retcode', '?')}: "
                        f"{getattr(result, 'comment', '')}")
                return ok
            if account.broker == "Binance":
                side = "buy" if direction == "BUY" else "sell"
                res = adapter.place_order(
                    symbol=symbol, side=side, amount=lot, order_type="market")
                if res.get("ok"):
                    order = res.get("order") or {}
                    self._last_order_id = str(order.get("id") or order.get("clientOrderId") or "")
                return bool(res.get("ok"))
            # Normalized OrderResult path (MEXC + future brokers)
            res = adapter.place_order(
                symbol=symbol, action=direction, lot=lot, sl=sl, tp=tp.get("tp1"),
            )
            if res.get("ok"):
                self._last_order_id = res.get("order_id")
            else:
                self._last_error = res.get("error") or "order rejected"
            return bool(res.get("ok"))
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _account_key(self, account: Account) -> str:
        """Username used in the OS keyring for this account's credentials."""
        ref = account.encrypted_ref or account.login or "default"
        return ref.rsplit(":", 1)[-1] if ":" in ref else ref

    def _account_connected(self, account: Account) -> bool:
        """True if the account has credentials stored for its broker."""
        from app.services.broker_adapters import keyring_namespace
        from app.services.vault import VaultService

        if not account.enabled:
            return False
        ns = keyring_namespace(account.broker)
        if ns is None:
            return False
        return bool(VaultService.get(ns, self._account_key(account)))

    def _account_status(self, account: Account) -> tuple[bool, str]:
        """Live connection status + human-readable reason (green/red)."""
        from app.services.vault import VaultService

        if not account.enabled:
            return False, "account disabled"
        cached = self._status_cache.get(account.id)
        if cached is not None:
            return bool(cached[0]), cached[1]
        key = self._account_key(account)
        from app.services.broker_adapters import keyring_namespace as _ns

        ns = _ns(account.broker)
        if ns is None:
            self._status_cache.put(account.id, (False, "unsupported broker"))
            return (False, "unsupported broker")
        if not VaultService.get(ns, key):
            result = (False, f"no {account.broker} credentials saved for this account")
        else:
            try:
                adapter = get_broker_adapter(account.broker, account_key=key)
                if adapter is None:
                    result = (False, "unsupported broker")
                elif not adapter.connect_stored():
                    result = (False, f"{account.broker} unreachable or credentials rejected")
                else:
                    if account.broker == "MT5":
                        equity = adapter.fetch_equity()
                        result = (
                            (True, f"connected to MT5 (equity {equity:,.2f})")
                            if equity is not None
                            else (False, "MT5 reachable but account info unavailable")
                        )
                    else:
                        bal = adapter.fetch_balance()
                        total = bal.get("total")
                        quote = bal.get("quote", "USDT")
                        result = (
                            (True, f"connected to {account.broker} (balance {total:,.2f} {quote})")
                            if total is not None
                            else (False, f"{account.broker} reached but API rejected the key (bad permissions?)")
                        )
            except Exception as exc:  # noqa: BLE001
                result = (False, f"{account.broker} connection failed: {type(exc).__name__}: {exc}")
        self._status_cache.put(account.id, result)
        return result

    def _account_equity(self, account: Account) -> float:
        """Live broker equity, else account balance, else the configured default."""
        live = self._live_equity(account)
        if live is not None:
            return live
        balance = getattr(account, "balance", None)
        if balance:
            return float(balance)
        return float(self.settings.get("default_account_equity", 10000.0))

    def _live_equity(self, account: Account) -> float | None:
        """Best-effort real equity from the broker; None when unavailable.

        Uses stored credentials without storing plaintext locally beyond the OS
        keyring. Results are cached for a few seconds to avoid hammering the
        broker on every dashboard poll / account query. Returns None on any
        connectivity/auth failure so the caller falls back to a configured
        equity.
        """
        if not account.enabled:
            return None
        key = self._account_key(account)
        cached = self._equity_cache.get(key)
        if cached is not None:
            return cached
        try:
            adapter = get_broker_adapter(account.broker, account_key=key)
            if adapter is None:
                return None
            equity: float | None = None
            if account.broker == "Binance":
                bal = adapter.fetch_balance()
                total = bal.get("total")
                equity = float(total) if total else None
            elif account.broker == "MT5":
                if adapter.connect_stored():
                    equity = adapter.fetch_equity()
            else:
                # Normalized path: fetch_equity() or balance total (MEXC + future)
                equity = adapter.fetch_equity()
                if equity is None:
                    bal = adapter.fetch_balance()
                    total = bal.get("total")
                    equity = float(total) if total else None
            if equity is not None:
                self._equity_cache.put(key, equity)
            return equity
        except Exception:  # noqa: BLE001
            return None
