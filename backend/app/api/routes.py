import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.db import get_session
from app.models.entities import (
    Account,
    Asset,
    BacktestRun,
    Scan,
    Setting,
    Strategy,
    Trade,
    Validation,
)
from app.services.rag import RagService
from app.services.risk_interceptor import RiskInterceptor
from app.services.settings_service import SettingsService
from app.services.vault import VaultService
router = APIRouter()


class SettingPayload(BaseModel):
    key: str
    value: object


class VaultSave(BaseModel):
    namespace: str
    username: str
    secret: str


class TradePayload(BaseModel):
    account_id: str
    direction: str
    entry_price: float
    sl: float
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    tp4: float | None = None
    lot_size: float = 0.1
    mode: str = "manual"
    symbol: str = ""  # broker symbol — required for the chart overlay to match the trade to an asset


@router.get("/health")
def health(session: Session = Depends(get_session)):
    try:
        session.exec(select(Setting).limit(1)).first()
        db = "ok"
    except Exception as exc:  # noqa: BLE001
        db = str(exc)
    return {"status": "ok", "database": db, "service": "meruem-backend"}


@router.get("/status")
def status(session: Session = Depends(get_session)):
    svc = SettingsService(session)
    cfg = svc.get_all()
    return {
        "system_status": cfg.get("system_status", "READY"),
        "auto_execution_enabled": cfg.get("auto_execution_enabled", False),
        "risk": RiskInterceptor(session).evaluate().to_dict(),
    }


# ---------- Settings ----------

@router.get("/settings")
def get_settings(session: Session = Depends(get_session)):
    return SettingsService(session).get_all()


@router.put("/settings")
def update_setting(payload: SettingPayload, session: Session = Depends(get_session)):
    SettingsService(session).set(payload.key, payload.value)
    return {"ok": True, "key": payload.key}


@router.post("/settings/reset")
def reset_settings(session: Session = Depends(get_session)):
    SettingsService(session).reset_to_defaults()
    return {"ok": True}


# ---------- Vault ----------

@router.post("/vault")
def vault_save(payload: VaultSave):
    VaultService.save(payload.namespace, payload.username, payload.secret)
    return {"ok": True}


@router.get("/vault/{namespace}/{username}")
def vault_get(namespace: str, username: str):
    value = VaultService.get(namespace, username)
    return {"present": value is not None, "value": value if value else None}


@router.delete("/vault/{namespace}/{username}")
def vault_delete(namespace: str, username: str):
    VaultService.delete(namespace, username)
    return {"ok": True}


@router.get("/vault/status")
def vault_status():
    from app.services.model_config import get_providers

    providers = get_providers()
    slots = []
    for p in providers:
        ref = p.get("key_ref", "")
        if ":" in ref:
            ns, user = ref.split(":", 1)
        else:
            ns, user = "llm", ref
        slots.append(
            {
                "namespace": ns,
                "username": user,
                "configured": VaultService.has(ns, user),
            }
        )
    return {
        "slots": slots,
        "providers": providers,
    }


@router.get("/agents/status")
def agents_status():
    """Live readout of active AI agents and the models they use."""
    from app.services.agent_status import agent_status

    return agent_status()


@router.get("/agents/catalog")
def agents_catalog():
    """Agent catalog with per-agent assigned model + recommended models (task #12)."""
    from app.services.model_config import get_agent_catalog

    return {"agents": get_agent_catalog()}


@router.get("/providers/{provider_id}/models")
def provider_models(provider_id: str):
    """List models an OpenAI-compatible provider offers (via its /models).

    Also handles the local 'ollama' pseudo-provider (uses /api/tags). Lets the
    user *find* the model to assign to an agent instead of typing a name.
    """
    from app.services.model_config import get_providers, list_provider_models

    if provider_id == "ollama":
        try:
            from app.agents.ollama_client import OllamaClient
            from app.services.model_config import get as model_get

            return {"provider": "ollama",
                    "models": OllamaClient(base_url=model_get("ollama_url")).list_models()}
        except Exception as exc:  # noqa: BLE001
            return {"provider": "ollama", "models": [], "error": str(exc)}

    provider = next((p for p in get_providers() if p.get("id") == provider_id), None)
    if provider is None:
        raise HTTPException(404, "unknown provider")
    models = list_provider_models(provider)
    return {
        "provider": provider_id,
        "models": models,
        "error": None if models else "no models listed (missing key or endpoint)",
    }


# ---------- Ollama connector ----------

@router.get("/ollama/models")
def ollama_models():
    from app.agents.ollama_client import OllamaClient
    from app.services.model_config import get as model_get

    try:
        return {"models": OllamaClient(base_url=model_get("ollama_url")).list_models()}
    except Exception as exc:  # noqa: BLE001
        return {"models": [], "error": str(exc)}


@router.post("/ollama/test")
def ollama_test():
    from app.services.model_config import get as model_get

    try:
        from app.agents.ollama_client import OllamaClient

        models = OllamaClient(base_url=model_get("ollama_url")).list_models()
        return {"ok": True, "models": len(models)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------- Broker / exchange connections ----------

class MT5Connect(BaseModel):
    login: str
    server: str
    password: str


class BinanceConnect(BaseModel):
    api_key: str
    secret: str


class BrokerConnect(BaseModel):
    broker: str  # MT5 | Binance | MEXC | ... (see broker_adapters.BROKER_REGISTRY)
    account_type: str = "slave"  # master | slave
    nickname: str | None = None
    login: str = ""
    server: str = ""
    password: str = ""
    api_key: str = ""
    secret: str = ""
    risk_scaling_mode: str = "proportional"
    scale_factor: float = 1.0


def _test_broker(payload: BrokerConnect) -> tuple[bool, str]:
    """Validate live broker credentials. Returns (ok, detail)."""
    if payload.broker == "MT5":
        try:
            from app.services.mt5_adapter import MT5Adapter

            adapter = MT5Adapter(login=payload.login or "default")
            ok = adapter.connect(int(payload.login), payload.server, payload.password)
            return ok, "MT5 connection OK" if ok else (
                "MetaTrader5 not initialized / terminal not running, or login/server rejected")
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    if payload.broker == "Binance":
        try:
            import ccxt

            ex = ccxt.binance({"apiKey": payload.api_key, "secret": payload.secret})
            ex.fetch_time()
            if payload.api_key:
                ex.fetch_balance()
            return True, "Binance reachable, API credentials accepted"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    if payload.broker == "MEXC":
        try:
            from app.services.mexc_adapter import MEXCAdapter

            adapter = MEXCAdapter()
            ok = adapter.connect(payload.api_key, payload.secret)
            return (
                (True, "MEXC reachable, API credentials accepted")
                if ok
                else (False, "MEXC unreachable or API credentials rejected")
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    return False, "unsupported broker"


def _enforce_single_master(session: Session, master_id: str) -> None:
    """Demote all other masters so exactly one is active at a time."""
    for acc in session.exec(select(Account)).all():
        if acc.id == master_id:
            continue
        if acc.account_type == "master":
            acc.account_type = "slave"
            session.add(acc)
    session.commit()


@router.post("/brokers/connect")
def broker_connect(payload: BrokerConnect, session: Session = Depends(get_session)):
    """Validate credentials, persist them, and upsert an Account.

    Enforces a single master account. Returns the saved account plus the live
    connection status (ok / detail).
    """
    from app.services.broker_adapters import supported_brokers

    norm = {b.upper(): b for b in supported_brokers()}
    broker = norm.get(payload.broker.upper())
    if broker is None:
        raise HTTPException(
            400, f"broker must be one of {', '.join(supported_brokers())}")

    ok, detail = _test_broker(payload)
    if not ok:
        return {"ok": False, "detail": detail, "account": None}

    ns = broker.lower()
    if broker == "MT5":
        ref = payload.login or "default"
        VaultService.save(ns, ref, payload.password)
        if payload.server:
            VaultService.save(ns, f"{ref}:server", payload.server)
        login = payload.login
    else:
        # Binance / MEXC / future API-key brokers share the key+secret shape.
        ref = payload.api_key or payload.login or "default"
        VaultService.save(ns, ref, payload.api_key)
        VaultService.save(ns, f"{ref}:secret", payload.secret)
        login = payload.login or ref

    # Upsert by broker + login so a re-connect updates rather than duplicates.
    existing = None
    for acc in session.exec(select(Account)).all():
        if acc.broker == broker and (acc.login or acc.encrypted_ref) == (login or ref):
            existing = acc
            break

    if existing is None:
        existing = Account(
            id=f"acct-{broker.lower()}-{login or ref}".replace(" ", "-"),
            broker=broker,
            account_type=payload.account_type,
            nickname=payload.nickname or (login or ref),
            risk_scaling_mode=payload.risk_scaling_mode,
            scale_factor=payload.scale_factor,
            encrypted_ref=f"{broker.lower()}:{login or ref}",
            login=login or ref,
            server=payload.server if broker == "MT5" else "",
            enabled=True,
        )
    else:
        existing.nickname = payload.nickname or existing.nickname
        existing.server = payload.server if broker == "MT5" else existing.server

    if payload.account_type == "master":
        existing.account_type = "master"
        session.add(existing)
        session.commit()
        _enforce_single_master(session, existing.id)
        SettingsService(session).set("selected_master_account_id", existing.id)
    else:
        existing.account_type = "slave"
        session.add(existing)
        session.commit()

    session.refresh(existing)
    from app.services.copier import CopierEngine

    connected, reason = CopierEngine(session)._account_status(existing)
    return {
        "ok": True,
        "detail": detail,
        "connected": connected,
        "reason": reason,
        "account": existing.model_dump(mode="json"),
    }


@router.post("/brokers/test")
def broker_test(payload: BrokerConnect):
    ok, detail = _test_broker(payload)
    return {"ok": ok, "detail": detail}


@router.post("/brokers/mt5/connect")
def broker_mt5_connect(payload: MT5Connect):
    VaultService.save("mt5", payload.login or "default", payload.password)
    if payload.server:
        VaultService.save("mt5", f"{payload.login or 'default'}:server", payload.server)
    return {"ok": True, "credential": f"mt5:{payload.login or 'default'}"}


@router.post("/brokers/mt5/test")
def broker_mt5_test(payload: MT5Connect):
    try:
        from app.services.mt5_adapter import MT5Adapter

        adapter = MT5Adapter(login=payload.login)
        ok = adapter.connect(int(payload.login), payload.server, payload.password)
        detail = "connection OK" if ok else "MetaTrader5 not initialized / terminal not running"
        return {"ok": ok, "detail": detail}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


@router.post("/brokers/binance/connect")
def broker_binance_connect(payload: BinanceConnect):
    VaultService.save("binance", "default", payload.api_key)
    VaultService.save("binance", "default:secret", payload.secret)
    return {"ok": True}


@router.post("/brokers/binance/test")
def broker_binance_test(payload: BinanceConnect):
    try:
        import ccxt

        ex = ccxt.binance({"apiKey": payload.api_key, "secret": payload.secret})
        ex.fetch_time()  # public reachability
        if payload.api_key:
            ex.fetch_balance()
        return {"ok": True, "detail": "Binance reachable, API credentials accepted"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


@router.get("/brokers/status")
def broker_status(session: Session = Depends(get_session)):
    from app.services.copier import CopierEngine

    copier = CopierEngine(session)
    accounts = session.exec(select(Account)).all()
    accounts_status = []
    for acc in accounts:
        connected, reason = copier._account_status(acc)
        accounts_status.append({
            "account_id": acc.id,
            "nickname": acc.nickname,
            "broker": acc.broker,
            "account_type": acc.account_type,
            "login": acc.login,
            "enabled": acc.enabled,
            "connected": connected,
            "reason": reason,
        })
    master = next((a for a in accounts if a.account_type == "master"), None)
    from app.services.broker_adapters import keyring_namespace, supported_brokers

    configured = {
        b.lower(): {"configured": VaultService.has(keyring_namespace(b), "default")}
        for b in supported_brokers()
    }
    return {
        **configured,
        "accounts": accounts_status,
        "master_id": master.id if master else None,
    }


# ---------- Copier ----------

class CopierPayload(BaseModel):
    master_account_id: str | None = None
    master_lot: float = 0.1
    master_equity: float | None = None
    sl_distance_pips: float = 50.0
    pip_value: float = 10.0
    entry_price: float
    sl: float
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    tp4: float | None = None
    direction: str = "BUY"
    symbol: str = ""


@router.post("/copier/plan")
def copier_plan(payload: CopierPayload, session: Session = Depends(get_session)):
    from app.services.copier import CopierEngine

    master = None
    if payload.master_account_id:
        master = session.get(Account, payload.master_account_id)
        if master is not None and master.account_type != "master":
            raise HTTPException(400, "selected account is not a master account")
    if master is None:
        selected = SettingsService(session).get("selected_master_account_id", "")
        if selected:
            master = session.get(Account, selected)
        if master is None or master.account_type != "master" or not master.enabled:
            master = session.exec(
                select(Account).where(Account.account_type == "master", Account.enabled.is_(True))
            ).first()
    if master is None:
        raise HTTPException(404, "master account not found")

    copier = CopierEngine(session)
    equity = payload.master_equity or copier._account_equity(master)
    plans = copier.fan_out(
        master_account=master,
        master_lot=payload.master_lot,
        master_equity=equity,
        sl_distance_pips=payload.sl_distance_pips,
        pip_value=payload.pip_value,
        entry_price=payload.entry_price,
        sl=payload.sl,
        tp={"tp1": payload.tp1, "tp2": payload.tp2, "tp3": payload.tp3, "tp4": payload.tp4},
        direction=payload.direction,
        symbol=payload.symbol,
        scan_id=None,
        mode="manual",
        dry_run=True,
    )
    return {
        "plans": [vars(p) for p in plans],
        "master": {"id": master.id, "equity": equity, "account_type": master.account_type},
    }


# ---------- Accounts ----------

@router.get("/accounts")
def list_accounts(session: Session = Depends(get_session)) -> list[Account]:
    return session.exec(select(Account)).all()


@router.post("/accounts")
def create_account(payload: Account, session: Session = Depends(get_session)) -> Account:
    session.add(payload)
    session.commit()
    session.refresh(payload)
    return payload


class AccountUpdate(BaseModel):
    nickname: str | None = None
    account_type: str | None = None
    broker: str | None = None
    risk_scaling_mode: str | None = None
    scale_factor: float | None = None
    login: str | None = None
    server: str | None = None
    base_currency: str | None = None
    balance: float | None = None
    enabled: bool | None = None


@router.put("/accounts/{account_id}")
def update_account(account_id: str, payload: AccountUpdate,
                   session: Session = Depends(get_session)):
    acc = session.get(Account, account_id)
    if acc is None:
        raise HTTPException(404, "account not found")
    changes = payload.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(acc, field, value)
    session.add(acc)
    session.commit()
    session.refresh(acc)
    return acc


@router.get("/accounts/equity")
def account_equities(session: Session = Depends(get_session)):
    from app.services.copier import CopierEngine

    copier = CopierEngine(session)
    return {
        "equities": [
            {
                "account_id": a.id,
                "equity": copier._account_equity(a),
                "connected": copier._account_connected(a),
                "reason": copier._account_status(a)[1],
            }
            for a in session.exec(select(Account)).all()
        ]
    }


@router.delete("/accounts/{account_id}")
def delete_account(account_id: str, session: Session = Depends(get_session)):
    acc = session.get(Account, account_id)
    if acc is None:
        raise HTTPException(404, "account not found")
    session.delete(acc)
    session.commit()
    return {"ok": True}


# ---------- Dashboard (performance) ----------

def _account_performance(session: Session, copier, account: Account, is_master: bool,
                         trades_by_account: dict[str, list[Trade]]) -> dict:
    """Compute a single account's dashboard row: funded capital, running
    capital, gain/loss in $ and %, and open/closed trade stats.

    ``trades_by_account`` is pre-grouped by the caller (one query for ALL
    accounts) so the dashboard stays O(accounts + trades), not O(accounts²).
    """
    running = copier._account_equity(account)
    funded = float(account.balance) if account.balance is not None else None
    connected = copier._account_connected(account)
    reason = copier._account_status(account)[1]
    gain = None
    gain_pct = None
    if funded is not None and running is not None:
        gain = round(running - funded, 2)
        gain_pct = round((running - funded) / funded * 100.0, 2) if funded else None

    trades = trades_by_account.get(account.id, [])
    closed = [t for t in trades if t.closed_at is not None and t.pnl is not None]
    open_trades = sum(1 for t in trades if t.closed_at is None)
    return {
        "account_id": account.id,
        "nickname": account.nickname,
        "account_type": account.account_type,
        "broker": account.broker,
        "enabled": account.enabled,
        "connected": connected,
        "reason": reason,
        "funded": funded,
        "running_capital": running,
        "equity": running,
        "gain": gain,
        "gain_pct": gain_pct,
        "open_trades": open_trades,
        "closed_trades": len(closed),
        "realized_pnl": round(sum(t.pnl for t in closed), 2) if closed else 0.0,
        "is_master": is_master,
    }


def _performance_curve(trades: list[Trade], baseline: float) -> dict:
    """Tradezella-style equity curve + stats from CLOSED trades.

    ``baseline`` is the funded capital the master started with. The curve walks
    closed trades in chronological order, layering each realised PnL onto the
    running equity; stats (win rate, profit factor, avg win/loss, max drawdown)
    are derived from the curve so the Dashboard can render a proper performance
    chart, not just a number.
    """
    closed = sorted(
        [t for t in trades if t.closed_at is not None and t.pnl is not None],
        key=lambda t: t.closed_at or t.opened_at,
    )
    equity = baseline
    curve = [{"ts": None, "equity": round(baseline, 2), "pnl": 0.0}]
    peak = baseline
    max_dd = 0.0
    wins = losses = 0
    gross_win = gross_loss = 0.0
    best = worst = 0.0
    for t in closed:
        equity += t.pnl or 0.0
        ts = (t.closed_at or t.opened_at).isoformat()
        curve.append({"ts": ts, "equity": round(equity, 2), "pnl": round(t.pnl or 0.0, 2)})
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        if (t.pnl or 0.0) >= 0:
            wins += 1
            gross_win += t.pnl or 0.0
        else:
            losses += 1
            gross_loss += abs(t.pnl or 0.0)
        best = max(best, t.pnl or 0.0)
        worst = min(worst, t.pnl or 0.0)
    total = wins + losses
    return {
        "curve": curve,
        "stats": {
            "baseline": round(baseline, 2),
            "equity": round(equity, 2),
            "total_closed": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100.0, 2) if total else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
            "gross_win": round(gross_win, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(gross_win / wins, 2) if wins else 0.0,
            "avg_loss": round(gross_loss / losses, 2) if losses else 0.0,
            "best": round(best, 2),
            "worst": round(worst, 2),
            "max_drawdown_pct": round(max_dd, 2),
        },
    }


@router.get("/dashboard")
def dashboard(session: Session = Depends(get_session)):
    from app.services.copier import CopierEngine

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

    # Pre-fetch EVERY account's trades in one query and group in memory — the
    # previous per-account ``select(Trade)`` loop was N+1 over the accounts.
    enabled_ids = [a.id for a in accounts if a.enabled]
    trades_by_account: dict[str, list[Trade]] = {}
    if enabled_ids:
        for t in session.exec(select(Trade).where(Trade.account_id.in_(enabled_ids))).all():
            trades_by_account.setdefault(t.account_id, []).append(t)

    rows = [
        _account_performance(session, copier, a, is_master=(a.id == master_id),
                             trades_by_account=trades_by_account)
        for a in accounts
        if a.enabled
    ]
    master = next((r for r in rows if r["is_master"]), None)
    slaves = [r for r in rows if not r["is_master"]]

    performance = None
    if master is not None and master_id is not None:
        baseline = master.get("funded") if master.get("funded") is not None else 0.0
        # Reuse the pre-fetched, pre-grouped trades — no second query.
        trades = trades_by_account.get(master_id, [])
        performance = _performance_curve(trades, baseline)

    return {
        "master": master,
        "slaves": slaves,
        "total_accounts": len(rows),
        "master_id": master_id,
        "performance": performance,
    }


# ---------- Assets ----------

@router.get("/assets")
def list_assets(session: Session = Depends(get_session)) -> list[Asset]:
    return session.exec(select(Asset)).all()


@router.post("/assets")
def create_asset(payload: Asset, session: Session = Depends(get_session)) -> Asset:
    session.add(payload)
    session.commit()
    session.refresh(payload)
    return payload


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: str, session: Session = Depends(get_session)):
    asset = session.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(404, "asset not found")
    session.delete(asset)
    session.commit()
    return {"ok": True}


# ---------- Strategies + RAG ----------

class StrategyPayload(BaseModel):
    id: str | None = None
    name: str
    concept_markdown: str = ""
    body_markdown: str
    asset_class: str = ""
    timeframe: str = ""
    status: str = "active"


class StrategyUpdatePayload(BaseModel):
    name: str | None = None
    concept_markdown: str | None = None
    body_markdown: str | None = None
    asset_class: str | None = None
    timeframe: str | None = None
    status: str | None = None
    qa_notes: str | None = None


class StrPreparationPayload(BaseModel):
    name: str = ""
    concept_markdown: str = ""
    body_markdown: str
    asset_class: str = ""
    timeframe: str = ""


@router.get("/strategies")
def list_strategies(session: Session = Depends(get_session)) -> list[Strategy]:
    return session.exec(select(Strategy)).all()


@router.post("/strategies")
def create_strategy(payload: StrategyPayload, session: Session = Depends(get_session)) -> Strategy:
    strategy = Strategy(
        id=payload.id or str(uuid.uuid4()),
        name=payload.name,
        concept_markdown=payload.concept_markdown,
        body_markdown=payload.body_markdown,
        asset_class=payload.asset_class,
        timeframe=payload.timeframe,
        status=payload.status,
    )
    session.add(strategy)
    session.commit()
    RagService(session).index_strategy(strategy)
    session.refresh(strategy)
    return strategy


class ReferencePayload(BaseModel):
    name: str
    concept_markdown: str = ""
    notes: str = ""


@router.post("/strategies/reference")
def save_reference(payload: ReferencePayload, session: Session = Depends(get_session)):
    """Save a trading concept as a REFERENCE document in the strategy library.

    References are concept-only docs (not yet backtestable) that the Synthesizer
    Agent combines into a machine-actionable entry. Saved with status ``reference``
    and indexed into RAG so later syntheses can compile across the whole library.
    """
    from app.services.strategy_pipeline import _reference_sources

    if not (payload.concept_markdown or "").strip():
        raise HTTPException(422, "A trading concept is required for a reference document.")

    ref = Strategy(
        id=str(uuid.uuid4()),
        name=payload.name,
        concept_markdown=payload.concept_markdown,
        body_markdown="",  # concept-only; the entry is BUILT by synthesis
        status="reference",
    )
    session.add(ref)
    session.commit()
    RagService(session).index_strategy(ref)
    session.refresh(ref)
    return {
        "strategy": ref.model_dump(mode="json"),
        "reference_count": len(_reference_sources(exclude_id=ref.id)) + 1,
    }


@router.post("/strategies/qa")
def prepare_strategy(payload: StrPreparationPayload):
    """Run the merged QA gate (deterministic + LLM self-check) without saving."""
    from app.agents.synthesizer import SynthesizerAgent

    return SynthesizerAgent().review_entry(
        name=payload.name, concept=payload.concept_markdown,
        markdown=payload.body_markdown,
    )


class LTRPayload(BaseModel):
    limit: int = 10


@router.post("/strategies/ltr")
def run_ltr(payload: LTRPayload, session: Session = Depends(get_session)):
    """LoseTradeReview: review the most recent unreviewed losing trades.

    Writes a journal of error per loss and returns the lessons block that is fed
    to the Synthesizer Agent as learning context."""
    from app.agents.ltr import LTRAgent

    return LTRAgent(session).review(limit=payload.limit)


@router.get("/strategies/ltr/records")
def ltr_records(session: Session = Depends(get_session)):
    """Reference the losing-trade review database (read-only).

    Backend reference for other agents / retraining: each entry pairs the raw
    loss (why it went wrong) with the correction (how it should have been done).
    """
    from app.models.entities import LTRRecord

    rows = session.exec(
        select(LTRRecord).order_by(LTRRecord.reviewed_at.desc()).limit(200)
    ).all()
    return [r.model_dump() for r in rows]


class StrategySubmitPayload(BaseModel):
    name: str
    concept_markdown: str = ""  # general trading concept (ingested as context, REQUIRED input)
    body_markdown: str = ""  # synthesized entry criteria; manual entry textbox retired (#8)
    asset_class: str = ""
    # Strategy is asset-agnostic by design. `symbol`/`timeframe` are optional
    # backtest hints (the markdown states which timeframes it trades); the pipeline
    # defaults to a representative basket when absent.
    symbol: str | None = None
    timeframe: str | None = None
    bars: int = 6000
    max_iterations: int = 6
    min_winrate: float = 55.0
    min_sharpe: float = 1.0
    strategy_id: str | None = None


@router.post("/strategies/submit")
def submit_strategy(payload: StrategySubmitPayload):
    """Auto pipeline: QA -> parameterize -> backtest/refine to best -> save.

    Returns a job_id to poll via GET /strategies/submit/{job_id}. Kept asynchronous
    because QA + refinement involves several LLM calls and backtests."""
    from app.services.strategy_pipeline import start

    job_id = start(payload.model_dump())
    return {"job_id": job_id, "ok": True}


@router.get("/strategies/submit/{job_id}")
def submit_status(job_id: str):
    from app.services.strategy_pipeline import status

    return status(job_id) or {"status": "unknown", "message": "job not found"}


@router.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: str, payload: StrategyUpdatePayload,
                    session: Session = Depends(get_session)) -> Strategy:
    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")
    changes = payload.model_dump(exclude_unset=True)
    for field, value in changes.items():
        if value is not None:
            setattr(strat, field, value)
    session.add(strat)
    session.commit()
    RagService(session).index_strategy(strat)  # re-embed on body change
    session.refresh(strat)
    return strat


@router.post("/strategies/{strategy_id}/approve")
def approve_strategy(strategy_id: str, session: Session = Depends(get_session)) -> Strategy:
    """Promote a DRAFT strategy to ACTIVE and apply its best refined params."""
    from app.services.strategy_pipeline import _metrics_green, _metrics_overfit_guard

    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")

    best = session.exec(
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy_id, BacktestRun.accepted.is_(1))
        .order_by(BacktestRun.created_at.desc())
    ).first()
    if best is not None:
        # Faithful code runs store {code, confluence}; apply both so the strategy
        # keeps its executable rules + the scanner's confluence params.
        try:
            parsed_spec = json.loads(best.spec_json)
        except (TypeError, ValueError):
            parsed_spec = {}
        code = parsed_spec.get("code") if isinstance(parsed_spec, dict) else None
        confluence = (
            parsed_spec.get("confluence")
            if isinstance(parsed_spec, dict) and parsed_spec.get("confluence")
            else parsed_spec
        )
        if code:
            strat.strategy_code = code
        strat.params_json = json.dumps(confluence or {})

    # Approval gate: if the strategy has a best spec (went through the pipeline),
    # all metrics must be green (anti-overfit) before it can go live. Strategies
    # without a backtest are allowed through as plain rule-only setups.
    if strat.params_json:
        gate = _metrics_overfit_guard({
            "train": json.loads(best.metrics_json or "{}").get("train", {}) if best else {},
            "test": json.loads(best.metrics_json or "{}").get("test", {}) if best else {},
        })
        if not _metrics_green(gate):
            raise HTTPException(409, "Not all metrics are green (possible overfit). Refine again.")

    strat.status = "active"
    session.add(strat)
    session.commit()
    RagService(session).index_strategy(strat)
    session.refresh(strat)
    return strat


@router.post("/strategies/{strategy_id}/review")
def review_strategy(strategy_id: str, session: Session = Depends(get_session)):
    """Run the merged QA gate (deterministic + LLM self-check) on an existing strategy."""
    from app.agents.synthesizer import SynthesizerAgent

    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")
    return SynthesizerAgent().review_entry(
        name=strat.name, concept=strat.concept_markdown,
        markdown=strat.body_markdown,
    )


@router.delete("/strategies/{strategy_id}")
def delete_strategy(strategy_id: str, session: Session = Depends(get_session)):
    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")
    RagService(session).delete_strategy(strategy_id)
    session.delete(strat)
    session.commit()
    return {"ok": True}


@router.post("/strategies/search")
def search_strategies(query: str, top_k: int = 3, session: Session = Depends(get_session)):
    return {"results": RagService(session).search(query, top_k)}


# ---------- IR primitives (new-tool proposal loop) ----------

@router.get("/strategies/primitives")
def list_primitives():
    """Current active mathematical primitive registry (names, descriptions,
    parameters) — the tools the Synthesizer's IR layer may reference."""
    from app.services.primitive_proposals import registry_snapshot

    return {"primitives": registry_snapshot()}


class PrimitiveRegisterPayload(BaseModel):
    primitives: list[dict] = []


@router.post("/strategies/primitives/register")
def register_primitives(payload: PrimitiveRegisterPayload):
    """Approve new-tool proposals: validate them deterministically, write the
    canonical registry functions into ir_primitives.py and live-reload the
    registry (hot-swappable primitives — the schema menu + compiler vocabulary
    update instantly for the running process)."""
    from app.services.primitive_proposals import register_primitives as _register

    try:
        return _register(payload.primitives)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a failed registry write is a client error
        raise HTTPException(400, f"Could not register primitives: {exc}") from exc


# ---------- Backtest + AI refinement ----------

def _stored_spec(strat: Strategy) -> dict | None:
    """Best refined spec applied to the strategy, if any."""
    if not getattr(strat, "params_json", ""):
        return None
    try:
        spec = json.loads(strat.params_json)
        return spec if isinstance(spec, dict) and spec else None
    except (TypeError, ValueError):
        return None


class BacktestPayload(BaseModel):
    symbol: str
    timeframe: str = "15m"
    bars: int = 1500
    spec: dict | None = None
    start_date: str | None = None  # exact window (extracted dataset / live range)
    end_date: str | None = None


@router.post("/strategies/{strategy_id}/backtest")
def run_backtest(strategy_id: str, payload: BacktestPayload,
                 session: Session = Depends(get_session)):
    from app.agents.strategy_codegen import StrategyCodeGen
    from app.services.backtest_engine import BacktestEngine

    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")
    try:
        if not getattr(strat, "strategy_code", ""):
            gen = StrategyCodeGen().generate(
                strat.name, strat.concept_markdown or "", strat.body_markdown,
                payload.symbol, payload.timeframe,
            )
            strat.strategy_code = gen.get("code") or ""
            session.add(strat)
            session.commit()
            session.refresh(strat)
        confluence = _stored_spec(strat) or {}
        return BacktestEngine().run_code(
            payload.symbol, payload.timeframe, strat.strategy_code,
            payload.bars, confluence=confluence,
            start_date=payload.start_date, end_date=payload.end_date,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class RefinePayload(BaseModel):
    symbol: str
    timeframe: str = "15m"
    bars: int = 1500
    max_iterations: int = 6
    min_winrate: float = 50.0
    min_sharpe: float = 1.0
    start_date: str | None = None
    end_date: str | None = None


@router.post("/strategies/{strategy_id}/backtest/refine")
def refine_backtest(strategy_id: str, payload: RefinePayload,
                    session: Session = Depends(get_session)):
    from app.agents.strategy_codegen import StrategyCodeGen
    from app.services.backtest_engine import BacktestEngine

    strat = session.get(Strategy, strategy_id)
    if strat is None:
        raise HTTPException(404, "strategy not found")
    if payload.max_iterations < 1:
        raise HTTPException(400, "max_iterations must be >= 1")
    try:
        if not getattr(strat, "strategy_code", ""):
            gen = StrategyCodeGen().generate(
                strat.name, strat.concept_markdown or "", strat.body_markdown,
                payload.symbol, payload.timeframe,
            )
            strat.strategy_code = gen.get("code") or ""
            session.add(strat)
            session.commit()
            session.refresh(strat)
        return BacktestEngine().refine_code_loop(
            session, strat, payload.symbol, payload.timeframe,
            strat.strategy_code, payload.bars, payload.max_iterations,
            payload.min_winrate, payload.min_sharpe,
            confluence=_stored_spec(strat) or {},
            concept=strat.concept_markdown or "",
            start_date=payload.start_date, end_date=payload.end_date,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/strategies/{strategy_id}/backtests")
def list_backtests(strategy_id: str, limit: int = 100,
                   session: Session = Depends(get_session)):
    if session.get(Strategy, strategy_id) is None:
        raise HTTPException(404, "strategy not found")
    rows = session.exec(
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy_id)
        .order_by(BacktestRun.created_at.desc())
        .limit(max(1, min(int(limit), 500)))
    ).all()
    return [
        {
            "id": r.id,
            "iteration": r.iteration,
            "symbol": r.symbol,
            "timeframe": r.timeframe,
            "spec": json.loads(r.spec_json or "{}"),
            "metrics": json.loads(r.metrics_json or "{}"),
            "rationale": r.rationale,
            "accepted": bool(r.accepted),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


# ---------- Scans / Validations / Trades ----------

@router.get("/scans")
def list_scans(limit: int = 200, session: Session = Depends(get_session)) -> list[Scan]:
    return session.exec(
        select(Scan).order_by(Scan.timestamp.desc()).limit(max(1, min(int(limit), 2000)))
    ).all()


@router.get("/validations")
def list_validations(session: Session = Depends(get_session)) -> list[Validation]:
    return session.exec(select(Validation).order_by(Validation.id)).all()


@router.get("/trades")
def list_trades(open_only: bool = False, limit: int = 500,
                session: Session = Depends(get_session)) -> list[Trade]:
    q = select(Trade).order_by(Trade.opened_at.desc())
    if open_only:
        q = q.where(Trade.closed_at.is_(None))
    return session.exec(q.limit(max(1, min(int(limit), 5000)))).all()


@router.post("/trades")
def create_trade(payload: TradePayload, session: Session = Depends(get_session)) -> Trade:
    acc = session.get(Account, payload.account_id)
    if acc is None:
        raise HTTPException(404, "account not found")
    decision = RiskInterceptor(session).evaluate(payload.account_id)
    if payload.mode == "auto" and not decision.allowed:
        raise HTTPException(403, decision.reason)

    trade = Trade(
        id=str(uuid.uuid4()),
        account_id=payload.account_id,
        entry_price=payload.entry_price,
        sl=payload.sl,
        tp1=payload.tp1,
        tp2=payload.tp2,
        tp3=payload.tp3,
        tp4=payload.tp4,
        direction=payload.direction,
        lot_size=payload.lot_size,
        mode=payload.mode,
        symbol=payload.symbol,
        broker=acc.broker,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


@router.put("/trades/{trade_id}/close")
def close_trade(trade_id: str, payload: dict | None = None,
                session: Session = Depends(get_session)):
    trade = session.get(Trade, trade_id)
    if trade is None:
        raise HTTPException(404, "trade not found")
    payload = payload or {}
    from app.services.position_manager import PositionManager

    PositionManager(session).close(
        trade,
        reason=str(payload.get("reason") or "manual"),
        exit_price=payload.get("exit_price"),
    )
    return {"ok": True, "id": trade_id}


@router.delete("/trades/{trade_id}")
def delete_trade(trade_id: str, session: Session = Depends(get_session)):
    """Delete a trade record (e.g. a phantom row that never reached the broker).

    Only really-removed records should be deleted. A live position should be
    closed via the close endpoint instead (it keeps the audit trail).
    """
    trade = session.get(Trade, trade_id)
    if trade is None:
        raise HTTPException(404, "trade not found")
    session.delete(trade)
    session.commit()
    return {"ok": True, "id": trade_id}


@router.post("/trades/clear-phantoms")
def clear_phantom_trades(session: Session = Depends(get_session)):
    """Delete trade rows that never confirmed on a broker (phantom trade).

    A phantom / failed placement is a Trade row that was created before the
    broker-confirmation guard landed: it has no symbol, no broker order id and
    no broker. It is NOT a real live trade and can never be LTR-reviewed, so we
    remove it so the Trades tab only reflects really-executed positions.
    """
    rows = session.exec(
        select(Trade).where(
            Trade.symbol == "",
            (Trade.broker == "") | (Trade.broker.is_(None)),
            Trade.broker_order_id.is_(None),
            Trade.closed_at.is_not(None),
        )
    ).all()
    removed = []
    for t in rows:
        removed.append(t.id)
        session.delete(t)
    session.commit()
    return {"ok": True, "removed": len(removed), "ids": removed}


# ---------- Chart overlays (open positions) ----------

@router.get("/chart/positions")
def chart_positions(symbol: str | None = None, session: Session = Depends(get_session)):
    """Open trades as chart overlays: entry, live managed SL, TPs, direction.

    The managed SL is the non-mutating projection from ``PositionManager.peek_managed_sl``
    so the chart can render the trailing stop / breakeven line exactly as the
    position-manager rules would apply it, without writing to the DB.
    """
    from app.services.position_manager import PositionManager

    q = select(Trade).where(Trade.closed_at.is_(None))
    if symbol:
        q = q.where(Trade.symbol == symbol)
    trades = session.exec(q.order_by(Trade.opened_at.desc())).all()
    if not trades:
        return {"positions": [], "symbol": symbol}

    pm = PositionManager(session)
    prices = _chart_prices(session, trades)
    positions = []
    for trade in trades:
        current = prices.get(trade.symbol)
        if current is None:
            managed_sl = trade.sl
        else:
            managed_sl = pm.peek_managed_sl(trade, current)
        positions.append({
            "id": trade.id,
            "account_id": trade.account_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "sl": trade.sl,
            "managed_sl": managed_sl,
            "tp1": trade.tp1,
            "tp2": trade.tp2,
            "tp3": trade.tp3,
            "tp4": trade.tp4,
            "lot_size": trade.lot_size,
            "opened_at": trade.opened_at.isoformat(),
            "last_price": current,
        })
    return {"positions": positions, "symbol": symbol}


def _chart_prices(session: Session, trades: list[Trade]) -> dict:
    """Latest close per symbol among the open trades (best-effort, cached by MD)."""
    import asyncio

    from app.services.market_data import MarketDataService

    symbols = sorted({t.symbol or t.account_id for t in trades})
    prices: dict = {}

    async def _fetch_all():
        md = MarketDataService()
        try:
            for sym in symbols:
                try:
                    data = await md.fetch(symbol=sym, timeframe="1h", limit=1)
                    rows = data.get("candles") or []
                    if rows:
                        prices[sym] = float(rows[-1]["close"])
                except Exception:  # noqa: BLE001
                    continue
        finally:
            await md.close()

    try:
        asyncio.run(_fetch_all())
    except Exception:  # noqa: BLE001
        pass
    return prices


# ---------- Data cleanup ----------

class CleanupPayload(BaseModel):
    retention_days: int | None = None
    dry_run: bool = False


@router.post("/cleanup/run")
def run_cleanup(payload: CleanupPayload, session: Session = Depends(get_session)):
    from app.services.cleanup import DataCleanupService

    return DataCleanupService(session, retention_days=payload.retention_days).cleanup(
        dry_run=payload.dry_run
    )


# ---------- News ----------

@router.get("/news")
def news(
    limit: int = 40,
    symbol: str | None = None,
    session: Session = Depends(get_session),
):
    from app.services.news import NewsService

    watchlist = [a.symbol for a in session.exec(select(Asset)).all()]
    try:
        return {
            "items": NewsService().fetch(limit=limit, symbol=symbol, watchlist=watchlist),
            "watchlist": watchlist,
        }
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "watchlist": watchlist, "error": str(exc)}


# ---------- Market data (chart) ----------

@router.get("/regime")
def regime(session: Session = Depends(get_session)):
    """Latest Market Regime classification (refreshed every 5 min by scheduler)."""
    from app.services.regime import RegimeAgent, state

    cached = state()
    if cached.get("refreshed_at") is None:
        cached = RegimeAgent().refresh(session)
    return cached


@router.post("/regime/refresh")
def regime_refresh(session: Session = Depends(get_session)):
    from app.services.regime import RegimeAgent

    return RegimeAgent().refresh(session)


@router.get("/news-agent/state")
def news_agent_state(session: Session = Depends(get_session)):
    """Latest AI news-agent summaries (refreshed every 15m by scheduler)."""
    from app.services.news_agent import NewsAgent, state

    cached = state()
    if cached.get("refreshed_at") is None:
        cached = NewsAgent().refresh(session)
    return cached


@router.post("/news-agent/refresh")
def news_agent_refresh(session: Session = Depends(get_session)):
    from app.services.news_agent import NewsAgent

    return NewsAgent().refresh(session)


@router.get("/candles")
async def candles(symbol: str, timeframe: str = "15m", limit: int = 500,
                  session: Session = Depends(get_session)):
    from app.services.market_data import MarketDataService

    # Master account is the basis for chart data: prefer its broker provider.
    preferred = None
    accounts = session.exec(select(Account)).all()
    master = next((a for a in accounts if a.account_type == "master" and a.enabled), None)
    if master is not None:
        preferred = master.broker
    try:
        return await MarketDataService().fetch(symbol=symbol, timeframe=timeframe, limit=limit, preferred=preferred)
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "timeframe": timeframe, "candles": [], "provider": "", "error": str(exc)}


# ---------- Signals (core AI pipeline output) ----------

class ScanSignalPayload(BaseModel):
    symbol: str | None = None


class AutomationPayload(BaseModel):
    enabled: bool
    interval_seconds: int | None = None


def _signal_from_scan(session: Session, scan: Scan,
                      validation_by_scan: dict[str, Validation] | None = None,
                      executed_scan_ids: set[str] | None = None) -> dict:
    validation = (
        validation_by_scan.get(scan.id) if validation_by_scan is not None else session.exec(
            select(Validation).where(Validation.scan_id == scan.id)
        ).first()
    )
    candidate = json.loads(scan.scanner_output_json or "{}")
    asset = session.get(Asset, scan.asset_id) if scan.asset_id else None
    executed = scan.id in (executed_scan_ids or set()) if executed_scan_ids is not None else (
        session.exec(select(Trade).where(Trade.scan_id == scan.id)).first() is not None
    )
    return {
        "scan_id": scan.id,
        "asset": asset.symbol if asset else candidate.get("asset", "?"),
        "timestamp": scan.timestamp.isoformat(),
        "candidate": candidate,
        "has_trade": bool(candidate.get("direction")),
        "validation": {
            "verdict": validation.verdict,
            "rationale": validation.rationale,
            "levels": json.loads(validation.suggested_levels_json or "{}"),
            "provider": validation.provider_used,
        }
        if validation
        else None,
        "executed": executed,
    }


@router.get("/signals")
def list_signals(session: Session = Depends(get_session)):
    from app.services.scheduler import scan_state

    assets = session.exec(select(Asset)).all()
    if not assets:
        return {"signals": [], "automation": scan_state(),
                "auto_execution_enabled": SettingsService(session).get("auto_execution_enabled", False)}

    # One query for the LATEST scan per asset via a row_number window (was
    # N+1 ``order_by(...).first()`` per asset), then batched lookups for the
    # signals' validations + executed flags.
    from sqlalchemy import func as sa_func

    rn = sa_func.row_number().over(
        partition_by=Scan.asset_id, order_by=Scan.timestamp.desc())
    ranked = session.exec(
        select(Scan.id, rn.label("rn"))
        .where(Scan.asset_id.in_([a.id for a in assets]))
    ).all()
    scan_ids = [sid for sid, r in ranked if r == 1]
    scans = {s.id: s for s in session.exec(select(Scan).where(Scan.id.in_(scan_ids))).all()}

    validation_by_scan: dict[str, Validation] = {}
    if scan_ids:
        for v in session.exec(select(Validation).where(Validation.scan_id.in_(scan_ids))).all():
            validation_by_scan.setdefault(v.scan_id, v)
    executed_scan_ids: set[str] = set()
    if scan_ids:
        executed_scan_ids = {
            t.scan_id for t in session.exec(
                select(Trade.scan_id).where(Trade.scan_id.in_(scan_ids))
            ).all() if t.scan_id
        }

    out = [
        _signal_from_scan(session, s, validation_by_scan, executed_scan_ids)
        for s in scans.values()
        if s is not None
    ]
    out.sort(key=lambda s: s["timestamp"], reverse=True)
    return {
        "signals": out,
        "automation": scan_state(),
        "auto_execution_enabled": SettingsService(session).get("auto_execution_enabled", False),
    }


@router.post("/signals/scan")
def run_signal_scan(payload: ScanSignalPayload, session: Session = Depends(get_session)):
    from app.services.orchestrator import Orchestrator
    from app.services.scheduler import scan_state

    if payload.symbol:
        asset = session.exec(select(Asset).where(Asset.symbol == payload.symbol)).first()
        if asset is None:
            raise HTTPException(404, f"{payload.symbol} is not in the watchlist")
        runs: list[Asset] = [asset]
    else:
        runs = list(session.exec(select(Asset)).all())

    out: list[dict] = []
    orch = Orchestrator(session)
    for asset in runs:
        result = orch.run_scan(asset)
        scan = session.get(Scan, result["scan_id"])
        if scan is not None:
            out.append(_signal_from_scan(session, scan))
    out.sort(key=lambda s: s["timestamp"], reverse=True)
    return {"signals": out, "automation": scan_state()}


@router.get("/signals/automation")
def get_automation(session: Session = Depends(get_session)):
    from app.services.scheduler import scan_state

    svc = SettingsService(session)
    state = scan_state()
    state["auto_execution_enabled"] = svc.get("auto_execution_enabled", False)
    return state


@router.post("/signals/automation")
def set_automation(payload: AutomationPayload, session: Session = Depends(get_session)):
    from app.services.scheduler import set_scan_enabled, set_scan_interval, scan_state

    svc = SettingsService(session)
    svc.set("auto_scan_enabled", payload.enabled)
    if payload.interval_seconds and payload.interval_seconds >= 10:
        svc.set("scan_interval_seconds", payload.interval_seconds)
        set_scan_interval(payload.interval_seconds)
    set_scan_enabled(payload.enabled)
    return scan_state()


@router.post("/signals/{scan_id}/accept")
def accept_signal(scan_id: str, session: Session = Depends(get_session)):
    from app.services.orchestrator import Orchestrator

    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(404, "scan not found")
    validation = session.exec(
        select(Validation).where(Validation.scan_id == scan_id)
    ).first()
    if validation is None or validation.verdict != "ACCEPTABLE":
        raise HTTPException(400, "signal is not actionable (no ACCEPTABLE validation)")
    if session.exec(select(Trade).where(Trade.scan_id == scan_id)).first() is not None:
        raise HTTPException(409, "signal already executed")
    return Orchestrator(session).execute_signal(scan, validation, mode="manual")


# ---------- Risk ----------

class RiskClearancePayload(BaseModel):
    account_id: str | None = None
    symbol: str = ""
    direction: str = "BUY"
    entry_price: float = 0.0
    sl: float = 0.0


@router.get("/risk/check")
def risk_check(session: Session = Depends(get_session)):
    return RiskInterceptor(session).evaluate().to_dict()


@router.get("/risk/plan")
def get_risk_plan(symbol: str | None = None, session: Session = Depends(get_session)):
    from app.services.risk_manager import RiskManager

    return RiskManager(session).plan(symbol=symbol)


@router.post("/risk/clear")
def risk_clearance(payload: RiskClearancePayload, session: Session = Depends(get_session)):
    from app.services.risk_manager import RiskManager

    return RiskManager(session).clearance(
        account_id=payload.account_id,
        symbol=payload.symbol,
        direction=payload.direction,
        entry_price=payload.entry_price,
        sl=payload.sl,
    )


# ---------- Datasets (Phase 8 Parquet Extraction) ----------

class DatasetExtractPayload(BaseModel):
    # ``None`` = the whole watchlist: every asset in the watchlist is extracted
    # automatically, one Parquet dataset per symbol.
    symbol: str | None = None
    timeframe: str = "15m"
    start_date: str | None = None  # YYYY-MM-DD (exact window, not a bar count)
    end_date: str | None = None
    asset_class: str = ""


@router.get("/datasets")
def list_datasets():
    from app.services.dataset_service import DatasetService

    return {"datasets": DatasetService().list_datasets()}


@router.post("/datasets/extract")
def extract_dataset(payload: DatasetExtractPayload,
                    session: Session = Depends(get_session)):
    """Extract historical OHLCV to Parquet for an EXACT date range.

    Without a ``symbol`` the extraction runs across the whole watchlist
    (every Asset row), returning ``{datasets, errors}`` so the UI can show
    per-symbol results."""
    from app.services.dataset_service import DatasetService

    try:
        svc = DatasetService()
        if payload.symbol and payload.symbol.strip():
            meta = svc.extract_dataset(
                symbol=payload.symbol,
                timeframe=payload.timeframe,
                start_date=payload.start_date,
                end_date=payload.end_date,
                asset_class=payload.asset_class,
            )
            return {"datasets": [meta], "errors": []}
        symbols = [a.symbol for a in session.exec(select(Asset)).all()]
        return svc.extract_watchlist(
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
            asset_class=payload.asset_class,
            symbols=symbols,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - extraction failure is a client error
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/{dataset_id}")
def preview_dataset(dataset_id: str, limit: int = 50):
    from app.services.dataset_service import DatasetService

    try:
        return DatasetService().preview_dataset(dataset_id, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"dataset {dataset_id} not found") from exc


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    from app.services.dataset_service import DatasetService

    ok = DatasetService().delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(404, f"dataset {dataset_id} not found")
    return {"ok": True, "id": dataset_id}


# ---------- ML Scanner — XGBoost (Phase 9) ----------

@router.get("/ml/status")
def ml_status():
    from app.services.ml_scanner import status as ml_status_fn

    return ml_status_fn()


@router.post("/ml/train")
def ml_train(session: Session = Depends(get_session)):
    """Trigger a (blocking) retrain of the XGBoost scanner model.

    Only trains on strategies that passed the full QA gate (active status,
    accepted BacktestRun). Returns train/test accuracy and metadata.
    """
    import threading

    from app.services import ml_scanner

    result_holder: dict = {}
    lock = threading.Event()

    def _bg():
        result_holder["result"] = ml_scanner.train(session)
        lock.set()

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    lock.wait(timeout=300)  # max 5 minutes for training
    result = result_holder.get("result", {"ok": False, "error": "Training timed out"})
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Training failed"))
    return result


@router.delete("/ml/model")
def ml_reset():
    """Delete the persisted XGBoost model (forces a retrain before next use)."""
    from app.services.ml_scanner import MODEL_PATH

    deleted = False
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        deleted = True
    return {"ok": True, "deleted": deleted}
