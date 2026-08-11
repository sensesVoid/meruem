import json
import uuid

from sqlmodel import Session, select

from app.agents.scanner import ScannerAgent
from app.agents.validator import ValidationFailed, ValidatorChain
from app.core.db import engine
from app.models.entities import Account, Scan, Strategy, Validation
from app.services.copier import CopierEngine
from app.services.risk_interceptor import RiskInterceptor
from app.services.settings_service import SettingsService


class Orchestrator:
    """Drives Watchlist -> Scanner -> RAG -> Validator -> Risk -> Copier."""

    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)
        self.scanner = ScannerAgent()
        self.validator_chain = ValidatorChain()


    def run_scan(self, asset: object) -> dict:
        ohlcv = self._market_ohlcv(asset.symbol, asset.asset_class)
        strategies = self._active_strategies(asset)
        references = self._reference_docs(asset)

        # Stop-scan risk gate (#13 Risk Management Agent): when risk guardrails
        # or max concurrent auto-trades are breached during auto-execution,
        # short-circuit the scan so LLM calls are not wasted on unexecutable trades.
        from app.services.risk_manager import RiskManager

        stop_reason = RiskManager(self.session).stop_scan_reason()
        if stop_reason:
            candidate = {
                "asset": asset.symbol,
                "asset_class": asset.asset_class,
                "timeframe": ohlcv.get("timeframe"),
                "direction": None,
                "confidence": 0.0,
                "rationale": f"Scan stopped by Risk Manager: {stop_reason}",
                "risk_blocked": True,
            }
            scan = Scan(
                id=str(uuid.uuid4()),
                asset_id=asset.id,
                scanner_output_json=json.dumps(candidate),
                strategy_refs=json.dumps([]),
            )
            self.session.add(scan)
            self.session.commit()
            return {"scan_id": scan.id, "candidate": candidate, "validation": None,
                    "risk_blocked": True, "stop_reason": stop_reason}

        # News trade-block gate (#13 adhoc): when the news agent has flagged
        # high-impact news for THIS asset, never scan for new setups - the baby
        # users queue a manual re-scan once the event has passed.
        block_reason = self._news_trade_block(asset.symbol)
        if block_reason:
            candidate = {
                "asset": asset.symbol,
                "asset_class": asset.asset_class,
                "timeframe": ohlcv.get("timeframe"),
                "direction": None,
                "confidence": 0.0,
                "rationale": f"Trade blocked by high-impact news: {block_reason}",
                "news_blocked": True,
            }
            scan = Scan(
                id=str(uuid.uuid4()),
                asset_id=asset.id,
                scanner_output_json=json.dumps(candidate),
                strategy_refs=json.dumps([]),
            )
            self.session.add(scan)
            self.session.commit()
            return {"scan_id": scan.id, "candidate": candidate, "validation": None,
                    "news_blocked": True}

        indicators = self._compute_indicators(ohlcv, strategies)
        from app.services.ml_artifact import context_for as ml_context_for

        candidate = self.scanner.scan(
            ohlcv, indicators, [self._rule_pack(s) for s in strategies],
            references=references,
            ml_context=ml_context_for(asset.symbol))
        candidate = self.scanner.normalize(candidate)

        # Persist concrete references to the approved strategies the scan used
        # (id + name), so refs are always resolvable, never free-form text.
        refs = [
            {"id": s.id, "name": s.name}
            for s in strategies
            if s.name in (candidate.get("strategy_refs") or [])
        ]
        if not refs:
            refs = [{"id": s.id, "name": s.name} for s in strategies]

        scan = Scan(
            id=str(uuid.uuid4()),
            asset_id=asset.id,
            scanner_output_json=json.dumps(candidate),
            strategy_refs=json.dumps(refs),
        )
        self.session.add(scan)
        self.session.commit()

        if not candidate.get("direction"):
            return {"scan_id": scan.id, "candidate": candidate, "validation": None}

        return self._validate(scan, candidate)

    def _active_strategies(self, asset: object) -> list[Strategy]:
        """All approved (active) strategies. The scanner reasons over every
        approved strategy's rules + proxy params for the scanned asset."""
        return self.session.exec(
            select(Strategy).where(Strategy.status == "active")
        ).all()

    def _reference_docs(self, asset: object | None = None,
                        symbol: str | None = None) -> list[dict]:
        """The saved REFERENCE library (status ``reference``) as a SECOND reference.

        References are concept-only documents (name + concept_markdown) that are
        not themselves executable — they feed the synthesizer and, per #13 adhoc,
        are ALSO cross-referenced by the scanner + Master Trader beside the active
        synthesized strategy. They are a knowledge source, never an execution rule.
        """
        rows = self.session.exec(
            select(Strategy).where(Strategy.status == "reference")
        ).all()
        result = []
        for s in rows:
            concept = (s.concept_markdown or "").strip()
            if concept:
                result.append({
                    "id": s.id,
                    "name": s.name,
                    "concept": concept,
                })
        return result

    def _rule_pack(self, strategy: Strategy) -> dict:
        """Structured rules + proxy params for one approved strategy."""
        params: dict = {}
        if strategy.params_json:
            try:
                parsed = json.loads(strategy.params_json)
                if isinstance(parsed, dict):
                    params = parsed
            except (TypeError, ValueError):
                params = {}
        return {
            "id": strategy.id,
            "name": strategy.name,
            "params": params,
            "concept": strategy.concept_markdown or "",
            "rules": strategy.body_markdown,
        }

    def _compute_indicators(self, ohlcv: dict, strategies: list[Strategy]) -> dict:
        """Fast vector math over the last bars; the scanner then reasons from real
        numbers instead of fabricating them."""
        rows = ohlcv.get("rows") or []
        if not rows:
            return {}
        closes = [float(r["close"]) for r in rows]
        highs = [float(r["high"]) for r in rows]
        lows = [float(r["low"]) for r in rows]
        out: dict = {
            "last_close": closes[-1],
            "last_high": highs[-1],
            "last_low": lows[-1],
        }
        if len(closes) >= 40:
            out["sma_fast"] = round(sum(closes[-10:]) / 10, 5)
            out["sma_slow"] = round(sum(closes[-40:]) / 40, 5)
        out["atr"] = self._atr(highs, lows, closes, 14)
        out["rsi"] = self._rsi(closes, 14)
        for s in strategies[:3]:
            if not s.params_json:
                continue
            try:
                p = json.loads(s.params_json)
            except (TypeError, ValueError):
                continue
            tag = f"{s.name}"
            out[f"{tag}_sma_fast"] = self._sma(closes, int(p.get("fast", 10)))
            out[f"{tag}_sma_slow"] = self._sma(closes, int(p.get("slow", 40)))
            out[f"{tag}_rsi"] = self._rsi(closes, int(p.get("rsi_period", 14)))
            out[f"{tag}_atr"] = self._atr(highs, lows, closes, int(p.get("atr_period", 14)))
        return out

    @staticmethod
    def _sma(values: list[float], period: int) -> float | None:
        if len(values) < period or period <= 0:
            return None
        return round(sum(values[-period:]) / period, 5)

    @staticmethod
    def _rsi(values: list[float], period: int) -> float | None:
        if len(values) < period + 1 or period <= 0:
            return None
        deltas = [b - a for a, b in zip(values[-(period + 1):], values[-(period + 1):][1:])]
        gains = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        if losses == 0:
            return 100.0
        rs = gains / losses
        return round(100 - 100 / (1 + rs), 2)

    @staticmethod
    def _atr(highs: list[float], lows: list[float], closes: list[float],
             period: int) -> float | None:
        if len(closes) < period + 1 or period <= 0:
            return None
        trs = []
        for i in range(1, len(closes)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        return round(sum(trs[-period:]) / period, 5)

    @staticmethod
    def _regime_context(symbol: str) -> str:
        """Snapshot of the latest regime cache for a symbol, as validator context."""
        try:
            from app.services.regime import state as regime_state

            cached = regime_state()
            assets = cached.get("assets") or {}
            asset = assets.get(symbol)
            overall = cached.get("overall")
            text = f"overall_market={overall}"
            if asset:
                text += (f", trend={asset.get('trend_label')}, adx={asset.get('trend_strength')}, "
                         f"volatility={asset.get('volatility_regime')}, rsi={asset.get('rsi')}, "
                         f"momentum={asset.get('momentum_label')}")
            return text
        except Exception:  # noqa: BLE001
            return "regime=unknown"

    @staticmethod
    def _news_context(symbol: str) -> str:
        """Latest AI news summary + fundamental bias + trade-block signal for a
        symbol, as context."""
        try:
            from app.services.news_agent import state as news_state

            cached = news_state()
            asset = (cached.get("assets") or {}).get(symbol)
            if not asset:
                return "news=no_summary"
            block = "news_trade_block=ON " + (asset.get("block_reason") or "") if asset.get("trade_block") else "news_trade_block=OFF"
            return (f"{block}; "
                    f"news_bias={asset.get('fundamental_bias')}; "
                    f"summary={asset.get('summary', '')[:400]}")
        except Exception:  # noqa: BLE001
            return "news=unknown"

    @staticmethod
    def _news_trade_block(symbol: str) -> str:
        """High-impact news reason for a symbol, resolved against the LIVE clock
        and the configured before/after window. Returns "" when not blocked now."""
        try:
            from app.services.news_agent import active_block

            status = active_block(symbol)
            if status.get("trade_block"):
                return str(status.get("block_reason") or "high-impact news")
        except Exception:  # noqa: BLE001, S110 - best-effort context read
            pass
        return ""

    @staticmethod
    def _market_ohlcv(symbol: str, asset_class: str, limit: int = 60) -> dict:
        """Pull real OHLCV for the scan; fall back to empty rows on any failure."""
        import asyncio

        timeframe = "15m"
        try:
            from app.services.market_data import MarketDataService

            async def _fetch() -> dict:
                md = MarketDataService()
                try:
                    return await md.fetch(symbol=symbol, timeframe=timeframe, limit=limit)
                finally:
                    await md.close()

            data = asyncio.run(_fetch())
            rows = data.get("candles") or []
            return {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "rows": rows,
            }
        except Exception:  # noqa: BLE001
            return {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "rows": [],
            }

    def _validate(self, scan: Scan, candidate: dict) -> dict:
        refs = json.loads(scan.strategy_refs or "[]")
        rules = []
        for ref in refs:
            strat = self.session.get(Strategy, ref.get("id") or "")
            if strat is not None:
                rules.append(self._rule_pack(strat))
        candidate_symbol = str(candidate.get("asset") or candidate.get("symbol") or "")
        references = self._reference_docs(symbol=candidate_symbol)
        try:
            result, provider = self.validator_chain.validate(
                candidate,
                {
                    "market": "context",
                    "strategies": rules,
                    "references": references,
                    "regime": self._regime_context(candidate_symbol),
                    "news": self._news_context(candidate_symbol),
                })
        except ValidationFailed:
            self._pause_llm_limit(scan)
            return {
                "scan_id": scan.id,
                "candidate": candidate,
                "validation": {"verdict": "PAUSED_LLM_LIMIT"},
            }

        validation = Validation(
            id=str(uuid.uuid4()),
            scan_id=scan.id,
            verdict=result.get("verdict", "NOT_ACCEPTABLE"),
            rationale=result.get("rationale", ""),
            suggested_levels_json=json.dumps(result.get("suggested_levels", {})),
            provider_used=provider,
        )
        self.session.add(validation)
        self.session.commit()

        out = {"scan_id": scan.id, "candidate": candidate,
               "validation": validation.model_dump()}
        if validation.verdict == "ACCEPTABLE" and self.settings.get("auto_execution_enabled"):
            out["execution"] = self._execute(scan, candidate, validation)
        return out

    def _pause_llm_limit(self, scan: Scan) -> None:
        self.settings.set("system_status", "PAUSED_LLM_LIMIT")
        self.session.exec(select(Scan)).first()  # touch session
        print("ALL validation providers failed: system -> PAUSED_LLM_LIMIT")

    def _execute(self, scan: Scan, candidate: dict, validation: Validation) -> dict:
        return self.execute_signal(scan, validation, mode="auto")

    def execute_signal(self, scan: Scan, validation: Validation, mode: str = "manual") -> dict:
        """Execute an accepted signal as a real trade (master + slave fan-out).

        ``mode="auto"`` requires auto-execution to be enabled; ``mode="manual"``
        is an explicit human action and does not. No order is ever fanned out when
        the auto gate is off on the auto path — this is strictly enforced here and
        is not repeated blindly by the copier.
        """
        if mode == "auto" and not self.settings.get("auto_execution_enabled"):
            return {"status": "blocked", "reason": "auto-execution is disabled"}

        risk = RiskInterceptor(self.session)
        decision = risk.evaluate(require_auto=(mode == "auto"))
        if not decision.allowed:
            self.settings.set("system_status", "PAUSED_RISK" if mode == "auto" else self.settings.get("system_status"))
            return {"status": "blocked", "reason": decision.reason}

        levels = json.loads(validation.suggested_levels_json or "{}")
        master = self._select_master()
        if master is None:
            return {"status": "error", "reason": "no master account configured"}

        candidate = json.loads(scan.scanner_output_json or "{}")
        direction = candidate.get("direction") or "BUY"
        symbol = candidate.get("asset") or ""
        entry_price = levels.get("entry", self._entry_from_candidate(candidate))
        stop_loss = levels.get("sl", candidate.get("invalidation") or 0)
        tp = {
            "tp1": levels.get("tp1"), "tp2": levels.get("tp2"),
            "tp3": levels.get("tp3"), "tp4": levels.get("tp4"),
        }
        sl_dist_pips = abs(entry_price - stop_loss)

        copier = CopierEngine(self.session)
        # Live/known master equity (falls back to configured default).
        master_equity = copier._account_equity(master)
        master_lot = self._master_lot(candidate, levels, master_equity)
        plans = copier.fan_out(
            master_account=master,
            master_lot=master_lot,
            master_equity=master_equity,
            sl_distance_pips=sl_dist_pips,
            pip_value=10.0,
            entry_price=entry_price,
            sl=stop_loss,
            tp=tp,
            direction=direction,
            symbol=symbol,
            scan_id=scan.id,
            mode=mode,
        )
        return {"status": "executed", "plans": [vars(p) for p in plans]}

    @staticmethod
    def _entry_from_candidate(candidate: dict) -> float:
        """Resolve a single entry price from a candidate's entry_zone without
        crashing when the zone is missing/malformed."""
        try:
            zone = candidate.get("entry_zone") or []
            if zone:
                return float(zone[0])
        except (TypeError, ValueError, IndexError):
            pass
        return candidate.get("invalidation") or 0.0

    def _select_master(self) -> Account | None:
        """The user-selected master account, falling back to the first master."""
        selected = self.settings.get("selected_master_account_id", "")
        if selected:
            acc = self.session.get(Account, selected)
            if acc is not None and acc.account_type == "master" and acc.enabled:
                return acc
        return self.session.exec(
            select(Account).where(
                Account.account_type == "master", Account.enabled.is_(True)
            )
        ).first()

    def _master_lot(self, candidate: dict, levels: dict, equity: float) -> float:
        cfg = self.settings.get_all()
        mode = cfg.get("master_lot_sizing_mode", "risk_pct")
        if mode == "manual":
            return float(cfg.get("master_manual_lot", 0.10))
        if equity <= 0:
            return 0.0
        risk_pct = float(cfg.get("master_risk_pct", 1.0)) / 100.0
        entry = levels.get("entry") or self._entry_from_candidate(candidate)
        sl = levels.get("sl") or candidate.get("invalidation") or 0
        risk_dist = abs(entry - sl)
        if risk_dist <= 0:
            return 0.0
        return round(equity * risk_pct / (risk_dist * 10.0), 2)


def scan_all_assets() -> dict:
    """Entry point used by scheduler / CLI.

    Scans **every** asset in the watchlist. Per-asset failures are swallowed so
    one bad symbol never aborts the cycle and starves the rest of the watchlist.
    """
    with Session(engine) as session:
        from app.models.entities import Asset

        assets = session.exec(select(Asset)).all()
        orch = Orchestrator(session)
        results = {}
        for asset in assets:
            try:
                results[asset.symbol] = orch.run_scan(asset)
            except Exception as exc:  # noqa: BLE001 - keep scanning the rest
                print(f"[scan_all_assets] {asset.symbol} failed: {exc}")
                results[asset.symbol] = {"asset": asset.symbol, "error": str(exc)}
        return results