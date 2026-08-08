"""Phase: strategy utilization — scanner gets only approved rules + refined params,
concrete strategy refs persisted, indicators computed with real numbers."""

import json
from unittest.mock import patch

import pytest

from app.models.entities import Asset, Scan, Strategy, Validation


def _strategy(session, name, asset_class="forex", status="active", params=None):
    s = Strategy(
        id=f"s-{name.lower().replace(' ', '-')}", name=name,
        body_markdown=f"Rules for {name}: buy SMA fast > slow with RSI filter.",
        asset_class=asset_class, timeframe="15m", status=status,
        params_json=json.dumps(params) if params else "",
    )
    session.add(s)
    session.commit()
    session.refresh(s)
    return s


def _reference(session, name, concept="Sweep liquidity then reversion to FVG."):
    r = Strategy(
        id=f"r-{name.lower().replace(' ', '-')}", name=name,
        body_markdown="", concept_markdown=concept, status="reference",
    )
    session.add(r)
    session.commit()
    session.refresh(r)
    return r


@pytest.fixture
def asset(session):
    a = Asset(id="a-eurusd", symbol="EURUSD", asset_class="forex")
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def test_rule_pack_includes_refined_params(session):
    s = _strategy(session, "Mom", params={"fast": 8, "slow": 34, "rsi_period": 11})
    from app.services.orchestrator import Orchestrator

    pack = Orchestrator(session)._rule_pack(s)
    assert pack["params"]["fast"] == 8
    assert pack["id"] == s.id
    assert "Rules for Mom" in pack["rules"]


def test_active_strategies_matches_exact_class_only(session):
    keep = _strategy(session, "ForexMom", "forex")
    _strategy(session, "CryptoMom", "cripto", status="draft")
    from app.services.orchestrator import Orchestrator

    matched = Orchestrator(session)._active_strategies(type("A", (), {
        "asset_class": "forex", "timeframe_watchlist": "15m"})())
    names = [s.name for s in matched]
    assert "ForexMom" in names
    assert "CryptoMom" not in names
    assert keep.status == "active"


def test_run_scan_persists_concrete_refs(session, asset):
    _strategy(session, "EUR Trend", "forex", params={"fast": 8, "slow": 34})
    from sqlmodel import select

    from app.services.orchestrator import Orchestrator

    fake_candidate = {
        "asset": "EURUSD", "timeframe": "15m", "asset_class": "forex",
        "direction": "BUY", "entry_zone": [1.08, 1.081], "invalidation": 1.075,
        "strategy_refs": ["EUR Trend"], "confidence": 0.7,
        "rationale": "trend + RSI",
    }
    orch = Orchestrator(session)
    with patch.object(Orchestrator, "_market_ohlcv",
                      return_value={"symbol": "EURUSD", "asset_class": "forex",
                                    "timeframe": "15m", "rows": []}), \
         patch.object(Orchestrator, "_validate", return_value={"verdict": "ACCEPTABLE"}), \
         patch.object(orch, "scanner") as mock_scanner:
        mock_scanner.scan.return_value = dict(fake_candidate)
        mock_scanner.normalize.side_effect = lambda c: c
        orch.run_scan(asset)

    strategies_arg = mock_scanner.scan.call_args[0][2]
    assert strategies_arg[0]["name"] == "EUR Trend"
    assert strategies_arg[0]["params"]["fast"] == 8

    scan = session.exec(select(Scan)).first()
    refs = json.loads(scan.strategy_refs)
    assert refs == [{"id": strategies_arg[0]["id"], "name": "EUR Trend"}]


def test_execute_signal_is_blocked_when_auto_disabled(session):
    """Auto-mode execution must never fan out when auto-execution is off."""
    from app.services.orchestrator import Orchestrator

    scan = Scan(id="s1", asset_id=None, scanner_output_json=json.dumps({
        "asset": "EURUSD", "direction": "BUY", "entry_zone": [1.08], "invalidation": 1.07}))
    validation = Validation(id="v1", scan_id="s1", verdict="ACCEPTABLE",
                            rationale="ok", suggested_levels_json="{}", provider_used="test")
    session.add(scan); session.add(validation)
    session.commit()
    orch = Orchestrator(session)
    res = orch.execute_signal(scan, validation, mode="auto")
    assert res["status"] == "blocked"
    assert "auto-execution" in res["reason"]


def test_execute_signal_guards_malformed_candidate(session):
    """A candidate with no direction / entry_zone must not raise 500 and must
    still yield a safe plan instead of crashing."""
    from app.models.entities import Account
    from app.services.orchestrator import Orchestrator

    session.add(Account(id="a-m", broker="MT5", account_type="master", nickname="m",
                        risk_scaling_mode="proportional", encrypted_ref="mt5:m",
                        login="m", balance=10000.0))
    session.commit()
    from app.services.settings_service import SettingsService
    SettingsService(session).set("auto_execution_enabled", True)
    SettingsService(session).set("selected_master_account_id", "a-m")

    scan = Scan(id="s2", asset_id=None, scanner_output_json=json.dumps({
        "asset": "BTCUSD", "direction": None, "entry_zone": None, "invalidation": None,
        "strategy_refs": []}))
    validation = Validation(id="v2", scan_id="s2", verdict="ACCEPTABLE",
                            rationale="ok", suggested_levels_json=json.dumps({}),
                            provider_used="test")
    session.add(scan); session.add(validation)
    session.commit()
    orch = Orchestrator(session)
    from app.services.copier import CopierEngine
    with patch.object(CopierEngine, "_place", return_value=False):
        res = orch.execute_signal(scan, validation, mode="manual")
    assert res["status"] == "executed"
    assert res["plans"][0]["account_type"] == "master"
    from app.services.orchestrator import Orchestrator

    rows = []
    for i in range(60):
        base = 100 + i
        rows.append({"open": base, "high": base + 1, "low": base - 1,
                     "close": base, "volume": 100})
    ohlcv = {"symbol": "X", "asset_class": "forex", "timeframe": "15m", "rows": rows}
    ind = Orchestrator(session)._compute_indicators(ohlcv, [])
    assert ind["sma_fast"] is not None
    assert ind["sma_slow"] is not None
    assert ind["rsi"] == 100.0  # strictly rising -> RSI 100


def test_run_scan_news_trade_block_short_circuits(session, asset):
    """#13 adhoc: when the news agent flags high-impact news for the asset, the
    scan must be blocked (no direction, no validation, no scanner call)."""
    from datetime import datetime, timezone

    from sqlmodel import select

    from app.services.news_agent import _STATE as NEWS_STATE
    from app.services.orchestrator import Orchestrator

    try:
        now = datetime.now(timezone.utc)
        NEWS_STATE["assets"] = {
            "EURUSD": {
                "block_events": [{
                    "published": now.isoformat(),
                    "published_at": now,
                    "terms": ["fomc"],
                }],
            },
        }
        orch = Orchestrator(session)
        with patch.object(Orchestrator, "_market_ohlcv",
                          return_value={"symbol": "EURUSD", "asset_class": "forex",
                                        "timeframe": "15m", "rows": []}), \
             patch.object(orch, "scanner") as mock_scanner:
            mock_scanner.scan.return_value = {"asset": "EURUSD", "direction": "BUY"}
            out = orch.run_scan(asset)

        assert out["news_blocked"] is True
        assert out["candidate"]["direction"] is None
        assert out["validation"] is None
        mock_scanner.scan.assert_not_called()

        scan = session.exec(select(Scan)).first()
        scan = session.exec(select(Scan)).first()
        assert scan is not None
        loaded = json.loads(scan.scanner_output_json)
        assert loaded["news_blocked"] is True
        assert "FOMC rate decision" in loaded["rationale"] or "high-impact" in loaded["rationale"].lower()
    finally:
        NEWS_STATE["assets"] = {}


def test_reference_docs_returns_only_concept_docs(session):
    _reference(session, "Sweep")
    _strategy(session, "Active", status="active")
    from app.services.orchestrator import Orchestrator

    docs = Orchestrator(session)._reference_docs()
    assert len(docs) == 1
    assert docs[0]["name"] == "Sweep"
    assert "Sweep liquidity" in docs[0]["concept"]


def test_reference_docs_excludes_empty_concepts(session):
    _reference(session, "Sweep", concept="  ")
    _reference(session, "Orderblock", concept="OB retest then displacement.")
    from app.services.orchestrator import Orchestrator

    docs = Orchestrator(session)._reference_docs()
    assert [d["name"] for d in docs] == ["Orderblock"]


def test_run_scan_passes_references_to_scanner(session, asset):
    from app.services.orchestrator import Orchestrator

    _reference(session, "Sweep", "Liquidity sweep then displacement.")
    _strategy(session, "EUR Breakout", "forex", params={"fast": 8, "slow": 34})

    orch = Orchestrator(session)
    with patch.object(Orchestrator, "_market_ohlcv",
                      return_value={"symbol": "EURUSD", "asset_class": "forex",
                                    "timeframe": "15m", "rows": []}), \
         patch.object(Orchestrator, "_validate", return_value={"verdict": "ACCEPTABLE"}), \
         patch.object(orch, "scanner") as mock_scanner:
        mock_scanner.scan.return_value = {"asset": "EURUSD", "direction": None}
        mock_scanner.normalize.side_effect = lambda c: c
        orch.run_scan(asset)

    refs_arg = mock_scanner.scan.call_args.kwargs.get("references") or mock_scanner.scan.call_args[0][3]
    assert any(r["name"] == "Sweep" for r in refs_arg)


def test_validate_context_includes_references(session, asset):
    import json as _json
    import uuid

    from sqlmodel import select

    from app.models.entities import Scan, Strategy
    from app.services.orchestrator import Orchestrator

    _reference(session, "Sweep", "Liquidity sweep then displacement.")
    _strategy(session, "EUR Breakout", "forex", params={"fast": 8, "slow": 34})

    strategy = session.exec(select(Strategy).where(Strategy.status == "active")).first()
    scan = Scan(
        id=str(uuid.uuid4()),
        asset_id=asset.id,
        scanner_output_json=_json.dumps({"asset": "EURUSD", "direction": "BUY"}),
        strategy_refs=_json.dumps([{"id": strategy.id, "name": strategy.name}]),
    )
    session.add(scan)
    session.commit()

    orch = Orchestrator(session)
    captured = {}

    class _FakeValidator:
        def validate(self, candidate, context):
            captured["context"] = context
            return ({"verdict": "ACCEPTABLE"}, "fake")

    orch.validator_chain = _FakeValidator()
    orch._validate(scan, {"asset": "EURUSD", "asset_class": "forex", "direction": "BUY"})

    names = [r["name"] for r in captured["context"].get("references", [])]
    assert names == ["Sweep"]