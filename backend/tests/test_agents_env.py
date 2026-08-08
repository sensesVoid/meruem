"""Tests for the Market Regime Agent and the News Agent."""

from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.news_agent import (
    _block_events,
    _detect_trade_block,
    _parse,
    _parse_published,
    _resolve_block,
)
from app.services.regime import (
    RegimeAgent,
    _adx,
    _breadth,
    _garch_vol,
    _markov_regime,
    _rsi,
    analyze_asset,
    state,
)


def _ohlcv_rows(closes, highs=None, lows=None):
    highs = highs or [c * 1.005 for c in closes]
    lows = lows or [c * 0.995 for c in closes]
    return [
        {"open": c, "high": h, "low": l, "close": c, "volume": 0.0}
        for c, h, l in zip(closes, highs, lows)
    ]


def test_rsi_overbought():
    values = [100.0 + i for i in range(20)]
    assert _rsi(values, 14) > 70


def test_rsi_oversold():
    values = [100.0 - i for i in range(20)]
    assert _rsi(values, 14) < 30


def test_adx_trend_is_bounded():
    up = [float(i) for i in range(40)]
    rows = _ohlcv_rows(up)
    adx = _adx([r["close"] for r in rows], [r["high"] for r in rows], [r["low"] for r in rows])
    assert 0.0 <= adx <= 100.0


def test_analyze_asset_insufficient_bars():
    read = analyze_asset("EURUSD", _ohlcv_rows([1.0, 1.1, 1.2]))
    assert read["trend_label"] == "RANGING"
    assert "insufficient" in read["note"]


def test_analyze_asset_uptrend_classified_strong_up():
    up = [1.0 + i * 0.0002 for i in range(120)]
    rows = _ohlcv_rows(up)
    read = analyze_asset("EURUSD", rows)
    assert read["trend_direction"] == 1


def test_breadth_mixed():
    reads = [
        {"trend_direction": 1}, {"trend_direction": -1},
        {"trend_direction": 1}, {"trend_direction": -1},
    ]
    assert _breadth(reads)["label"] == "MIXED"


def test_breadth_broad_up():
    reads = [
        {"trend_direction": 1}, {"trend_direction": 1},
        {"trend_direction": -1},
    ]
    assert _breadth(reads)["label"] == "BROAD-UP"


def test_state_returns_snapshot_without_mutation():
    from app.services.regime import _STATE
    _STATE["overall"] = "BULLISH_REGIME"
    snap = state()
    snap["overall"] = "MUTATED"
    assert state()["overall"] == "BULLISH_REGIME"


def test_refresh_handles_empty_watchlist(session):
    result = RegimeAgent().refresh(session)
    assert result["overall"] in (
        "BULLISH_REGIME", "BEARISH_REGIME", "MIXED_REGIME"
    )
    assert result["breadth"] or result["error"] is None


def test_news_parse_extracts_fields(monkeypatch):
    raw = (
        '{"summary": "Rates steady in the FOMC.", '
        '"fundamental_bias": "NEUTRAL", '
        '"key_events": ["FOMC hold"], '
        '"precision_notes": "Dot plot unchanged."}'
    )
    parsed = _parse(raw)
    assert parsed["fundamental_bias"] == "NEUTRAL"
    assert parsed["key_events"] == ["FOMC hold"]


def test_news_parse_tolerates_bad_json():
    parsed = _parse("not json at all")
    assert parsed["summary"] == ""


def test_trade_block_no_impact_no_block():
    items = [{"title": "Gold consolidates near key resistance", "summary": "rangebound session"}]
    block = _detect_trade_block(items)
    assert block["trade_block"] is False


def test_trade_block_fomc_flagged():
    items = [{"title": "FOMC rate decision due this week", "summary": "markets eye the fed"}]
    block = _detect_trade_block(items)
    assert block["trade_block"] is True
    assert "FOMC" in block["block_reason"].upper()


def test_trade_block_nfp_flag_and_case_insensitive():
    items = [{"title": "Nonfarm payrolls stun markets", "summary": ""}]
    block = _detect_trade_block(items)
    assert block["trade_block"] is True
    assert "NONFARM PAYROLL" in block["block_reason"].upper()


def test_trade_block_matches_summary_field():
    items = [{"title": "Daily wrap", "summary": "CPI inflation data came in hot."}]
    block = _detect_trade_block(items)
    assert block["trade_block"] is True


def test_parse_published_zulu_and_naive():
    dt = _parse_published("2026-08-08T12:00:00Z")
    assert dt is not None and dt.tzinfo is not None
    dt2 = _parse_published("2026-08-08T12:00:00")
    assert dt2 is not None and dt2.tzinfo is not None
    assert _parse_published("") is None
    assert _parse_published("not a date") is None


def test_block_events_extracts_terms():
    items = [{"title": "FOMC decision tonight", "summary": "", "published": "2026-08-08T12:00:00Z"}]
    events = _block_events(items)
    assert len(events) == 1
    assert "FOMC" in events[0]["terms"] or "fomc" in [t.lower() for t in events[0]["terms"]]


def test_resolve_block_active_within_window():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    event = {"published_at": now, "terms": ["fomc"]}
    assert _resolve_block([event], now=now)["trade_block"] is True


def test_resolve_block_expires_after_window():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    # Event published 5h ago -> outside default before(30)/after(45) window.
    old = {"published_at": now - timedelta(hours=5), "terms": ["cpi"]}
    assert _resolve_block([old], now=now)["trade_block"] is False


def test_resolve_block_inactive_before_window():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    future = {"published_at": now + timedelta(hours=2), "terms": ["cpi"]}
    assert _resolve_block([future], now=now)["trade_block"] is False


def test_resolve_block_anchors_missing_timestamp_to_now():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    block = {"published_at": None, "terms": ["cpi"]}
    assert _resolve_block([block], now=now)["trade_block"] is True


def test_resolve_block_reports_window_bounds():
    now = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    event = {"published_at": now, "terms": ["fomc"]}
    res = _resolve_block([event], now=now)
    assert res["block_start"] is not None
    assert res["block_end"] is not None
    assert res["block_start"] < res["block_end"]


# ---------------------------------------------------------------------------
# Phase 4 — GARCH + Markov quant layers
# ---------------------------------------------------------------------------

def _switching_closes(seed, calm_sigma, vol_sigma, calm_bars=100, vol_bars=100):
    rng = np.random.default_rng(seed)
    rets = np.concatenate([
        rng.normal(0.0, calm_sigma, calm_bars),
        rng.normal(0.0, vol_sigma, vol_bars),
    ])
    return np.cumprod(1.0 + rets).tolist()


def test_garch_vol_requires_enough_bars():
    g = _garch_vol(_switching_closes(1, 0.001, 0.02, calm_bars=20, vol_bars=20))
    assert g["note"] == "insufficient"


def test_garch_vol_ratio_expands_on_vol_shock():
    closes = _switching_closes(2, 0.001, 0.03)
    g = _garch_vol(closes)
    assert g["vol_ratio"] > 1.0
    assert g["alpha"] > 0.0 and g["beta"] > 0.0
    assert 0.0 < g["cond_vol"] < 1.0


def test_garch_vol_fit_is_deterministic():
    closes = _switching_closes(3, 0.001, 0.02)
    assert _garch_vol(closes) == _garch_vol(closes)


def test_markov_detects_calm_to_volatile_switch():
    closes = _switching_closes(4, 0.001, 0.03)
    m = _markov_regime(closes)
    assert m["state"] == "TURBULENT"
    assert m["prob_high"] >= 0.5
    assert m["vol_turb"] > m["vol_calm"]


def test_markov_reports_calm_on_quiet_series():
    rng = np.random.default_rng(5)
    closes = np.cumprod(1.0 + rng.normal(0.0, 0.001, 220)).tolist()
    m = _markov_regime(closes)
    # No regime separation -> separation guard keeps it CALM (prob_high < 0.5).
    assert m["state"] == "CALM"
    assert m["prob_high"] < 0.5


def test_markov_requires_enough_bars():
    m = _markov_regime(_switching_closes(6, 0.001, 0.02, calm_bars=30, vol_bars=30))
    assert m["note"] == "insufficient"


def test_analyze_asset_embeds_quant_layers():
    closes = _switching_closes(7, 0.001, 0.03)
    rows = _ohlcv_rows(closes)
    read = analyze_asset("XAUUSD", rows)
    assert read["garch"] is not None
    assert read["garch_label"] in ("GARCH_STABLE", "GARCH_EXPANDING", "GARCH_COMPRESSING")
    assert read["markov"] is not None
    assert read["markov"]["state"] in ("CALM", "TURBULENT")
    assert read["trend_label"] in ("UP", "DOWN", "RANGING", "STRONG_UP", "STRONG_DOWN")


def test_analyze_asset_insufficient_bars_has_null_quant():
    read = analyze_asset("EURUSD", _ohlcv_rows([1.0, 1.1, 1.2]))
    assert read["garch"] is None
    assert read["markov"] is None