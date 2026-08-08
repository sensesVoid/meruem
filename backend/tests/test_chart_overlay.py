"""Tests for #11: chart overlays of open positions + managed (trailing) SL."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.entities import Account, Trade
from app.services.position_manager import PositionManager


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _mk_account(session, account_id="acc-master-1"):
    acc = Account(
        id=account_id, broker="MT5", account_type="master",
        nickname="Master", risk_scaling_mode="proportional", scale_factor=1.0,
        encrypted_ref="mt5:default", enabled=True,
    )
    session.add(acc)
    session.commit()
    return acc


def _mk_trade(session, account_id, direction="BUY", entry=2000.0, sl=1990.0,
              tp1=2010.0, symbol="XAUUSD", closed=None):
    trade = Trade(
        id=f"trd-{uuid4hex()}",
        account_id=account_id,
        entry_price=entry,
        sl=sl,
        tp1=tp1,
        direction=direction,
        lot_size=0.1,
        mode="manual",
        symbol=symbol,
        closed_at=closed,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def uuid4hex():
    import uuid
    return str(uuid.uuid4())


def test_peek_managed_sl_breakeven_when_tp1_hit(session):
    _mk_account(session)
    trade = _mk_trade(session, "acc-master-1")
    pm = PositionManager(session)
    # Price is at/above TP1 -> auto-breakeven moves SL to entry.
    managed = pm.peek_managed_sl(trade, 2010.0)
    assert managed == trade.entry_price
    # Non-mutating: DB still holds the original SL.
    session.refresh(trade)
    assert trade.sl == 1990.0


def test_peek_managed_sl_trailing_above_breakeven(session):
    _mk_account(session)
    trade = _mk_trade(session, "acc-master-1", entry=100.0, sl=98.0, tp1=110.0)
    pm = PositionManager(session)
    # Price ran to 120; activation = entry + 1.5*risk(2) = 103 -> active.
    # step = 10 -> projected SL = 120 - 10 = 110 (above entry, so it sticks).
    managed = pm.peek_managed_sl(trade, 120.0)
    assert managed == 110.0


def test_peek_managed_sl_no_change_before_activation(session):
    _mk_account(session)
    trade = _mk_trade(session, "acc-master-1", entry=100.0, sl=98.0, tp1=110.0)
    pm = PositionManager(session)
    managed = pm.peek_managed_sl(trade, 101.0)
    assert managed == 98.0


def test_peek_managed_sl_seLL_direction_not_trailed(session):
    _mk_account(session)
    trade = _mk_trade(session, "acc-master-1", direction="SELL", entry=100.0, sl=102.0, tp1=98.0)
    pm = PositionManager(session)
    # SELL: price 96 <= tp1 (98) triggers auto-breakeven to entry (100);
    # trailing is BUY-only, so there is no step-ratchet below 100.
    managed = pm.peek_managed_sl(trade, 96.0)
    assert managed == trade.entry_price
    assert managed != trade.sl


def test_peek_managed_sl_closed_trade_returns_sl(session):
    _mk_account(session)
    trade = _mk_trade(session, "acc-master-1", closed=datetime.now(timezone.utc))
    pm = PositionManager(session)
    assert pm.peek_managed_sl(trade, 99999.0) == trade.sl


def test_chart_positions_returns_open_trades(client):
    client.post("/api/accounts", json={
        "id": "acc-master-1", "broker": "MT5", "account_type": "master",
        "nickname": "Master", "risk_scaling_mode": "proportional",
        "scale_factor": 1.0, "encrypted_ref": "mt5:default",
    })
    client.post("/api/trades", json={
        "account_id": "acc-master-1", "direction": "BUY",
        "entry_price": 2000.0, "sl": 1990.0, "tp1": 2010.0, "lot_size": 0.1,
        "mode": "manual", "symbol": "XAUUSD",
    })
    client.post("/api/trades", json={
        "account_id": "acc-master-1", "direction": "BUY",
        "entry_price": 3000.0, "sl": 2990.0, "lot_size": 0.1,
        "mode": "manual", "symbol": "BTCUSD",
    })
    # Without a symbol filter the endpoint returns every open trade; the chart
    # maps them to the currently charted asset client-side.
    r = client.get("/api/chart/positions")
    assert r.status_code == 200
    assert len(r.json()["positions"]) == 2
    pos = next(p for p in r.json()["positions"] if p["entry_price"] == 2000.0)
    assert pos["direction"] == "BUY"
    assert pos["symbol"] == "XAUUSD"
    assert pos["sl"] == 1990.0
    assert pos["managed_sl"] >= pos["sl"]
    assert pos["tp1"] == 2010.0

    # The chart's actual request: filter by the charted symbol — the XAUUSD
    # trade must be returned and the BTCUSD trade must NOT leak into it.
    r = client.get("/api/chart/positions", params={"symbol": "XAUUSD"})
    assert r.status_code == 200
    positions = r.json()["positions"]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "XAUUSD"
    assert positions[0]["entry_price"] == 2000.0


def test_chart_positions_specific_symbol(session, client):
    _mk_account(session)
    _mk_trade(session, "acc-master-1", symbol="XAUUSD")
    _mk_trade(session, "acc-master-1", symbol="BTCUSD")
    r = client.get("/api/chart/positions", params={"symbol": "BTCUSD"})
    assert r.status_code == 200
    assert len(r.json()["positions"]) == 1
    assert r.json()["positions"][0]["symbol"] == "BTCUSD"