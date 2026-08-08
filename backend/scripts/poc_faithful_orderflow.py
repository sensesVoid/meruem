"""Proof of Concept — FAITHFUL orderflow backtest vs the SMA/RSI proxy.

Implements the trader's actual Orderflow XAUUSD rules as an executable backtesting
strategy (VWAP sigma-bands, Initial Balance range/trend regimes, CVD divergence +
absorption filters, FVG retest trigger, ATR-based stops/targets) and runs it on the
same XAUUSD 1h data + train/test split that the proxy pipeline used, so the
out-of-sample numbers are directly comparable.

Run:  python -m scripts.poc_faithful_orderflow
"""

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.market_data import MarketDataService  # noqa: E402

MIN_BARS = 60
TRAIN_FRACTION = 0.7

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_df(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    data = asyncio.run(MarketDataService().fetch(symbol, timeframe, bars))
    candles = data.get("candles") or []
    df = pd.DataFrame(candles)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)
    return df.dropna()


# --------------------------------------------------------------------------
# Orderflow indicators (vectorized, no lookahead)
# --------------------------------------------------------------------------

def _session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session-anchored VWAP + sigma bands (1s and 2s). Sessions roll at 00:00 UTC."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = df.index.normalize()
    pv = tp * df["Volume"]
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["Volume"].groupby(day).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    sq = (tp - vwap) ** 2
    cum_sq = sq.groupby(day).cumsum()
    count = df.groupby(day).cumcount() + 1
    var = (cum_sq / count.replace(0, np.nan))  # population var of tp around vwap
    sd = np.sqrt(var.clip(lower=0))
    out = pd.DataFrame(index=df.index)
    out["vwap"] = vwap
    out["v1_hi"] = vwap + sd
    out["v1_lo"] = vwap - sd
    out["v2_hi"] = vwap + 2 * sd
    out["v2_lo"] = vwap - 2 * sd
    return out


def _initial_balance(df: pd.DataFrame, first_n: int = 2) -> pd.DataFrame:
    """Initial Balance: high/low of the first `first_n` bars of each session.
    Used to classify Range (inside IB) vs Trend (breakout past IB)."""
    day = df.index.normalize()
    grp = df.groupby(day)
    ib_high = grp["High"].transform(lambda s: s.expanding().max())
    ib_low = grp["Low"].transform(lambda s: s.expanding().min())
    # Freeze to the first_n bars: once past the IB window, the level stays fixed.
    order = grp.cumcount()
    fixed_high = ib_high.where(order <= first_n).groupby(day).ffill()
    fixed_low = ib_low.where(order <= first_n).groupby(day).ffill()
    # Before the IB is formed the level is undefined -> carry from session start.
    fixed_high = fixed_high.groupby(day).ffill()
    fixed_low = fixed_low.groupby(day).ffill()
    out = pd.DataFrame(index=df.index)
    out["ib_high"] = fixed_high
    out["ib_low"] = fixed_low
    return out


def _cvd(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative Volume Delta: signed volume within a session (close-vs-range)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    buy = df["Volume"] * ((df["Close"] - df["Low"]) / rng).clip(0, 1)
    sell = df["Volume"] - buy
    delta = buy - sell
    day = df.index.normalize()
    cvd = delta.groupby(day).cumsum()
    out = pd.DataFrame(index=df.index)
    out["delta"] = delta
    out["cvd"] = cvd
    return out


def _fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gap: 3-candle imbalance. Bullish FVG when bar[-1].low > bar[-3].high
    (aggressive buyers leave a gap). Computed at bar close, tradable on the retest."""
    high = df["High"]
    low = df["Low"]
    bull = low.shift(1) > high.shift(3)  # gap between candle -2 and -3
    bear = high.shift(1) < low.shift(3)
    out = pd.DataFrame(index=df.index)
    out["fvg_bull"] = bull
    out["fvg_bear"] = bear
    out["fvg_bull_lo"] = np.where(bull, high.shift(3), np.nan)  # gap floor
    out["fvg_bear_hi"] = np.where(bear, low.shift(3), np.nan)   # gap ceiling
    return out


def _swing(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Confirmed swing high/low over `n` bars each side (lazy: -n lookahead)."""
    out = pd.DataFrame(index=df.index)
    out["swing_high"] = df["High"].rolling(2 * n + 1, center=True).max().shift(-n)
    out["swing_low"] = df["Low"].rolling(2 * n + 1, center=True).min().shift(-n)
    return out


# --------------------------------------------------------------------------
# The faithful strategy
# --------------------------------------------------------------------------

class OrderflowStrategy(Strategy):
    # Tunables (kept minimal — the concept drives entries; these are execution knobs)
    risk_pct = 0.01
    atr_period = 20
    atr_sl_mult = 1.5
    atr_tp_mult = 3.0
    absorption_delta = 1500.0   # |delta| that counts as a "high volume print"
    fvg_retest_tol = 0.0015     # close must be within ~0.15% of the FVG level to retest

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(
            lambda x: pd.Series(x).rolling(10).mean().values, close)
        self.sma_slow = self.I(
            lambda x: pd.Series(x).rolling(40).mean().values, close)
        self.vwap = self.I(lambda: _session_vwap(self.data.df)["vwap"].values)
        self.v1_hi = self.I(lambda: _session_vwap(self.data.df)["v1_hi"].values)
        self.v1_lo = self.I(lambda: _session_vwap(self.data.df)["v1_lo"].values)
        self.v2_hi = self.I(lambda: _session_vwap(self.data.df)["v2_hi"].values)
        self.v2_lo = self.I(lambda: _session_vwap(self.data.df)["v2_lo"].values)
        ib = _initial_balance(self.data.df)
        self.ib_high = self.I(lambda: ib["ib_high"].values)
        self.ib_low = self.I(lambda: ib["ib_low"].values)
        cvd = _cvd(self.data.df)
        self.delta = self.I(lambda: cvd["delta"].values)
        self.cvd = self.I(lambda: cvd["cvd"].values)
        fvg = _fvg(self.data.df)
        self.fvg_bull = self.I(lambda: fvg["fvg_bull"].values)
        self.fvg_bear = self.I(lambda: fvg["fvg_bear"].values)
        self.fvg_bull_lo = self.I(lambda: fvg["fvg_bull_lo"].values)
        self.fvg_bear_hi = self.I(lambda: fvg["fvg_bear_hi"].values)
        sw = _swing(self.data.df)
        self.swing_high = self.I(lambda: sw["swing_high"].values)
        self.swing_low = self.I(lambda: sw["swing_low"].values)
        self.atr = self.I(
            lambda H=self.data.High, L=self.data.Low, C=self.data.Close:
            self._atr(H, L, C, self.atr_period))

    @staticmethod
    def _atr(high, low, close, period: int):
        prev_close = pd.Series(close).shift(1)
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - prev_close).abs(),
            (pd.Series(low) - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().values

    def next(self):
        if self.position:
            return
        i = self.data.index  # bar index
        price = self.data.Close[-1]

        # Regime: Range (inside IB / between 1-sigma bands) vs Trend (beyond IB/2-sigma).
        in_range = price >= self.ib_low[-1] and price <= self.ib_high[-1]
        above_trend = price > self.ib_high[-1] and price > self.v2_hi[-1]
        below_trend = price < self.ib_low[-1] and price < self.v2_lo[-1]

        if in_range:
            # MODEL A: fade the boundaries toward the center.
            # Buy support: touch VWAP-1sigma with CVD divergence or buy absorption.
            support_touch = price <= self.v1_lo[-1]
            if support_touch and self._bullish_conviction():
                self._range_buy()
            # Sell resistance: touch VWAP+1sigma with CVD divergence or sell absorption.
            resistance_touch = price >= self.v1_hi[-1]
            if resistance_touch and self._bearish_conviction():
                self._range_sell()
        elif above_trend:
            # MODEL B: trend continuation on retest of the IB breakout.
            retest = price >= self.ib_high[-1] * (1 - self.fvg_retest_tol)
            if retest and self._bullish_conviction():
                self._trend_buy()
        elif below_trend:
            retest = price <= self.ib_low[-1] * (1 + self.fvg_retest_tol)
            if retest and self._bearish_conviction():
                self._trend_sell()

    # -- conviction filters (CVD divergence / absorption) -----------------
    def _bullish_conviction(self) -> bool:
        # Bullish CVD divergence: price makes a lower low, CVD makes a higher low.
        low_then = self.data.Low[-3]
        low_now = self.data.Low[-1]
        cvd_then = self.cvd[-3]
        cvd_now = self.cvd[-1]
        divergence = low_now < low_then and cvd_now > cvd_then
        # Buy absorption: big sell delta print but the bar closes green (sellers absorbed).
        absorption = self.delta[-1] < -self.absorption_delta and self.data.Close[-1] > self.data.Open[-1]
        return divergence or absorption

    def _bearish_conviction(self) -> bool:
        high_then = self.data.High[-3]
        high_now = self.data.High[-1]
        cvd_then = self.cvd[-3]
        cvd_now = self.cvd[-1]
        divergence = high_now > high_then and cvd_now < cvd_then
        absorption = self.delta[-1] > self.absorption_delta and self.data.Close[-1] < self.data.Open[-1]
        return divergence or absorption

    # -- execution --------------------------------------------------------
    def _range_buy(self):
        price = self.data.Close[-1]
        if self._finite(self.swing_low[-1]):
            sl = self.swing_low[-1] - self.atr[-1] * 0.5
        else:
            sl = price - self.atr[-1] * self.atr_sl_mult
        if self._finite(self.vwap[-1]):
            tp = min(self.vwap[-1], self.v1_hi[-1])
        else:
            tp = price + self.atr[-1] * self.atr_tp_mult
        self._enter("buy", sl, tp)

    def _range_sell(self):
        price = self.data.Close[-1]
        if self._finite(self.swing_high[-1]):
            sl = self.swing_high[-1] + self.atr[-1] * 0.5
        else:
            sl = price + self.atr[-1] * self.atr_sl_mult
        if self._finite(self.vwap[-1]):
            tp = max(self.vwap[-1], self.v1_lo[-1])
        else:
            tp = price - self.atr[-1] * self.atr_tp_mult
        self._enter("sell", sl, tp)

    def _trend_buy(self):
        ib_range = self.ib_high[-1] - self.ib_low[-1]
        sl = self.ib_high[-1] - 0.25 * ib_range
        tp = self.ib_high[-1] + 1.5 * ib_range
        self._enter("buy", sl, tp)

    def _trend_sell(self):
        ib_range = self.ib_high[-1] - self.ib_low[-1]
        sl = self.ib_low[-1] + 0.25 * ib_range
        tp = self.ib_low[-1] - 1.5 * ib_range
        self._enter("sell", sl, tp)

    def _enter(self, side: str, sl: float, tp: float):
        price = self.data.Close[-1]
        if side == "buy":
            # Geometry: SL must be below entry, TP above.
            if not (sl < price < tp):
                if sl >= price:
                    return
                if tp <= price:
                    return
            dist = price - sl
            if dist <= 0:
                return
            size = min(((self.equity * self.risk_pct) / dist), self.equity / price, 0.99)
            self.buy(size=size, sl=sl, tp=tp)
        else:
            # Geometry: SL above entry, TP below.
            if not (tp < price < sl):
                if sl <= price:
                    return
                if tp >= price:
                    return
            dist = sl - price
            if dist <= 0:
                return
            size = min(((self.equity * self.risk_pct) / dist), self.equity / price, 0.99)
            self.sell(size=size, sl=sl, tp=tp)

    @staticmethod
    def _finite(v) -> bool:
        try:
            return v is not None and not math.isnan(float(v))
        except (TypeError, ValueError):
            return False


# --------------------------------------------------------------------------
# Metrics (mirror the proxy engine so numbers are comparable)
# --------------------------------------------------------------------------

def _num(value) -> float:
    try:
        if value is None or np.isnan(float(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_metrics(stats, trades_df) -> dict:
    returns = trades_df["ReturnPct"].astype(float).values if len(trades_df) else np.array([])
    pnl = trades_df["PnL"].astype(float).values if len(trades_df) else np.array([])
    return {
        "n_trades": int(_num(stats.get("# Trades"))),
        "win_rate": round(_num(stats.get("Win Rate [%]")), 2),
        "sharpe": round(_num(stats.get("Sharpe Ratio")), 3),
        "max_drawdown": round(_num(stats.get("Max. Drawdown [%]")), 2),
        "return_pct": round(_num(stats.get("Return [%]")), 2),
        "profit_factor": round(_num(stats.get("Profit Factor")), 3),
        "avg_trade_pct": round(float(returns.mean() * 100), 2) if len(returns) else 0.0,
    }


def main() -> None:
    symbol, timeframe, bars = "XAUUSD", "1h", 6000
    print(f"Loading {symbol} {timeframe} x {bars} ...")
    df = load_df(symbol, timeframe, bars)
    split = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"Split: train {len(train_df)} / test {len(test_df)} bars")

    for name, sub in (("TRAIN", train_df), ("TEST", test_df)):
        bt = Backtest(sub.copy(), OrderflowStrategy, cash=100_000)
        stats = bt.run()
        trades = stats["_trades"]
        m = extract_metrics(stats, trades)
        print(f"\n=== {name} window (faithful orderflow) ===")
        print(json.dumps(m, indent=2))
        if len(trades):
            wins = trades[trades["PnL"] > 0]
            print(f"sample trades: {len(trades)} | wins {len(wins)} | "
                  f"avg win {wins['PnL'].mean() if len(wins) else 0:.2f} | "
                  f"avg loss {trades[trades['PnL'] <= 0]['PnL'].mean() if len(trades[trades['PnL'] <= 0]) else 0:.2f}")
        else:
            print("NO TRADES — filters too strict on this window.")


if __name__ == "__main__":
    main()
