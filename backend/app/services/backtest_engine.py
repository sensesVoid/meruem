"""Backtest engine built on the ``backtesting`` library with an AI refinement loop.

Generates a parameterized SMA/RSI/ATR trend-following strategy from a strategy's
markdown body, runs it against real OHLCV via ``MarketDataService``, reports
metrics (win rate, drawdown, max win/loss, Sharpe + Monte-Carlo Sharpe) and runs
an LLM-driven refinement loop that guards against overfitting by holding out a
test split.
"""

import asyncio
import json
import math
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from sqlmodel import Session

from app.agents.refinement import RefinementAgent, clamp_spec
from app.models.entities import BacktestRun
from app.models.entities import Strategy as StrategyModel
from app.services.market_data import MarketDataService

MIN_BARS = 60
MIN_OOS_TRADES = 5
TRAIN_FRACTION = 0.7
MC_BOOTSTRAPS = 1500

# Cost modeling (backtesting 0.3.x has no spread/slippage args, only commission).
# Spread is charged deterministically as a round-turn commission fraction; extra
# slippage is sampled per-trade inside the Monte-Carlo bootstrap so the reported
# "cost-adjusted" regime reflects randomized fills.
DEFAULT_SPREAD_BPS = 6.0      # ~0.06% round-trip spread
DEFAULT_SLIPPAGE_BPS = 5.0    # ~0.05% randomized per-trade slippage
SLIPPAGE_SCATTER = 0.5        # +/-50% multiplicative scatter on slippage


def _read_bps(key: str, default: float) -> float:
    """Read a cost setting (bp) from the DB, falling back to config default."""
    try:
        from app.services.model_config import _read

        val = _read(key)
        if val is not None:
            return float(val)
    except Exception:  # noqa: BLE001,S110 - cost settings are best-effort
        pass
    return default


def _cost_commission() -> float:
    """Round-trip commission fraction pricing the spread into every fill."""
    return _read_bps("bt_spread_bps", DEFAULT_SPREAD_BPS) / 10_000.0


def _cost_bps() -> dict:
    return {
        "spread_bps": _read_bps("bt_spread_bps", DEFAULT_SPREAD_BPS),
        "slippage_bps": _read_bps("bt_slippage_bps", DEFAULT_SLIPPAGE_BPS),
    }


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _candles_for_backtest(
    symbol: str, timeframe: str, bars: int,
    start_date: str | None = None, end_date: str | None = None,
) -> tuple[list[dict], str]:
    """Source OHLCV for a backtest, in priority order:

    1. The extracted Parquet dataset for ``(symbol, timeframe)`` (sliced to the
       requested bar count / date window) — provider marked ``dataset``;
    2. Live providers for the EXACT date window when dates are requested and no
       usable dataset exists (``fetch_range``);
    3. Live providers, most recent ``bars`` candles.
    """
    from app.services.dataset_service import DatasetService, read_dataset_candles

    start = end = None
    if start_date:
        start = DatasetService._parse_date(start_date, None, "start_date")
    if end_date:
        end = DatasetService._parse_date(end_date, None, "end_date")
    if start is not None and end is not None and start >= end:
        raise ValueError("start_date must be before end_date.")

    try:
        rows = read_dataset_candles(symbol, timeframe, bars=bars, start=start, end=end)
    except Exception:  # noqa: BLE001 - dataset read is best-effort
        rows = None
    if rows and len(rows) >= MIN_BARS:
        return rows, "dataset"

    if start is not None or end is not None:
        start_ts = int(start.timestamp()) if start is not None else 0
        end_ts = int(end.timestamp()) if end is not None else int(time.time())
        candles, prov = asyncio.run(
            MarketDataService().fetch_range(symbol, timeframe, start_ts, end_ts))
        return candles, f"{prov} (range)"

    data = asyncio.run(MarketDataService().fetch(symbol, timeframe, bars))
    candles = data.get("candles") or []
    return candles, str(data.get("provider", ""))


def _load_df(symbol: str, timeframe: str, bars: int,
             start_date: str | None = None,
             end_date: str | None = None) -> tuple[pd.DataFrame, str]:
    candles, provider = _candles_for_backtest(
        symbol, timeframe, bars, start_date, end_date)
    if len(candles) < MIN_BARS:
        raise ValueError(
            f"Only {len(candles)} candles available for {symbol} {timeframe}; "
            f"need at least {MIN_BARS}. Try a larger timeframe."
        )
    df = pd.DataFrame(candles)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)
    df = df.dropna()
    if len(df) < MIN_BARS:
        raise ValueError("Not enough clean OHLCV rows after parsing.")
    return df, provider


# --------------------------------------------------------------------------
# Indicator helpers (no extra TA dependency)
# --------------------------------------------------------------------------

def _risk_size(equity: float, risk_pct: float, distance: float, price: float) -> float:
    """Risk-scaled position size.

    Backtesting requires the size to be a whole number of units OR a fraction of
    equity (< 1). It also rejects orders whose notional (units * price) exceeds
    available equity, so the unit count is capped to what the account can afford.
    A position too small for one full unit is returned as a fraction of equity.
    """
    if distance <= 0 or price <= 0 or equity <= 0:
        return 0.0
    risk_units = (equity * risk_pct) / distance
    max_units = equity / price
    if risk_units >= 1 and max_units >= 1:
        return float(max(1, int(min(risk_units, max_units))))
    return min(max(((equity * risk_pct) / price), 0.0), 0.99)


def _rsi(close, period: int):
    s = pd.Series(close).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def _atr(high, low, close, period: int):
    h = pd.Series(high).astype(float)
    l = pd.Series(low).astype(float)
    c = pd.Series(close).astype(float)
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().values


# --------------------------------------------------------------------------
# Parameterized strategy
# --------------------------------------------------------------------------

class ParameterStrategy(Strategy):
    fast = 10
    slow = 40
    rsi_period = 14
    rsi_buy = 30
    rsi_sell = 70
    atr_period = 14
    atr_sl_mult = 1.5
    atr_tp_mult = 2.0
    risk_pct = 0.01

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(
            lambda x: pd.Series(x).rolling(self.fast).mean().values, close)
        self.sma_slow = self.I(
            lambda x: pd.Series(x).rolling(self.slow).mean().values, close)
        self.rsi = self.I(lambda x: _rsi(x, self.rsi_period), close)
        high, low = self.data.High, self.data.Low
        self.atr = self.I(
            lambda H=high, L=low, C=close: _atr(H, L, C, self.atr_period))

    def next(self):
        if self.position:
            return
        values = (self.sma_fast[-1], self.sma_slow[-1], self.rsi[-1], self.atr[-1])
        if any(not np.isfinite(v) for v in values):
            return
        price = self.data.Close[-1]
        if self.sma_fast[-1] > self.sma_slow[-1] and self.rsi[-1] < self.rsi_buy:
            sl = price - self.atr[-1] * self.atr_sl_mult
            if sl <= 0:
                return
            size = _risk_size(self.equity, self.risk_pct, price - sl, price)
            self.buy(size=size, sl=sl, tp=price + self.atr[-1] * self.atr_tp_mult)
        elif self.sma_fast[-1] < self.sma_slow[-1] and self.rsi[-1] > self.rsi_sell:
            sl = price + self.atr[-1] * self.atr_sl_mult
            size = _risk_size(self.equity, self.risk_pct, sl - price, price)
            self.sell(size=size, sl=sl, tp=price - self.atr[-1] * self.atr_tp_mult)


def _build_strategy(spec: dict) -> type:
    return type("GeneratedStrategy", (ParameterStrategy,), dict(spec))


# --------------------------------------------------------------------------
# Metric extraction
# --------------------------------------------------------------------------

def _num(value) -> float:
    try:
        if value is None or np.isnan(float(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mc_sharpe(returns, n_boot: int = MC_BOOTSTRAPS) -> dict:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return {"p5": 0.0, "median": 0.0, "p95": 0.0, "n": 0}
    rng = np.random.default_rng(7)
    n = arr.size
    values: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        sd = sample.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            continue
        values.append(float(sample.mean() / sd * math.sqrt(n)))
    values.sort()
    p5 = values[int(len(values) * 0.05)]
    p95 = values[min(int(len(values) * 0.95), len(values) - 1)]
    return {
        "p5": round(p5, 3),
        "median": round(values[len(values) // 2], 3),
        "p95": round(p95, 3),
        "n": len(values),
    }


def _mc_costs(returns, spread_bps: float, slippage_bps: float,
              n_boot: int = MC_BOOTSTRAPS) -> dict:
    """Monte-Carlo Sharpe with RANDOMIZED spread + slippage per trade.

    The deterministic `commission` prices the base spread into every fill; this
    bootstrap additionally samples a per-trade slippage cost (varying within
    +/-``SLIPPAGE_SCATTER`` of the configured rate) so the distribution reflects
    randomized fills / market impact. Returns percentile bands for both the
    cost-adjusted Sharpe and the realized (post-cost) mean return.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return {
            "spread_bps": spread_bps, "slippage_bps": slippage_bps,
            "sharpe_p5": 0.0, "sharpe_median": 0.0, "sharpe_p95": 0.0,
            "ret_p5": 0.0, "ret_median": 0.0, "ret_p95": 0.0, "n": 0,
        }
    rng = np.random.default_rng(1234)
    half = SLIPPAGE_SCATTER
    n = arr.size
    sharpe_vals: list[float] = []
    ret_vals: list[float] = []
    # Slippage as a fraction of a per-trade return: random within the band.
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True) / 100.0  # % -> fraction
        slip = rng.uniform(1 - half, 1 + half) * (slippage_bps / 10_000.0)
        costed = sample - slip
        avg = float(costed.mean())
        sd = costed.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            continue
        ret_vals.append(avg)
        sharpe_vals.append(float(avg / sd * math.sqrt(n)))
    sharpe_vals.sort()
    ret_vals.sort()
    lo = max(0, int(len(sharpe_vals) * 0.05)) - 1
    hi = min(int(len(sharpe_vals) * 0.95), len(sharpe_vals) - 1)
    mid = len(sharpe_vals) // 2
    return {
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "sharpe_p5": round(sharpe_vals[lo], 3),
        "sharpe_median": round(sharpe_vals[mid], 3),
        "sharpe_p95": round(sharpe_vals[hi], 3),
        "ret_p5": round(float(ret_vals[lo]), 4),
        "ret_median": round(float(ret_vals[mid]), 4),
        "ret_p95": round(float(ret_vals[hi]), 4),
        "n": len(sharpe_vals),
    }


def _metrics(stats, trades_df, cost_bps: dict | None = None) -> dict:
    returns = trades_df["ReturnPct"].astype(float).values if len(trades_df) else np.array([])
    pnl = trades_df["PnL"].astype(float).values if len(trades_df) else np.array([])
    out = {
        "n_trades": int(_num(stats.get("# Trades"))),
        "win_rate": round(_num(stats.get("Win Rate [%]")), 2),
        "sharpe": round(_num(stats.get("Sharpe Ratio")), 3),
        "max_drawdown": round(_num(stats.get("Max. Drawdown [%]")), 2),
        "return_pct": round(_num(stats.get("Return [%]")), 2),
        "profit_factor": round(_num(stats.get("Profit Factor")), 3),
        "max_win": round(float(pnl.max())) if len(pnl) else 0.0,
        "max_loss": round(float(pnl.min())) if len(pnl) else 0.0,
        "avg_trade_pct": round(float(returns.mean() * 100), 2) if len(returns) else 0.0,
        "mc_sharpe": _mc_sharpe(returns),
    }
    if cost_bps:
        out["mc_costs"] = _mc_costs(
            returns, cost_bps["spread_bps"], cost_bps["slippage_bps"])
    return out


def _composite(metrics: dict) -> float:
    if metrics["n_trades"] < MIN_OOS_TRADES:
        return -100.0
    sharpe = metrics.get("sharpe") or 0.0
    win = metrics.get("win_rate") or 0.0
    dd = abs(metrics.get("max_drawdown") or 0.0)
    return sharpe * 100 + win * 0.5 - dd * 0.25


def _equity_points(equity_df, max_points: int = 600) -> list[dict]:
    if equity_df is None or len(equity_df) == 0:
        return []
    step = max(1, len(equity_df) // max_points)
    points = []
    for i in range(0, len(equity_df), step):
        row = equity_df.iloc[i]
        points.append({
            "t": int(equity_df.index[i].timestamp()),
            "equity": round(float(row.get("Equity", 0.0)), 2),
            "dd": round(float(row.get("DrawdownPct", 0.0)), 3),
        })
    last = equity_df.index[-1]
    points.append({
        "t": int(last.timestamp()),
        "equity": round(float(equity_df.iloc[-1].get("Equity", 0.0)), 2),
        "dd": round(float(equity_df.iloc[-1].get("DrawdownPct", 0.0)), 3),
    })
    return points


def _trades_list(trades_df, max_trades: int = 200) -> list[dict]:
    if trades_df is None or len(trades_df) == 0:
        return []
    out = []
    for _, row in trades_df.head(max_trades).iterrows():
        out.append({
            "size": round(float(row.get("Size", 0.0)), 4),
            "entry_price": round(float(row.get("EntryPrice", 0.0)), 4),
            "exit_price": round(float(row.get("ExitPrice", 0.0)), 4),
            "pnl": round(float(row.get("PnL", 0.0)), 2),
            "ret_pct": round(float(row.get("ReturnPct", 0.0)) * 100, 2),
            "entry_time": row.get("EntryTime"),
            "exit_time": row.get("ExitTime"),
            "duration": str(row.get("Duration", "")),
        })
    return out


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class BacktestEngine:
    def __init__(self):
        self.agent = RefinementAgent()

    def run(self, symbol: str, timeframe: str, spec: dict, bars: int = 1500,
            start_date: str | None = None,
            end_date: str | None = None) -> dict:
        spec = clamp_spec(spec)
        df, provider = _load_df(symbol, timeframe, bars, start_date, end_date)
        stats, trades_df, equity_df = self._run_df(df, spec)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "provider": provider,
            "spec": spec,
            "cost_bps": _cost_bps(),
            "metrics": _metrics(stats, trades_df, _cost_bps()),
            "equity_curve": _equity_points(equity_df),
            "trades": _trades_list(trades_df),
            "status": "ok",
        }

    def _run_df(self, df: pd.DataFrame, spec: dict, commission: float | None = None):
        bt = Backtest(df.copy(), _build_strategy(spec), cash=100_000,
                      commission=_cost_commission() if commission is None else commission)
        stats = bt.run()
        trades_df = stats.get("_trades")
        equity_df = stats.get("_equity_curve")
        return stats, trades_df, equity_df

    def run_code(self, symbol: str, timeframe: str, code: str,
                 bars: int = 1500, confluence: dict | None = None,
                 start_date: str | None = None,
                 end_date: str | None = None) -> dict:
        """FAITHFUL backtest: run the trader's AI-generated strategy code directly.
        No proxy SMA/RSI/ATR parameters are used to drive entries — the code IS the
        strategy. `confluence` is only echoed for transparency (scanner-only).
        ``start_date``/``end_date`` select an exact window (extracted dataset or
        live range); ``None`` keeps the last-``bars`` behavior."""
        from app.services.code_sandbox import build_strategy_class, validate_code

        ok, err = validate_code(code)
        if not ok:
            raise ValueError(f"Generated strategy rejected by sandbox: {err}")
        df, provider = _load_df(symbol, timeframe, bars, start_date, end_date)
        cls = build_strategy_class(code)
        bt = Backtest(df.copy(), cls, cash=100_000, commission=_cost_commission())
        stats = bt.run()
        trades_df = stats.get("_trades")
        equity_df = stats.get("_equity_curve")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "provider": provider,
            "faithful": True,
            "spec": dict(confluence or {}),
            "cost_bps": _cost_bps(),
            "metrics": _metrics(stats, trades_df, _cost_bps()),
            "equity_curve": _equity_points(equity_df),
            "trades": _trades_list(trades_df),
            "status": "ok",
        }

    def refine_loop(
        self,
        session: Session,
        strategy: StrategyModel,
        symbol: str,
        timeframe: str,
        bars: int = 1500,
        max_iterations: int = 6,
        min_winrate: float = 50.0,
        min_sharpe: float = 1.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        df, provider = _load_df(symbol, timeframe, bars, start_date, end_date)
        split = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        if len(test_df) < MIN_BARS or len(train_df) < MIN_BARS:
            raise ValueError("Not enough candles to split into train/test windows.")

        # Seed from the spec already applied to the strategy (if any) so the
        # refinement loop continues from the last optimized state rather than
        # re-deriving an initial guess each time.
        start = None
        if getattr(strategy, "params_json", ""):
            try:
                parsed = json.loads(strategy.params_json)
                if isinstance(parsed, dict) and parsed:
                    start = parsed
            except (TypeError, ValueError):
                start = None
        spec = clamp_spec(start or self.agent.initial_spec(strategy.body_markdown, symbol, timeframe))
        best: dict | None = None
        iterations: list[dict] = []
        status = "stalled"

        for it in range(max_iterations):
            train_stats, _, _ = self._run_df(train_df, spec)
            test_stats, test_trades, _ = self._run_df(test_df, spec)
            train_m = _metrics(train_stats, train_stats.get("_trades"))
            test_m = _metrics(test_stats, test_trades)
            score = _composite(test_m)
            train_score = _composite(train_m)

            accepted, reason = self._accept(test_m, train_m, score, train_score, best)
            if accepted:
                best = {
                    "iteration": it,
                    "spec": spec,
                    "train": train_m,
                    "test": test_m,
                    "score": score,
                    "train_score": train_score,
                }

            self._persist(session, strategy.id, symbol, timeframe, it,
                          spec, train_m, test_m, reason, int(accepted))

            iterations.append({
                "iteration": it,
                "accepted": accepted,
                "spec": spec,
                "train": train_m,
                "test": test_m,
                "rationale": reason,
            })

            if best and self._target_met(best["train"], best["test"], min_winrate, min_sharpe):
                status = "target_met"
                break
            if it < max_iterations - 1:
                refined = self.agent.refine(spec, train_m, test_m, strategy.body_markdown)
                spec = clamp_spec(refined["spec"])
                iterations[-1]["next_rationale"] = refined["rationale"]
                iterations[-1]["predicted_improvement"] = refined.get("predicted_improvement", "")

        if best is not None and status != "target_met":
            status = "improved" if best["score"] >= _composite(iterations[0]["test"]) + 1.5 else "stalled"

        return {
            "strategy_id": strategy.id,
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "split": {"train": len(train_df), "test": len(test_df)},
            "provider": provider,
            "status": status,
            "best": best,
            "iterations": iterations,
        }

    @staticmethod
    def _accept(test_m: dict, train_m: dict, score: float,
                train_score: float, best: dict | None) -> tuple[bool, str]:
        if best is None:
            return True, f"baseline accepted (iteration 1, {test_m['n_trades']} trades)"
        if test_m["n_trades"] < MIN_OOS_TRADES:
            return False, f"rejected: fewer than {MIN_OOS_TRADES} out-of-sample trades"
        b_score = best["score"]
        gap = train_score - score
        if score >= b_score + 1.5:
            return True, f"significant out-of-sample improvement (score {score:.1f} vs {b_score:.1f})"
        if score >= b_score and gap <= 12 and test_m["win_rate"] >= 40:
            return True, f"moderate improvement without overfit signal (score {score:.1f} vs {b_score:.1f})"
        return False, f"rejected: not a significant improvement (score {score:.1f} vs best {b_score:.1f})"

    def refine_code_loop(
        self,
        session: Session,
        strategy: StrategyModel,
        symbol: str,
        timeframe: str,
        code: str,
        bars: int = 1500,
        max_iterations: int = 6,
        min_winrate: float = 50.0,
        min_sharpe: float = 1.0,
        confluence: dict | None = None,
        concept: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Faithful backtest loop over AI-generated strategy code.

        Runs the trader's ACTUAL rules (compiled via the code sandbox) on the
        same train/test split as the proxy loop, then iteratively asks the LLM to
        improve its own implementation (repair loop). Returns the best iteration
        with the accepted code, so the pipeline persists exactly what backtested.
        """
        from app.agents.strategy_codegen import StrategyCodeGen
        from app.services.code_sandbox import (
            build_strategy_class,
            validate_code,
        )

        df, provider = _load_df(symbol, timeframe, bars, start_date, end_date)
        split = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        if len(test_df) < MIN_BARS or len(train_df) < MIN_BARS:
            raise ValueError("Not enough candles to split into train/test windows.")

        ok, err = validate_code(code)
        if not ok:
            raise ValueError(f"Generated strategy rejected: {err}")

        agent = StrategyCodeGen()
        best: dict | None = None
        iterations: list[dict] = []
        status = "stalled"
        current = code
        confluence = dict(confluence or {})

        for it in range(max_iterations):
            try:
                cls = build_strategy_class(current)
                bt_train = Backtest(train_df.copy(), cls, cash=100_000,
                                    commission=_cost_commission())
                train_stats = bt_train.run()
                bt_test = Backtest(test_df.copy(), cls, cash=100_000,
                                   commission=_cost_commission())
                test_stats = bt_test.run()
                # Include the randomized spread+slippage Monte-Carlo benchmark so
                # the synthesis diagnostics can show the cost-adjusted edge.
                cost_bps = _cost_bps()
                train_m = _metrics(train_stats, train_stats.get("_trades"), cost_bps)
                test_m = _metrics(test_stats, test_stats.get("_trades"), cost_bps)
            except Exception as exc:  # noqa: BLE001
                if best is not None:
                    break  # a later repair broke the code — keep the best we had
                raise ValueError(f"Generated strategy failed to run: {exc}")

            score = _composite(test_m)
            train_score = _composite(train_m)
            accepted, reason = self._accept(test_m, train_m, score, train_score, best)
            if accepted:
                best = {
                    "iteration": it,
                    "code": current,
                    "confluence": confluence,
                    "spec": confluence,
                    "train": train_m,
                    "test": test_m,
                    "score": score,
                    "train_score": train_score,
                }

            self._persist(session, strategy.id, symbol, timeframe, it,
                          {"code": current, "confluence": confluence},
                          train_m, test_m, reason, int(accepted))

            iterations.append({
                "iteration": it,
                "accepted": accepted,
                "spec": confluence,
                "train": train_m,
                "test": test_m,
                "rationale": reason,
                "code_changed": it > 0,
            })

            if best and self._target_met(best["train"], best["test"],
                                         min_winrate, min_sharpe):
                status = "target_met"
                break

            if it < max_iterations - 1:
                try:
                    repaired = agent.repair(
                        strategy.name, concept or strategy.concept_markdown,
                        strategy.body_markdown, symbol, timeframe,
                        current, train_m, test_m,
                    )
                    candidate = repaired.get("code") or ""
                    rok, _rerr = validate_code(candidate)
                    if rok and candidate != current:
                        current = candidate
                        if repaired.get("confluence"):
                            confluence = repaired["confluence"]
                        iterations[-1]["next_rationale"] = repaired.get("notes", "")
                except Exception:  # noqa: BLE001
                    break  # LLM unavailable — stop improving, keep best

        if best is not None and status != "target_met":
            status = "improved" if best["score"] >= _composite(iterations[0]["test"]) + 1.5 \
                else "stalled"

        return {
            "strategy_id": strategy.id,
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "split": {"train": len(train_df), "test": len(test_df)},
            "provider": provider,
            "faithful": True,
            "status": status,
            "best": best,
            "iterations": iterations,
        }

    @staticmethod
    def _target_met(train_m: dict, test_m: dict, min_winrate: float, min_sharpe: float) -> bool:
        for metrics in (train_m, test_m):
            if metrics["n_trades"] < MIN_OOS_TRADES:
                return False
            if metrics["win_rate"] < min_winrate:
                return False
            if metrics["sharpe"] < min_sharpe:
                return False
        return True

    @staticmethod
    def _persist(session: Session, strategy_id: str, symbol: str, timeframe: str,
                 iteration: int, spec: dict, train_m: dict, test_m: dict,
                 rationale: str, accepted: int) -> None:
        row = BacktestRun(
            id=f"bt-{uuid4().hex[:16]}",
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            iteration=iteration,
            spec_json=json.dumps(spec),
            metrics_json=json.dumps({"train": train_m, "test": test_m}),
            rationale=rationale,
            accepted=accepted,
        )
        session.add(row)
        session.commit()
