"""AI strategy code generator.

Turns a trader's markdown rule set into an executable ``backtesting`` Strategy
class. This is the FULLY-FAITHFUL path: instead of mapping every strategy onto a
fixed SMA/RSI/ATR proxy, the LLM writes strategy-specific code (orderflow, price
action, Munehisa Homa candlesticks, breakouts, ...) so the backtest/refine loop
and later the Scanner/Validator judge the trader's actual logic.

Creativity is allowed in HOW a concept is implemented (which indicator functions,
window sizes, thresholds to realize a rule) but the ENTRY / EXIT / RISK rules the
trader submitted are a hard boundary: the model must not invent new entry logic.

Generated code is executed ONLY through the strict sandbox (see
``app/services/code_sandbox.py``) which blocks unsafe imports/IO.
"""

import json

from app.agents.ollama_client import parse_json
from app.agents.validator import ValidatorChain

CODEGEN_PROMPT = """\
You are Meruem Strategy Engineer. Write a complete, executable ``backtesting`` \
strategy that faithfully implements the trader's ENTRY CRITERIA, applied to their \
TRADING CONCEPT. Priority #1 is FAITHFULNESS to the submitted entry rules.

STRATEGY NAME: {name}
SYMBOL: {symbol}   TIMEFRAME: {timeframe}

TRADING CONCEPT (context — the author's idea; guides HOW you realize the entry, \
but the ENTRY CRITERIA below is the hard boundary):
---BEGIN CONCEPT---
{concept}
---END CONCEPT---

TRADER'S ENTRY CRITERIA (HARD BOUNDARY — implement these EXACT rules):
---BEGIN ENTRY CRITERIA---
{entry}
---END ENTRY CRITERIA---

REQUIREMENTS (hard constraints, never violate):
1. ALLOWED IMPORTS ONLY: `from backtesting import Strategy, Backtest`, \
`import numpy as np`, `import pandas as pd`, `import math`. No other imports.
2. Your class MUST be named `GeneratedStrategy` and MUST subclass `Strategy`.
3. Define `risk_pct` (float, 0.002-0.05) as a class attribute, plus any \
execution knobs the rules imply (e.g. ATR multipliers, swing window N).
4. `init(self)`: compute all indicators/profiles you need from self.data
   (self.data.Open/High/Low/Close/Volume / self.data.index). Use self.I() for
   any derived series. Do NOT look ahead: values at bar i must use only data
   through bar i (use .shift() for past bars), never future bars.
5. `next(self)`: implement the trader's EXACT entry, stop, and target rules.
   - Entry must follow ONLY the trader's ENTRY CRITERIA. You may implement a rule
     creatively (exact window, threshold) but do NOT add entry logic the trader
     did not state.
- Call self.buy(size=..., sl=..., tp=...) or self.sell(...) with concrete
      values. Respect max 1 position open at a time (`if self.position: return`).
   - Guard every indicator/level for NaN/inf before using it (skip the bar if
      any needed value is not finite).
   - AVOID conditions that rarely or never co-occur on the SAME bar: a fresh
     breakout (close beyond an N-bar extreme) and a same-bar sweep of the OPPOSITE
     N-bar extreme (low below the prior low WHILE close breaks the prior high)
     cannot both fire — that bar does not exist. Sequence multi-part setups across
     bars (e.g. sweep on bar t, confirmation/retest on t+1) or relax one condition,
     so the strategy actually generates trades.
- Size so that if SL is hit you lose ~ risk_pct of equity:
        raw = risk_pct * self.equity / (entry - sl)
        size = raw if 0 < raw < 1 else max(1, int(round(raw)))
     backtesting REQUIRES size to be a fraction of equity (0<size<1) OR a whole
     number of units — never pass a fractional number >= 1 (e.g. 312.5 is invalid).
   - Validate SL/TP geometry: for buy, sl < entry < tp; for sell, tp < entry < sl.
     If inverted on a bar, do not trade that bar.
6. Do NOT reference files, network, raw volumes-as-time, or anything outside the
   OHLCV data provided by self.data. Keep it deterministic (no RNG).
7. If the trader's ENTRY CRITERIA uses terms NOT computable from OHLCV alone (e.g.
   true premium/discount from COMEX futures), implement the closest deterministic
   OHLCV approximation and RETURN a "notes" string stating exactly what you
   approximated.

Keep faithful: HOW a concept is realized (windows, thresholds) is somewhat flexible,
but the trader's ENTRY / EXIT / RISK rules are a hard boundary — never change them.

Return STRICT JSON ONLY with keys:
"code": string (the full Python source of your Strategy class),
"notes": string (what you implemented, and any approximation you had to make),
"confluence": object. This is used ONLY as an extra confirmation layer by the live \
scanner; it must NOT drive entries. Pick values that resemble the concept: \
  fast (int 3-50), slow (int 10-200, > fast), rsi_period (int 5-28),
  rsi_buy (int 10-45), rsi_sell (int 55-90, > rsi_buy),
  atr_period (int 5-28), atr_sl_mult (float 0.3-4.0), atr_tp_mult (float 0.6-6.0),
  risk_pct (float 0.002-0.05).

Code block must be valid Python and contain no markdown fences/newline-slash-escapes.
"""


REPAIR_PROMPT = """\
You are Meruem Strategy Engineer. The strategy you wrote is being improved via \
backtesting WITHOUT overfitting: OUT-OF-SAMPLE (holdout) performance is what \
matters. You may adjust how an entry rule is implemented (windows, thresholds, guards) \
but you MUST keep the trader's ENTRY CRITERIA and TRADING CONCEPT intact.

STRATEGY NAME: {name}
SYMBOL: {symbol}   TIMEFRAME: {timeframe}

TRADING CONCEPT (unchanged, context only):
---BEGIN CONCEPT---
{concept}
---END CONCEPT---

TRADER'S ENTRY CRITERIA (unchanged, keep faithful):
---BEGIN ENTRY CRITERIA---
{entry}
---END ENTRY CRITERIA---

CURRENT CODE:
---BEGIN CODE---
{code}
---END CODE---

RECENT METRICS — TRAIN (in-sample): {train_metrics}
RECENT METRICS — TEST (out-of-sample holdout): {test_metrics}

Analyze the metrics and improve ONLY the implementation to raise OUT-OF-SAMPLE \
win rate / Sharpe while keeping drawdown controlled and at least ~30 trades.

KEY RULES:
- If n_trades is 0 or very low, the implementation is too strict: widen windows, \
relax the filter thresholds, or loosen stop placement. Getting real trades is priority #1.
- If n_trades is healthy but win rate is poor, tighten confirmation / improve \
risk-reward (raise TP relative to SL).
- If TEST collapses below TRAIN by a lot, you are overfitting — simplify, and do \
not chase the in-sample curve.
- Never change the trader's ENTRY CRITERIA concept. Only adjust how faithfully each \
rule is computed or the risk/filter knobs.
- Keep class name `GeneratedStrategy`, allowed imports only \
(backtesting Strategy/Backtest, numpy, pandas, math), no lookahead, guard NaN.

Return STRICT JSON ONLY with keys: "code", "notes", "confluence".
"""


class StrategyCodeGen:
    """Generates executable strategy code from a trader's markdown rules."""

    def __init__(self):
        self.chain = ValidatorChain(agent_id="synthesizer")

    def generate(self, title: str, concept: str, markdown: str, symbol: str,
                 timeframe: str, ir_blueprint: list[dict] | None = None) -> dict:
        ir_block = ""
        if ir_blueprint:
            ir_block = (
                "\n\nMACHINE-VERIFIED IR BLUEPRINT (from IRCompilerEngine — implement "
                "these exact primitives; the backtest engine already owns their code):\n"
                f"{json.dumps({'rules': ir_blueprint}, default=str, indent=2)}\n"
            )
        prompt = CODEGEN_PROMPT.format(
            name=title, concept=concept[:6000], entry=markdown[:6000],
            symbol=symbol, timeframe=timeframe,
        ) + ir_block
        return self._run(prompt)

    def repair(self, title: str, concept: str, markdown: str, symbol: str, timeframe: str,
               code: str, train_metrics: dict, test_metrics: dict) -> dict:
        prompt = REPAIR_PROMPT.format(
            name=title, concept=concept or "", entry=markdown[:6000],
            symbol=symbol, timeframe=timeframe,
            code=code,
            train_metrics={k: v for k, v in train_metrics.items()
                           if k not in ("equity_curve", "trades")},
            test_metrics={k: v for k, v in test_metrics.items()
                          if k not in ("equity_curve", "trades")},
        )
        return self._run(prompt)

    def _run(self, prompt: str) -> dict:
        try:
            raw, provider = self.chain.generate(prompt)
            data = parse_json(raw)
        except Exception as exc:
            raise RuntimeError(f"Strategy codegen LLM failed: {exc}") from exc
        code = (data.get("code") or "").strip()
        if not code:
            raise RuntimeError("Codegen returned no strategy code.")
        return {
            "code": code,
            "notes": str(data.get("notes") or ""),
            "confluence": data.get("confluence") or {},
            "provider": provider,
        }


def heuristic_strategy() -> dict:
    """Fallback template when the LLM is unavailable: a safe SMA/RSI/ATR proxy
    so the pipeline still completes. Matches the old parameterized default."""
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import math\n"
        "from backtesting import Strategy\n"
        "\n"
        "class GeneratedStrategy(Strategy):\n"
        "    fast = 10\n"
        "    slow = 40\n"
        "    rsi_period = 14\n"
        "    rsi_buy = 30\n"
        "    rsi_sell = 70\n"
        "    atr_period = 14\n"
        "    atr_sl_mult = 1.5\n"
        "    atr_tp_mult = 2.0\n"
        "    risk_pct = 0.01\n"
        "\n"
        "    def init(self):\n"
        "        close = self.data.Close\n"
        "        self.sma_fast = self.I(lambda x: pd.Series(x).rolling(self.fast).mean().values, close)\n"
        "        self.sma_slow = self.I(lambda x: pd.Series(x).rolling(self.slow).mean().values, close)\n"
        "        self.rsi = self.I(lambda x: self._rsi(x, self.rsi_period), close)\n"
        "        self.atr = self.I(lambda H=self.data.High, L=self.data.Low, C=self.data.Close: self._atr(H, L, C, self.atr_period))\n"
        "\n"
        "    @staticmethod\n"
        "    def _rsi(close, period):\n"
        "        s = pd.Series(close)\n"
        "        delta = s.diff()\n"
        "        g = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()\n"
        "        l = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()\n"
        "        return (100 - 100/(1 + g/l.replace(0, np.nan))).values\n"
        "\n"
        "    @staticmethod\n"
        "    def _atr(high, low, close, period):\n"
        "        pc = pd.Series(close).shift(1)\n"
        "        tr = pd.concat([pd.Series(high)-pd.Series(low), (pd.Series(high)-pc).abs(), (pd.Series(low)-pc).abs()], axis=1).max(axis=1)\n"
        "        return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values\n"
        "\n"
        "    def next(self):\n"
        "        if self.position:\n"
        "            return\n"
        "        price = self.data.Close[-1]\n"
        "        if not math.isfinite(self.sma_fast[-1]) or not math.isfinite(self.sma_slow[-1]):\n"
        "            return\n"
        "        if self.sma_fast[-1] > self.sma_slow[-1] and self.rsi[-1] < self.rsi_buy:\n"
        "            sl = price - self.atr[-1] * self.atr_sl_mult\n"
        "            if sl <= 0: return\n"
        "            size = min(self.risk_pct*self.equity/(price-sl), self.equity/price)\n"
        "            self.buy(size=size, sl=sl, tp=price + self.atr[-1]*self.atr_tp_mult)\n"
        "        elif self.sma_fast[-1] < self.sma_slow[-1] and self.rsi[-1] > self.rsi_sell:\n"
        "            sl = price + self.atr[-1] * self.atr_sl_mult\n"
        "            size = min(self.risk_pct*self.equity/(sl-price), self.equity/price)\n"
        "            self.sell(size=size, sl=sl, tp=price - self.atr[-1]*self.atr_tp_mult)\n"
    )
    return {"code": code, "notes": "Heuristic fallback (LLM offline).",
            "confluence": {
                "fast": 10, "slow": 40, "rsi_period": 14, "rsi_buy": 30,
                "rsi_sell": 70, "atr_period": 14, "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.0, "risk_pct": 0.01,
            }, "provider": "heuristic"}