"""AI strategy refinement agent.

Produces an initial parameter spec from a strategy's markdown body, and refines
it based on backtest metrics. Runs on the local Ollama model by default. Falls
back to a bounded heuristic perturbation when the LLM is unreachable so the
refinement loop still works offline.
"""

import random

from app.agents.ollama_client import OllamaClient, parse_json
from app.agents.validator import ValidationFailed, ValidatorChain
from app.services.model_config import get as model_get
from app.services.model_config import get_agent_model

# Allowed ranges / defaults for the parameterized trend strategy.
SPEC_DEFAULTS = {
    "fast": 10,
    "slow": 40,
    "rsi_period": 14,
    "rsi_buy": 30,
    "rsi_sell": 70,
    "atr_period": 14,
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 2.0,
    "risk_pct": 0.01,
}

SPEC_RANGES = {
    "fast": (3, 50),
    "slow": (10, 200),
    "rsi_period": (5, 28),
    "rsi_buy": (10, 45),
    "rsi_sell": (55, 90),
    "atr_period": (5, 28),
    "atr_sl_mult": (0.3, 4.0),
    "atr_tp_mult": (0.6, 6.0),
    "risk_pct": (0.002, 0.05),
}

INITIAL_PROMPT = """\
You are Meruem Strategy Refiner. Read the trading strategy below and derive the \
parameter set for a systematic long/short trend-following strategy that uses SMA \
crossovers with RSI confirmation and ATR-based stops/targets. Reason about what \
values would suit the strategy's time horizon and the asset class.

Strategy (markdown):
---
{markdown}
---
Symbol: {symbol}  Timeframe: {timeframe}

Return STRICT JSON only with these integer/float fields within the stated ranges:
"fast" (SMA fast, 3-50), "slow" (SMA slow, 10-200, must exceed fast by at least 5),
"rsi_period" (5-28), "rsi_buy" (10-45), "rsi_sell" (55-90, must exceed rsi_buy),
"atr_period" (5-28), "atr_sl_mult" (0.3-4.0), "atr_tp_mult" (0.6-6.0),
"risk_pct" (0.002-0.05), "rationale" (short explanation of the choices).
"""

REFINE_PROMPT = """\
You are Meruem Strategy Refiner. You are iteratively improving a trading strategy \
by backtesting WITHOUT overfitting: out-of-sample (holdout) performance is what \
matters; if a change only improves in-sample metrics it is overfitting and must be \
rejected. Prefer small, explainable parameter changes over large jumps.

Strategy (markdown):
---
{markdown}
---
Current parameters: {spec}
Recent metrics — TRAIN (in-sample): {train_metrics}
Recent metrics — TEST (out-of-sample holdout): {test_metrics}

Analyze the metrics (win rate, Sharpe, max drawdown, max win/loss, profit factor, \
number of trades) and propose improved parameters that raise out-of-sample \
win rate and Sharpe while keeping drawdown under control and at least ~30 trades.

KEY RULES:
- If n_trades is 0 or very low, the current parameters are too strict. LOOSEN the \
entries so the strategy actually trades: widen the RSI bands (raise rsi_buy toward \
40-45, lower rsi_sell toward 55-60), enlarge atr_sl_mult toward 2-3, and shorten \
the slow SMA toward 30-50. Getting real trades is the priority.
- If n_trades is healthy but win rate is poor, tighten trend confirmation and \
improve the risk/reward (raise atr_tp_mult relative to atr_sl_mult).

Return STRICT JSON only: "fast" (3-50), "slow" (10-200, > fast by >= 5), \
"rsi_period" (5-28), "rsi_buy" (10-45), "rsi_sell" (55-90, > rsi_buy), \
"atr_period" (5-28), "atr_sl_mult" (0.3-4.0), "atr_tp_mult" (0.6-6.0), \
"risk_pct" (0.002-0.05), plus "rationale" (what failed and the fix) and \
"predicted_improvement" (expected effect on the metrics).
"""


def clamp_spec(spec: dict) -> dict:
    out: dict = {}
    for key, default in SPEC_DEFAULTS.items():
        value = spec.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        lo, hi = SPEC_RANGES[key]
        value = max(lo, min(hi, value))
        if key in ("fast", "slow", "rsi_period", "rsi_buy", "rsi_sell", "atr_period"):
            out[key] = int(round(value))
        else:
            out[key] = round(value, 4)
    # Cross-field constraints.
    if out["slow"] - out["fast"] < 5:
        out["slow"] = out["fast"] + 5
        out["slow"] = min(200, out["slow"])
    if out["rsi_sell"] <= out["rsi_buy"]:
        out["rsi_sell"] = min(90, out["rsi_buy"] + 20)
    return out


class RefinementAgent:
    """AI strategy refiner. Uses the universal LLM provider chain (configured
    in the Vault) for refinement with the local Ollama model as final fallback,
    and a bounded heuristic perturbation when every LLM is offline."""

    def __init__(self):
        self.chain = ValidatorChain(agent_id="refiner")
        self.client = OllamaClient(
            base_url=model_get("ollama_url"),
            model=get_agent_model("refiner") or model_get("ollama_validator_model"),
        )

    def _generate_json(self, prompt: str) -> dict:
        try:
            raw, _ = self.chain.generate(prompt)
            return parse_json(raw)
        except ValidationFailed:
            return self.client.generate_json(prompt)

    def initial_spec(self, markdown: str, symbol: str, timeframe: str) -> dict:
        prompt = INITIAL_PROMPT.format(
            markdown=markdown[:4000], symbol=symbol, timeframe=timeframe
        )
        try:
            return clamp_spec(self._generate_json(prompt))
        except Exception:  # noqa: BLE001
            return self._heuristic(dict(SPEC_DEFAULTS), reason="LLM unavailable")["spec"]

    def refine(self, spec: dict, train: dict, test: dict, markdown: str) -> dict:
        prompt = REFINE_PROMPT.format(
            markdown=markdown[:4000],
            spec=spec,
            train_metrics={k: v for k, v in train.items() if k not in ("equity_curve", "trades")},
            test_metrics={k: v for k, v in test.items() if k not in ("equity_curve", "trades")},
        )
        try:
            data = self._generate_json(prompt)
            new_spec = clamp_spec(data)
            return {
                "spec": new_spec,
                "rationale": str(data.get("rationale", "LLM refinement")),
                "predicted_improvement": str(data.get("predicted_improvement", "")),
            }
        except Exception:  # noqa: BLE001
            return self._heuristic(spec, reason="LLM unavailable")

    def _heuristic(self, spec: dict, reason: str) -> dict:
        rng = random.Random(42)
        new = dict(spec)
        for key in ("fast", "slow", "rsi_period", "atr_period"):
            new[key] = max(int(SPEC_RANGES[key][0]),
                           min(int(SPEC_RANGES[key][1]),
                               int(round(new[key] * rng.uniform(0.85, 1.2)))))
        for key in ("atr_sl_mult", "atr_tp_mult", "risk_pct"):
            lo, hi = SPEC_RANGES[key]
            new[key] = round(max(lo, min(hi, new[key] * rng.uniform(0.85, 1.2))), 4)
        if new["slow"] - new["fast"] < 5:
            new["slow"] = min(200, new["fast"] + 5)
        if new["rsi_sell"] <= new["rsi_buy"]:
            new["rsi_sell"] = min(90, new["rsi_buy"] + 20)
        return {"spec": new, "rationale": f"{reason} — heuristic perturbation",
                "predicted_improvement": "unknown"}
