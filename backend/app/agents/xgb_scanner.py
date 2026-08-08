"""XGBoost Scanner Agent.

Drop-in replacement for the Ollama ScannerAgent. When a trained XGBoost model
exists it produces a candidate dict using the ML classifier. If no model has
been trained yet (or inference fails), falls back to the Ollama LLM scanner
transparently so the system always works without requiring upfront training.

The candidate output schema is identical to the LLM scanner's output so the
Validator and rest of the pipeline are unaffected.
"""

from __future__ import annotations

import json


class XGBoostScannerAgent:
    """XGBoost-based setup detector with automatic Ollama fallback.

    Priority chain:
      1. XGBoost inference (fast, deterministic) — if a model is trained
      2. Ollama LLM scanner (rule-grounded)      — always available fallback
    """

    def scan(
        self,
        ohlcv: dict,
        indicators: dict,
        strategies: list[dict],
        references: list[dict] | None = None,
    ) -> dict:
        rows = ohlcv.get("rows") or []
        symbol = ohlcv.get("symbol", "")
        asset_class = ohlcv.get("asset_class", "")
        timeframe = ohlcv.get("timeframe", "")

        # -- Try XGBoost first --
        ml_result = self._ml_scan(rows, symbol, asset_class, timeframe, strategies)
        if ml_result is not None:
            return ml_result

        # -- Fallback to Ollama LLM scanner --
        return self._llm_scan(ohlcv, indicators, strategies, references)

    # ------------------------------------------------------------------ ML
    def _ml_scan(
        self,
        rows: list[dict],
        symbol: str,
        asset_class: str,
        timeframe: str,
        strategies: list[dict],
    ) -> dict | None:
        """Run XGBoost inference. Returns None if model not available."""
        try:
            from app.services.ml_scanner import predict  # type: ignore

            result = predict(rows)
            if result is None:
                return None  # Model not trained yet → fall back

            direction = result.get("direction")  # "BUY" / "SELL" / None
            confidence = result.get("confidence", 0.0)
            proba = result.get("probabilities", {})

            # Strategy reference attribution: all active strategies are cited.
            strategy_refs = [s.get("name", "") for s in strategies if s.get("name")]

            # Build an entry_zone hint using the last close ± ATR if possible.
            entry_zone: list[float] = []
            invalidation: float | None = None
            if rows:
                last_close = float(rows[-1].get("close", 0))
                closes = [float(r["close"]) for r in rows]
                highs = [float(r["high"]) for r in rows]
                lows = [float(r["low"]) for r in rows]
                atr = self._atr(highs, lows, closes)
                if atr and direction == "BUY":
                    entry_zone = [round(last_close, 5), round(last_close + atr * 0.5, 5)]
                    invalidation = round(last_close - atr * 1.5, 5)
                elif atr and direction == "SELL":
                    entry_zone = [round(last_close - atr * 0.5, 5), round(last_close, 5)]
                    invalidation = round(last_close + atr * 1.5, 5)

            rationale = (
                f"[XGBoost ML Scanner] Classifier confidence: {confidence:.1%}. "
                f"P(BUY)={proba.get('buy', 0):.3f} | P(SELL)={proba.get('sell', 0):.3f} | "
                f"P(NONE)={proba.get('none', 0):.3f}. "
                f"Based on {len(rows)} OHLCV bars; feature set: RSI, ATR%, ADX, EMA cross, "
                f"momentum, vol-ratio, candle geometry, trend dir, GARCH, Markov."
            )

            return {
                "asset": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "direction": direction,
                "confidence": round(confidence, 4),
                "entry_zone": entry_zone,
                "invalidation": invalidation,
                "rationale": rationale,
                "strategy_refs": strategy_refs,
                "ml_scanner": True,
                "ml_probabilities": proba,
            }

        except Exception as exc:  # noqa: BLE001
            # Any import/inference failure → fall back to LLM
            return None

    # ------------------------------------------------------------------ LLM fallback
    def _llm_scan(
        self,
        ohlcv: dict,
        indicators: dict,
        strategies: list[dict],
        references: list[dict] | None,
    ) -> dict:
        try:
            from app.agents.scanner import ScannerAgent  # type: ignore

            return ScannerAgent().scan(ohlcv, indicators, strategies, references=references)
        except Exception as exc:  # noqa: BLE001
            return {
                "asset": ohlcv.get("symbol"),
                "direction": None,
                "confidence": 0.0,
                "rationale": f"Both ML and LLM scanners unavailable: {exc}",
                "error": True,
            }

    def normalize(self, candidate: dict) -> dict:
        """Pass-through normaliser (same contract as ScannerAgent)."""
        refs = candidate.get("strategy_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        candidate["strategy_refs"] = [r for r in refs if isinstance(r, str)]
        return candidate

    @staticmethod
    def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        return sum(trs[-period:]) / period
