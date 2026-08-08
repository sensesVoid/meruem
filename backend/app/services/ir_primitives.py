"""Default mathematical primitive catalog for the IR layer (IR.md).

Each primitive is a pure OHLCV computation the backtest engine actually supports.
The signatures declare the exact arguments the Synthesizer Agent and code generator
may reference; ``register_primitive`` introspects them into the Tool Schema Menu.

Primitives are pure — they only describe *what* to compute, never touch IO, which
keeps the IR layer usable by the sandboxed codegen path.
"""

from app.services.ir import DynamicToolRegistry


def build_default_registry() -> DynamicToolRegistry:
    """Build and register the pipeline's supported math primitives."""
    registry = DynamicToolRegistry()

    @registry.register_primitive(
        "SMA_CROSS",
        "Trend bias from a fast/slow simple-moving-average crossover "
        "(fast above slow is bullish, below is bearish).")
    def sma_cross(fast: int = 10, slow: int = 50,
                  direction: str = "long") -> dict:
        return {"fast": fast, "slow": slow, "direction": direction}

    @registry.register_primitive(
        "RSI_FILTER",
        "Overbought/oversold confirmation gate from RSI (buy below buy_below, "
        "sell above sell_above).")
    def rsi_filter(period: int = 14, buy_below: int = 30,
                   sell_above: int = 70) -> dict:
        return {"period": period, "buy_below": buy_below,
                "sell_above": sell_above}

    @registry.register_primitive(
        "ATR_STOP",
        "Volatility-consistent stop-loss distance in ATR multiples.")
    def atr_stop(period: int = 14, multiplier: float = 1.5) -> dict:
        return {"period": period, "multiplier": multiplier}

    @registry.register_primitive(
        "ATR_TARGET",
        "Volatility-consistent take-profit distance in ATR multiples.")
    def atr_target(period: int = 14, multiplier: float = 2.0) -> dict:
        return {"period": period, "multiplier": multiplier}

    @registry.register_primitive(
        "BREAKOUT",
        "Close beyond the highest/lowest extreme of the prior N bars.")
    def breakout(n: int, direction: str = "long") -> dict:
        return {"n": n, "direction": direction}

    @registry.register_primitive(
        "LIQUIDITY_SWEEP",
        "Sweep of a prior swing high/low then rejection (stop hunt).")
    def liquidity_sweep(lookback: int = 20, direction: str = "long") -> dict:
        return {"lookback": lookback, "direction": direction}

    @registry.register_primitive(
        "CHoCH_MSS",
        "Break of market structure (CHoCH / MSS) confirming a reversal.")
    def choch_mss(lookback: int = 20, direction: str = "long") -> dict:
        return {"lookback": lookback, "direction": direction}

    @registry.register_primitive(
        "FVG_IMBALANCE",
        "Fair-value gap in the body of consecutive candles. For longs: "
        "candle3.low > candle1.high, gap zone = [candle1.high, candle3.low] "
        "(candle2 is the displacement); for shorts: candle3.high < candle1.low. "
        "Optional min_gap_atr filters gaps that are at least k x ATR(14) wide "
        "(gap_size = candle3.low - candle1.high for longs) so tiny imbalances "
        "do not count.")
    def fvg_imbalance(lookback: int = 3, direction: str = "long",
                      min_gap_atr: float = 0.0) -> dict:
        return {"lookback": lookback, "direction": direction,
                "min_gap_atr": min_gap_atr}

    @registry.register_primitive(
        "VWAP_BAND",
        "Volume-weighted average price as a bias / mean-reversion filter. "
        "period=0 means an anchored VWAP (session-style, from the start of the "
        "data); period>0 is a rolling VWAP over that many bars. Entry occurs "
        "when price trades to the VWAP band: longs on a pullback below VWAP by "
        "up to distance_atr x ATR then reclaim of VWAP, shorts mirrored.")
    def vwap_band(period: int = 0, distance_atr: float = 0.5,
                  direction: str = "long") -> dict:
        return {"period": period, "distance_atr": distance_atr,
                "direction": direction}

    @registry.register_primitive(
        "VOLUME_PROFILE",
        "Volume-profile breakout / value-area filter: build a volume profile "
        "over the lookback window and define the value area (the pct% of volume "
        "around the POC). Longs favour reactions off the value-area low / POC "
        "with acceptance back inside; shorts off the value-area high.")
    def volume_profile(lookback: int = 100, pct: float = 70.0,
                       direction: str = "long") -> dict:
        return {"lookback": lookback, "pct": pct, "direction": direction}

    @registry.register_primitive(
        "BULLISH_ENGULFING",
        "Two-candle reversal: a bullish candle whose body fully engulfs the "
        "prior bearish body, confirmed at a support / value level within the "
        "last `lookback` bars.")
    def bullish_engulfing(lookback: int = 3, direction: str = "long") -> dict:
        return {"lookback": lookback, "direction": direction}

    @registry.register_primitive(
        "RISK_PCT",
        "Fixed-fractional risk per trade as a fraction of equity.")
    def risk_pct(value: float = 0.01) -> dict:
        return {"value": value}

    return registry


DEFAULT_REGISTRY = build_default_registry()

# Hot-swappable: the single reference engine the whole pipeline uses. To add a
# new execution concept, register one function above and the schema menu + the
# validation vocabulary update instantly (IR.md: hot-swappable primitives).
IR_ENGINE = None  # imported/instantiated by consumers to avoid import cycles
