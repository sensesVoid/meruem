"""QiMera Structural Building Blocks (QiMera.md §6.3 / M1).

Implements higher-level market structures (Fair Value Gaps, VWAP, Order Blocks,
and Liquidity Sweeps) as normalized feature expressions.
"""

import pandas as pd
import numpy as np
from qimera.app.feature_library.normalization import tanh_scale, minmax_scale
from qimera.app.feature_library.primitives import compute_atr


def detect_fvgs(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int = 14) -> pd.Series:
    """Detects Fair Value Gaps (FVG) and expresses them relative to ATR.
    
    - Bullish FVG (Bullish displacement): low[i] > high[i-2].
    - Bearish FVG (Bearish displacement): high[i] < low[i-2].
    
    Returns a series bounded to [-1.0, +1.0] containing the gap size relative to ATR.
    """
    n = len(close)
    gap_series = pd.Series(0.0, index=close.index)
    atr = compute_atr(high, low, close, atr_period)
    
    h_vals = high.values
    l_vals = low.values
    atr_vals = atr.values
    
    for i in range(2, n):
        # Bullish gap
        if l_vals[i] > h_vals[i - 2]:
            gap = l_vals[i] - h_vals[i - 2]
            denom = atr_vals[i] if atr_vals[i] > 0 else 1.0
            gap_series.iloc[i] = gap / denom
            
        # Bearish gap
        elif h_vals[i] < l_vals[i - 2]:
            gap = l_vals[i - 2] - h_vals[i]
            denom = atr_vals[i] if atr_vals[i] > 0 else 1.0
            gap_series.iloc[i] = -gap / denom
            
    return tanh_scale(gap_series)


def compute_anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, session_boundary: pd.Series) -> pd.Series:
    """Computes Anchored VWAP resetting at each session boundary.
    
    Returns the percentage distance of close price from the VWAP line,
    normalized to [-1.0, +1.0] using tanh.
    """
    n = len(close)
    typical_price = (high + low + close) / 3.0
    
    vwap_values = np.zeros(n)
    
    cum_pv = 0.0
    cum_vol = 0.0
    
    tp_arr = typical_price.values
    vol_arr = volume.values
    boundary_arr = session_boundary.values
    
    for i in range(n):
        if boundary_arr[i]:
            cum_pv = 0.0
            cum_vol = 0.0
            
        cum_pv += tp_arr[i] * vol_arr[i]
        cum_vol += vol_arr[i]
        
        if cum_vol > 0:
            vwap_values[i] = cum_pv / cum_vol
        else:
            vwap_values[i] = tp_arr[i]
            
    vwap_series = pd.Series(vwap_values, index=close.index)
    
    # Calculate % distance of price from VWAP
    distance = (close - vwap_series) / vwap_series
    return tanh_scale(distance)


def detect_liquidity_sweeps(high: pd.Series, low: pd.Series, close: pd.Series,
                            lookback: int = 20) -> pd.Series:
    """Detects liquidity sweeps over a rolling lookback window.
    
    A high sweep occurs when high[i] > max(high[i-lookback:i]) but close[i] < max(high[i-lookback:i]).
    A low sweep occurs when low[i] < min(low[i-lookback:i]) but close[i] > min(low[i-lookback:i]).
    
    Returns a series normalized to [-1.0, +1.0] representing sweep strength.
    """
    n = len(close)
    sweep_signals = pd.Series(0.0, index=close.index)
    
    h_vals = high.values
    l_vals = low.values
    c_vals = close.values
    
    for i in range(lookback, n):
        window_highs = h_vals[i - lookback : i]
        window_lows = l_vals[i - lookback : i]
        
        max_high = np.max(window_highs)
        min_low = np.min(window_lows)
        
        # High sweep (swept buy-side liquidity)
        if h_vals[i] > max_high and c_vals[i] < max_high:
            sweep_signals.iloc[i] = (h_vals[i] - max_high) / (max_high - min_low if max_high > min_low else 1.0)
            
        # Low sweep (swept sell-side liquidity)
        elif l_vals[i] < min_low and c_vals[i] > min_low:
            sweep_signals.iloc[i] = -(min_low - l_vals[i]) / (max_high - min_low if max_high > min_low else 1.0)
            
    return tanh_scale(sweep_signals)


def order_blocks_proximity(high: pd.Series, low: pd.Series, close: pd.Series,
                           volume: pd.Series, period: int = 20) -> pd.Series:
    """Computes proximity to historical order blocks (OB) with freshness decay.
    
    An Order Block is detected on high-volume swing pivots. We track these zones
    and compute price proximity (scaled by ATR) to the nearest active zone.
    
    Returns a series normalized to [-1.0, +1.0].
    """
    n = len(close)
    proximity = pd.Series(0.0, index=close.index)
    atr = compute_atr(high, low, close, 14)
    
    h_arr = high.values
    l_arr = low.values
    c_arr = close.values
    v_arr = volume.values
    atr_arr = atr.values
    
    # Track active zones: list of {"type": "bull"|"bear", "level": float, "age": int}
    active_zones = []
    
    # Typical threshold for high-volume OB identification (e.g. volume > rolling average)
    avg_vol = volume.rolling(window=period).mean().values
    
    for i in range(1, n):
        # Update zone ages
        for zone in active_zones:
            zone["age"] += 1
            
        # Filter expired or mitigated zones (e.g., price crossed/mitigated the zone)
        mitigated_indices = []
        for idx, zone in enumerate(active_zones):
            if zone["type"] == "bull" and c_arr[i] < zone["level"]:  # Bull OB invalidated
                mitigated_indices.append(idx)
            elif zone["type"] == "bear" and c_arr[i] > zone["level"]:  # Bear OB invalidated
                mitigated_indices.append(idx)
            elif zone["age"] > 100:  # Freshness decay limit
                mitigated_indices.append(idx)
                
        active_zones = [z for idx, z in enumerate(active_zones) if idx not in mitigated_indices]
        
        # Detect new zone: high-volume engulfing or pivot bars
        is_high_volume = v_arr[i] > (avg_vol[i] * 1.5) if not np.isnan(avg_vol[i]) else False
        
        if is_high_volume:
            if c_arr[i] > h_arr[i - 1] and c_arr[i - 1] < c_arr[i]:  # Engulfing Up
                active_zones.append({"type": "bull", "level": l_arr[i - 1], "age": 0})
            elif c_arr[i] < l_arr[i - 1] and c_arr[i - 1] > c_arr[i]:  # Engulfing Down
                active_zones.append({"type": "bear", "level": h_arr[i - 1], "age": 0})
                
        # Calculate proximity score to nearest active zone
        if active_zones:
            distances = []
            for zone in active_zones:
                dist = c_arr[i] - zone["level"]
                # Apply decay weight
                decay = np.exp(-0.02 * zone["age"])
                atr_val = atr_arr[i] if atr_arr[i] > 0 else 1.0
                distances.append((dist / atr_val) * decay)
                
            # Take the closest zone distance
            closest_idx = np.argmin([abs(d) for d in distances])
            proximity.iloc[i] = distances[closest_idx]
            
    return tanh_scale(proximity)
