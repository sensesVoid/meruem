"""QiMera Base Primitives (QiMera.md §6.3 / M1).

Base technical primitives used as the vocabulary inputs for typed GP and Agentic Synthesis.
All primitives produce series normalized to [-1.0, +1.0].
"""

import pandas as pd
import numpy as np
from qimera.app.feature_library.normalization import tanh_scale, minmax_scale


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Computes standard Average True Range (Wilder's smoothing).
    
    Returns raw (positive float) series.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def atr_normalized(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Normalized ATR representing relative volatility (ATR divided by price).
    
    Scaled into [-1.0, +1.0] using tanh.
    """
    atr = compute_atr(high, low, close, period)
    relative_atr = atr / close
    return tanh_scale(relative_atr)


def rolling_volume_norm(volume: pd.Series, period: int = 20) -> pd.Series:
    """Normalizes volume relative to its rolling average.
    
    y = volume / rolling_average_volume - 1.0, scaled using tanh.
    """
    ma_vol = volume.rolling(window=period).mean()
    norm_vol = (volume / ma_vol) - 1.0
    norm_vol.iloc[:period] = 0.0  # fill early bars
    return tanh_scale(norm_vol)


def swing_pivots(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5) -> tuple[pd.Series, pd.Series]:
    """Detects swing high and swing low pivot points.
    
    A bar is a swing high if its high is greater than the left & right window highs.
    Returns two series:
      - swing_high: +1.0 at pivot bar, -1.0 elsewhere.
      - swing_low: -1.0 at pivot bar, +1.0 elsewhere.
    """
    n = len(high)
    sh = pd.Series(-1.0, index=high.index)
    sl = pd.Series(1.0, index=low.index)
    
    h_arr = high.values
    l_arr = low.values
    
    for i in range(left, n - right):
        val_h = h_arr[i]
        val_l = l_arr[i]
        
        # Check high pivot
        is_high = True
        for j in range(i - left, i):
            if h_arr[j] >= val_h:
                is_high = False
                break
        if is_high:
            for j in range(i + 1, i + right + 1):
                if h_arr[j] > val_h:
                    is_high = False
                    break
        if is_high:
            sh.iloc[i] = 1.0
            
        # Check low pivot
        is_low = True
        for j in range(i - left, i):
            if l_arr[j] <= val_l:
                is_low = False
                break
        if is_low:
            for j in range(i + 1, i + right + 1):
                if l_arr[j] < val_l:
                    is_low = False
                    break
        if is_low:
            sl.iloc[i] = -1.0
            
    return sh, sl


def price_momentum_norm(close: pd.Series, period: int = 14) -> pd.Series:
    """Returns normalized momentum representing normalized price returns.
    
    Scaled into [-1.0, +1.0] using tanh.
    """
    returns = close.pct_change(periods=period)
    returns.fillna(0.0, inplace=True)
    return tanh_scale(returns)
