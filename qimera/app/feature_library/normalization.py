"""QiMera Normalization Utilities (QiMera.md §6.3 / M1).

Exposes scaling utilities to normalize features into standard [-1.0, +1.0] bounds.
"""

import numpy as np
import pandas as pd


def minmax_scale(series: pd.Series, min_val: float = None, max_val: float = None) -> pd.Series:
    """Scales a series to [-1.0, +1.0] bounds using min-max scaling."""
    if min_val is None:
        min_val = series.min()
    if max_val is None:
        max_val = series.max()
        
    denom = max_val - min_val
    if denom == 0:
        return pd.Series(0.0, index=series.index)
        
    return 2.0 * ((series - min_val) / denom) - 1.0


def tanh_scale(series: pd.Series, mean: float = None, std: float = None) -> pd.Series:
    """Scales a series to [-1.0, +1.0] using robust hyperbolic tangent scaling.
    
    y = tanh((x - mean) / (std * scale_factor))
    """
    if mean is None:
        mean = series.mean()
    if std is None:
        std = series.std()
        
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
        
    # Scale factor spreads out standard deviations across the tanh curve
    return pd.Series(np.tanh((series - mean) / (std * 1.5)), index=series.index)


def sigmoid_scale(series: pd.Series, mean: float = None, std: float = None) -> pd.Series:
    """Scales a series to [-1.0, +1.0] bounds using sigmoid mapping: 2 / (1 + exp(-z)) - 1."""
    if mean is None:
        mean = series.mean()
    if std is None:
        std = series.std()
        
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
        
    z = (series - mean) / std
    return 2.0 / (1.0 + np.exp(-z)) - 1.0


def clip_bounds(series: pd.Series, lower: float = -1.0, upper: float = 1.0) -> pd.Series:
    """Clips Series values to enforce hard [-1.0, +1.0] boundaries."""
    return series.clip(lower=lower, upper=upper)
