"""QiMera Shared Feature Library (QiMera.md §6.3 / M1).

Exposes technical primitives, structural building blocks, and normalization functions.
"""

from .normalization import minmax_scale, tanh_scale, sigmoid_scale, clip_bounds
from .primitives import (
    compute_atr,
    atr_normalized,
    rolling_volume_norm,
    swing_pivots,
    price_momentum_norm
)
from .building_blocks import (
    detect_fvgs,
    compute_anchored_vwap,
    detect_liquidity_sweeps,
    order_blocks_proximity
)
