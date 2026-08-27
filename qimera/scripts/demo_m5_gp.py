"""Quick demo of the M5 multi-indicator GP on synthetic FX data."""
import numpy as np
import pandas as pd

from qimera.app.gp_indicators import run_multi_gp, GPConfig

np.random.seed(7)
n = 800
close = 1.09 + np.cumsum(np.random.normal(0.0002, 0.001, n))
op = close + np.random.normal(0, 0.0005, n)
hi = np.maximum(op, close) + np.random.uniform(0, 0.001, n)
lo = np.minimum(op, close) - np.random.uniform(0, 0.001, n)
df = pd.DataFrame({
    "time": np.arange(n) * 900 + 1690000000,
    "open": op, "high": hi, "low": lo, "close": close,
    "volume": np.random.uniform(10, 1000, n),
})

res = run_multi_gp(df, GPConfig(population=30, generations=10, seed=11, min_trades=3))
print("evals:", res["evals"], "| gated:", res["gated"], "| accepted:", res["accepted"])
b = res["best"]
print("champion indicator:", b["indicator"])
print("champion genes [ind,p0,p1,p2,atr_len,atr_mult]:", b["genes"])
print("champion metrics:", {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                               for k, v in b["metrics"].items()})
print("final gen avg/max fitness:", round(res["logbook"][-1]["avg"], 4), "/",
      round(res["logbook"][-1]["max"], 4))
