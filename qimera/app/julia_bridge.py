"""Python ↔ Julia interop bridge (QiMera.md §6.4 / M2).

Loads the precompiled ``QiMera`` Julia package (qimera/julia) in-process via
``juliacall`` and exposes its fitness/sweep functions to the API. The pure-Python
twins (``fitness_python`` / ``sweep_python``) are kept numerically identical to the
Julia originals so the benchmark endpoint can prove parity and measure the
speed-up — the exact validation M2 calls for.

Everything is lazy: if Julia or ``juliacall`` is absent the bridge reports
``available() == False`` and the API degrades gracefully instead of crashing.
"""

import shutil
import time
from pathlib import Path

import numpy as np

_JULIA_ROOT = Path(__file__).resolve().parent.parent / "julia"
_PROJECT = _JULIA_ROOT / "Project.toml"
_SRC = _JULIA_ROOT / "src" / "QiMera.jl"

_jl = None          # cached juliacall Main
_qimera = None      # cached QiMera julia module
_error = ""


class JuliaUnavailable(RuntimeError):
    pass


def julia_on_path() -> bool:
    return bool(shutil.which("julia"))


def available() -> bool:
    """True when both the Julia runtime and juliacall exist."""
    if not julia_on_path():
        return False
    try:
        import juliacall  # noqa: F401
    except ImportError:
        return False
    return True


def ensure():
    """Load the QiMera Julia module (idempotent). Raises JuliaUnavailable."""
    global _jl, _qimera, _error
    if _qimera is not None:
        return _qimera
    if not available():
        raise JuliaUnavailable(
            "Julia unavailable: install Julia (e.g. `winget install JulianLang.juliaup`) "
            "and `pip install juliacall`."
        )
    try:
        from juliacall import Main as jl
    except Exception as exc:
        _error = f"juliacall import failed: {exc}"
        raise JuliaUnavailable(_error) from exc
    _jl = jl
    try:
        jl.seval("using QiMera")
    except Exception:  # noqa: BLE001 - develop once, then use the module
        try:
            jl.seval(
                f'import Pkg; Pkg.develop(path="{_JULIA_ROOT.as_posix()}")'
            )
            jl.seval("using QiMera")
        except Exception as exc:
            _error = f"could not load QiMera Julia package: {exc}"
            raise JuliaUnavailable(_error) from exc
    _qimera = jl.QiMera
    return _qimera


def _call(fn, *args):
    """Resolve a QiMera.jl function and call it (reliable across juliacall)."""
    ensure()
    func = _jl.seval(f"QiMera.{fn}")
    return func(*args)


def _f64(a):
    """Python -> Julia float64 vector via numpy (typed, near-zero-copy)."""
    return np.asarray(list(a), dtype=np.float64)


def status() -> dict:
    out = {"available": available(), "project": str(_PROJECT), "error": _error}
    if available():
        try:
            from juliacall import Main as jl
            out["julia_version"] = str(jl.VERSION)
            ensure()
            out["module_loaded"] = True
            out["functions"] = []
            for name in ("fitness", "sweep_stats", "mc_equities"):
                try:
                    jl.seval(f"QiMera.{name}")
                    out["functions"].append(name)
                except Exception:  # noqa: BLE001, S110 - best-effort probe
                    pass
        except JuliaUnavailable as exc:
            out["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{exc}"
    return out


# ---------------------------------------------------------------------------
# Julia calls
# ---------------------------------------------------------------------------

def fitness(close, high, low, ema_len, atr_len, atr_mult):
    """Julia fitness (vector metrics), Python floats in -> list[float] out."""
    res = _call("fitness", _f64(close), _f64(high), _f64(low),
                int(ema_len), int(atr_len), float(atr_mult))
    return [float(x) for x in res]


def sweep_stats(close, lookback):
    """Julia rolling min/max -> (list, list)."""
    mn, mx = _call("sweep_stats", _f64(close), int(lookback))
    return [float(x) for x in mn], [float(x) for x in mx]


def mc_equities(pnls, iterations: int, seed: int, slippage_std: float = 0.0) -> np.ndarray:
    """Julia MC null distribution (M4): total equity per iteration under random
    sign-flips of the realized trade PnLs + slippage noise (edge-vs-luck null;
    a plain shuffle would leave the equity sum invariant)."""
    res = _call("mc_equities", _f64(pnls), int(iterations), int(seed),
                float(slippage_std))
    return np.asarray([float(x) for x in res])


def mc_equities_python(pnls, iterations: int, seed: int, slippage_std: float = 0.0) -> np.ndarray:
    """Pure-Python twin of ``mc_equities`` — statistically equivalent (same
    shape and mean; RNG draws differ from Julia by design)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(pnls), dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(iterations)
    out = np.empty(iterations)
    for it in range(iterations):
        signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        s = (arr * signs).sum()
        if slippage_std > 0.0:
            s += rng.normal(0.0, slippage_std, n).sum()
        out[it] = s
    return out


# ---------------------------------------------------------------------------
# Python reference twins (must stay numerically identical to the Julia code)
# ---------------------------------------------------------------------------

def fitness_python(close, high, low, ema_len, atr_len, atr_mult):
    n = len(close)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    k = 2.0 / (ema_len + 1)
    e = [0.0] * n
    e[0] = close[0]
    for i in range(1, n):
        e[i] = e[i - 1] + k * (close[i] - e[i - 1])

    prev_close = close[0]
    a = [0.0] * n
    rma = 0.0
    for i in range(n):
        tr = max(high[i] - low[i],
                 abs(high[i] - prev_close),
                 abs(low[i] - prev_close))
        if i == 0:
            rma = tr
        elif i < atr_len:
            rma = (rma * i + tr) / (i + 1)
        else:
            rma = (rma * (atr_len - 1) + tr) / atr_len
        a[i] = rma
        prev_close = close[i]

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    entry = 0.0
    stop = 0.0
    for i in range(1, n):
        if entry == 0.0 and close[i - 1] <= e[i - 1] and close[i] > e[i]:
            entry = close[i]
            stop = entry - atr_mult * a[i]
            continue
        if entry != 0.0:
            cand = close[i] - atr_mult * a[i]
            stop = max(stop, cand)
            if close[i] < stop:
                pnl = close[i] - entry
                equity += pnl
                trades += 1
                if pnl >= 0:
                    wins += 1
                    gross_win += pnl
                else:
                    gross_loss -= pnl
                entry = 0.0
        if equity > peak:
            peak = equity
        elif peak > 0.0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    # max_dd is the running maximum drawdown from peak (mirrors QiMera.jl).
    if gross_loss > 0.0:
        pf = min(gross_win / gross_loss, 1.0e9)
    elif gross_win > 0.0:
        pf = 1.0e9
    else:
        pf = 0.0
    win_rate = wins / trades if trades > 0 else 0.0
    return [equity, win_rate, pf, max_dd, float(trades)]


def sweep_python(close, lookback):
    n = len(close)
    mn = [0.0] * n
    mx = [0.0] * n
    for i in range(n):
        lo = close[i]
        hi = close[i]
        for j in range(max(0, i - lookback), i):
            c = close[j]
            lo = min(lo, c)
            hi = max(hi, c)
        mn[i] = lo
        mx[i] = hi
    return mn, mx


# ---------------------------------------------------------------------------
# Benchmark (M2: Julia vs pure-Python, parity asserted)
# ---------------------------------------------------------------------------

def _synthetic_close(n, seed=7):
    import random

    rng = random.Random(seed)
    price = 100.0
    out = []
    for _ in range(n):
        price *= 1.0 + rng.gauss(0.0, 0.0015)
        out.append(price)
    return out


def _synthetic_ohlc(close):
    high = [c * 1.002 for c in close]
    low = [c * 0.998 for c in close]
    return high, low


def benchmark(bars: int = 20000, lookback: int = 200, runs: int = 3) -> dict:
    """Time Julia vs pure Python on the O(n·lookback) sweep; assert parity."""
    close = _synthetic_close(bars)
    mn_jl, mx_jl = [], []
    result = {
        "bars": bars,
        "lookback": lookback,
        "runs": runs,
        "parity_fitness": None,
        "parity_sweep": False,
        "julia_sweep_s": None,
        "python_sweep_s": None,
        "speedup": None,
        "fitness": None,
    }

    # fitness parity on the same data (deterministic values, not timing)
    high, low = _synthetic_ohlc(close)
    params = (50, 14, 3.0)
    fit_jl = fitness(close, high, low, *params)
    fit_py = fitness_python(close, high, low, *params)
    result["parity_fitness"] = all(
        abs(x - y) < 1e-9 for x, y in zip(fit_jl, fit_py)
    )
    result["fitness"] = {
        "net_pnl": fit_jl[0], "win_rate": fit_jl[1],
        "profit_factor": fit_jl[2], "max_drawdown_pct": fit_jl[3],
        "trades": fit_jl[4],
    }

    # warm-up (first Julia call also absorbs compilation)
    mn_jl, mx_jl = sweep_stats(close, lookback)
    mn_py, mx_py = sweep_python(close, lookback)
    result["parity_sweep"] = (len(mn_jl) == len(mn_py)
                              and all(abs(x - y) < 1e-9 for x, y in
                                      zip(mn_jl, mn_py))
                              and all(abs(x - y) < 1e-9 for x, y in
                                      zip(mx_jl, mx_py)))

    t_jl = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        sweep_stats(close, lookback)
        t_jl = min(t_jl, time.perf_counter() - t0)
    t_py = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        sweep_python(close, lookback)
        t_py = min(t_py, time.perf_counter() - t0)

    result["julia_sweep_s"] = round(t_jl, 6)
    result["python_sweep_s"] = round(t_py, 6)
    result["speedup"] = round(t_py / t_jl, 2) if t_jl > 0 else None
    return result