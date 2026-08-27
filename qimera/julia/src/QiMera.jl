# QiMera — Julia compute core (MiQ 2: Python/Julia interop).
#
# Stdlib-only by design: no external Julia packages, so the package
# precompiles fast and works fully offline. juliacall (PythonCall) drives it
# from Python for an in-process, precompiled fitness path — the pattern the
# Parametric GP (QiMera.md §6.7) will call millions of times.
#
# The Python twin of these functions lives in
# qimera/app/julia_bridge.py (fitness_python / sweep_python) and must stay
# numerically identical — the benchmark route asserts parity between the two.

module QiMera

using Random  # stdlib: MersenneTwister / shuffle for the MC bootstrap

export ema, atr_wilder, fitness, sweep_stats, mc_equities

"""
    ema(values, len)

Standard exponential moving average (seed = first value, smoothing k = 2/(len+1)).
"""
function ema(values::Vector{Float64}, len::Int)
    n = length(values)
    out = Vector{Float64}(undef, n)
    n == 0 && return out
    k = 2.0 / (len + 1)
    acc = values[1]
    out[1] = acc
    @inbounds for i in 2:n
        acc += k * (values[i] - acc)
        out[i] = acc
    end
    out
end

"""
    atr_wilder(high, low, close, len)

True Range running mean (Wilder): TR over the first `len` bars is a simple
mean, then Wilder smoothing. Mirrored exactly by the Python bridge.
"""
function atr_wilder(high::Vector{Float64}, low::Vector{Float64},
                    close::Vector{Float64}, len::Int)
    n = length(close)
    out = zeros(Float64, n)
    n == 0 && return out
    prev_close = close[1]
    rma = 0.0
    @inbounds for i in 1:n
        tr = max(high[i] - low[i],
                 abs(high[i] - prev_close),
                 abs(low[i] - prev_close))
        if i == 1
            rma = tr
        elseif i <= len
            rma = (rma * (i - 1) + tr) / i
        else
            rma = (rma * (len - 1) + tr) / len
        end
        out[i] = rma
        prev_close = close[i]
    end
    out
end

"""
    fitness(close, high, low, ema_len, atr_len, atr_mult)

One long-only strategy simulation — the shape of a Parametric-GP fitness call.
Entry on an EMA cross above (from below), ratcheting trailing stop of
`atr_mult` × ATR. Returns `[net_pnl, win_rate, profit_factor, max_drawdown_pct,
trades]` (profit factor capped at 1e9 to stay JSON-safe). `max_drawdown_pct`
is the *running maximum* drawdown from peak equity over the whole run — not
just the terminal drawdown — so the GP hurdle gate actually bounds drawdowns.

Accepts any `AbstractVector{Float64}` (PythonCall hands numpy arrays in as
PyArray) and copies to a real `Vector` once at the boundary so the inner loops
stay allocation-free and fast.
"""
function fitness(close::AbstractVector{Float64}, high::AbstractVector{Float64},
                 low::AbstractVector{Float64}, ema_len::Int, atr_len::Int,
                 atr_mult::Float64)
    close = Vector{Float64}(close)
    high = Vector{Float64}(high)
    low = Vector{Float64}(low)
    n = length(close)
    n == 0 && return [0.0, 0.0, 0.0, 0.0, 0.0]
    e = ema(close, ema_len)
    a = atr_wilder(high, low, close, atr_len)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    entry = 0.0   # long entry price; 0.0 = flat
    stop = 0.0
    @inbounds for i in 2:n
        if entry == 0.0 && close[i - 1] <= e[i - 1] && close[i] > e[i]
            entry = close[i]
            stop = entry - atr_mult * a[i]
            continue
        end
        if entry != 0.0
            cand = close[i] - atr_mult * a[i]
            if cand > stop
                stop = cand
            end
            if close[i] < stop
                pnl = close[i] - entry
                equity += pnl
                trades += 1
                if pnl >= 0
                    wins += 1
                    gross_win += pnl
                else
                    gross_loss -= pnl
                end
                entry = 0.0
            end
        end
        if equity > peak
            peak = equity
        elseif peak > 0.0
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd
                max_dd = dd
            end
        end
    end
    pf = if gross_loss > 0.0
        min(gross_win / gross_loss, 1.0e9)
    elseif gross_win > 0.0
        1.0e9
    else
        0.0
    end
    win_rate = trades > 0 ? wins / trades : 0.0
    [equity, win_rate, pf, max_dd, Float64(trades)]
end

"""
    sweep_stats(close, lookback)

Rolling min/max of `close` over the previous `lookback` bars (inclusive of the
current bar), per bar. Deliberately an O(n·lookback) nested loop — the exact
shape of pure-Python hot loop the benchmark highlights Julia against
(liquidity-sweep / swing-level style rolling features). Accepts
`AbstractVector{Float64}` (numpy in) and copies once to a real `Vector`.
"""
function sweep_stats(close::AbstractVector{Float64}, lookback::Int)
    close = Vector{Float64}(close)
    n = length(close)
    mn = zeros(Float64, n)
    mx = zeros(Float64, n)
    lookback > 0 || return mn, mx
    @inbounds for i in 1:n
        lo_start = max(1, i - lookback)
        lo = close[i]
        hi = close[i]
        for j in (lo_start):(i - 1)
            c = close[j]
            if c < lo
                lo = c
            end
            if c > hi
                hi = c
            end
        end
        mn[i] = lo
        mx[i] = hi
    end
    mn, mx
end

"""
    mc_equities(pnls, iterations, seed, slippage_std)

Monte-Carlo *null* distribution (QiMera.md §6.5 / M4): each realized trade PnL's
sign is randomly flipped (p = 0.5) with Gaussian slippage noise added, and the
total equity for that iteration is returned. This answers "would this equity be
surprising under random trade signs?" — the classic edge-vs-luck test (a plain
order shuffle would leave the sum invariant and test nothing). One in-process
Julia call replaces tens of thousands of Python loops; the pure-Python twin
(julia_bridge.mc_equities_python) matches statistically (same shape/mean; RNG
draws differ by design).
"""
function mc_equities(pnls::AbstractVector{Float64}, iterations::Int, seed::Int,
                     slippage_std::Float64)
    pnls = Vector{Float64}(pnls)
    n = length(pnls)
    out = Vector{Float64}(undef, iterations)
    n == 0 && return zeros(Float64, iterations)
    rng = MersenneTwister(seed)
    @inbounds for it in 1:iterations
        s = 0.0
        for i in 1:n
            p = pnls[i]
            if rand(rng) < 0.5
                p = -p
            end
            if slippage_std > 0.0
                p += randn(rng) * slippage_std
            end
            s += p
        end
        out[it] = s
    end
    out
end

end # module QiMera