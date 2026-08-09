import { useCallback, useEffect, useRef, useState } from "react";
import {
  CandlestickData,
  CandlestickSeries,
  createChart,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  MouseEventParams,
  UTCTimestamp,
} from "lightweight-charts";
import { Card } from "@/components/ui/card";
import { api, Asset } from "@/lib/api";
import { cn, usePolling } from "@/lib/utils";

export interface Timeframe {
  label: string;
  sec: number;
}

export const TIMEFRAMES: Timeframe[] = [
  { label: "M1", sec: 60 },
  { label: "5m", sec: 300 },
  { label: "15m", sec: 900 },
  { label: "1h", sec: 3600 },
  { label: "4h", sec: 14400 },
  { label: "1D", sec: 86400 },
  { label: "1W", sec: 604800 },
  { label: "1M", sec: 2592000 },
];

const TICK_MS = 2000;
const MAX_BARS = 10000;
const DEFAULT_BARS = 1500;
const VISIBLE_BARS = 120;

function seedPrice(symbol: string): number {
  let h = 0;
  for (const ch of symbol) h = (h * 31 + ch.charCodeAt(0)) % 997;
  return 1 + (h / 997) * 12000;
}

const CRYPTO_BASES = new Set([
  "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "LINK",
  "AVAX", "UNI", "SHIB", "TRX", "XLM", "ATOM", "NEAR", "APT", "ARB", "OP",
  "SUI", "SEI", "PEPE", "BNB", "TON", "INJ",
]);

function priceDecimals(symbol: string): number {
  const base = symbol.replace(/USDT$|USDC$|USD$/, "");
  if (CRYPTO_BASES.has(base)) return 1;
  if (["XAU", "XAG", "XPT", "XPD"].includes(base)) return 2; // metals
  if (/^[A-Za-z]{6}$/.test(symbol)) return 5; // FX pairs (EURUSD, GBPUSD, ...)
  return 2;
}

interface LiveChartProps {
  symbol: string;
  barSeconds: number;
  bars: number;
  timeframe: string;
  onTick?: (price: number) => void;
  onProvider?: (provider: string) => void;
  /** Called with true while the market is open (last candle is current);
   *  false when the last candle is stale (weekend / closed market). */
  onOpenChange?: (open: boolean) => void;
  resetKey?: number;
  positions?: ChartPositionOverlay[];
}

interface ChartPositionOverlay {
  id: string;
  direction: string;
  entry_price: number;
  sl: number;
  managed_sl: number;
  tp1?: number | null;
  tp2?: number | null;
  tp3?: number | null;
  tp4?: number | null;
  lot_size?: number;
  opened_at?: string;
  last_price?: number | null;
}

interface CandleResponse {
  symbol: string;
  timeframe: string;
  provider: string;
  candles: { time: number; open: number; high: number; low: number; close: number }[];
}

function LiveChart({ symbol, barSeconds, bars, timeframe, onTick, onProvider, onOpenChange, resetKey, positions }: LiveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lastPriceRef = useRef(0);
  const priceLinesRef = useRef<{ [key: string]: IPriceLine }>({});
  const lineOverlayRef = useRef<Map<IPriceLine, ChartPositionOverlay>>(new Map());
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const hoverLineRef = useRef<IPriceLine | null>(null);
  const prevKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      layout: {
        background: { color: "transparent" },
        textColor: "#7c8aa0",
      },
      grid: {
        vertLines: { color: "#1f2732" },
        horzLines: { color: "#1f2732" },
      },
      timeScale: { borderColor: "#1f2732", timeVisible: true, rightOffset: 8 },
      rightPriceScale: { borderColor: "#1f2732" },
      // Explicitly enable pan & zoom (drag, wheel, pinch, touch).
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
      kineticScroll: {
        mouse: true,
        touch: true,
      },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });
    chartRef.current = chart;
    seriesRef.current = series;

    // Marker tooltips: hover near an overlay price line (entry/SL/TP) to see
    // the live-trade details (opened at, lot size, managed SL, last price).
    chart.subscribeCrosshairMove((param: MouseEventParams) => {
      const el = tooltipRef.current;
      const c = chartRef.current;
      const s = seriesRef.current;
      if (!el || !c || !s) return;
      // Find the overlay price line (entry/SL/TP) under the cursor.
      const line =
        param.point != null
          ? Object.values(priceLinesRef.current).find((l) => {
              const y = s.priceToCoordinate(l.options().price);
              return y !== null && Math.abs(y - param.point!.y) < 8;
            }) ?? null
          : null;
      if (line) {
        const p = lineOverlayRef.current.get(line);
        if (p) {
          hoverLineRef.current = line;
          const y = s.priceToCoordinate(line.options().price) ?? 40;
          showOverlayTooltip(p, param.point!.x, y);
          return;
        }
      }
      if (hoverLineRef.current) {
        hoverLineRef.current = null;
        el.style.display = "none";
      }
    });

    const onResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Render open-trade overlays: entry marker + SL/TP price lines. The managed
  // SL line tracks the trailing stop / breakeven projection live.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series || !chartRef.current) return;
    // Remove stale price lines.
    for (const line of Object.values(priceLinesRef.current)) {
      series.removePriceLine(line);
    }
    priceLinesRef.current = {};
    lineOverlayRef.current.clear();
    hoverLineRef.current = null;
    if (tooltipRef.current) tooltipRef.current.style.display = "none";

    const overlays = (positions ?? []).filter((p) => p && Number.isFinite(p.entry_price));
    if (overlays.length === 0) return;

    overlays.forEach((p) => {
      const tps = [p.tp1, p.tp2, p.tp3, p.tp4].filter(
        (x): x is number => typeof x === "number" && x > 0,
      );
      const key = p.id;
      const entryLine = series.createPriceLine({
        price: p.entry_price,
        color: "#8b5cf6",
        lineWidth: 1,
        lineStyle: 1 /* LineStyle.Dashed */,
        axisLabelVisible: true,
        title: `ENTRY ${p.direction === "BUY" ? "▲" : "▼"}`,
      });
      priceLinesRef.current[`${key}:entry`] = entryLine;
      lineOverlayRef.current.set(entryLine, p);
      const slLine = series.createPriceLine({
        price: p.managed_sl,
        color: p.direction === "BUY" ? "#ef4444" : "#22c55e",
        lineWidth: 1,
        lineStyle: 2 /* LineStyle.Dotted */,
        axisLabelVisible: true,
        title: "SL",
      });
      priceLinesRef.current[`${key}:sl`] = slLine;
      lineOverlayRef.current.set(slLine, p);
      tps.forEach((tp, i) => {
        const line = series.createPriceLine({
          price: tp,
          color: "#22c55e",
          lineWidth: 1,
          lineStyle: 1 /* LineStyle.Dashed */,
          axisLabelVisible: true,
          title: `TP${i + 1}`,
        });
        priceLinesRef.current[`${key}:tp${i + 1}`] = line;
        lineOverlayRef.current.set(line, p);
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positions, symbol, barSeconds]);

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;
    let cancelled = false;
    let tickId: ReturnType<typeof setInterval> | undefined;
    let candles: CandlestickData[] = [];

    const buildSimulated = (): CandlestickData[] => {
      const rows: CandlestickData[] = [];
      let price = seedPrice(symbol);
      const now = Math.floor(Date.now() / 1000);
      for (let i = 0; i < bars; i++) {
        const delta = (Math.random() - 0.48) * price * 0.01;
        const open = price;
        const close = open + delta;
        const high = Math.max(open, close) + Math.random() * price * 0.004;
        const low = Math.min(open, close) - Math.random() * price * 0.004;
        rows.push({
          time: (now - (bars - i) * barSeconds) as UTCTimestamp,
          open,
          high,
          low,
          close,
        });
        price = close;
      }
      return rows;
    };

    const marketOpen = (): boolean => {
      const last = candles[candles.length - 1];
      if (!last) return false;
      // The market is open when the newest candle is (roughly) the current bar.
      // When the last candle is stale (weekend / market closed) we must NOT
      // fabricate movement — the price is frozen at the last real close.
      const t = Math.floor(Date.now() / 1000);
      return t - (last.time as number) < barSeconds * 2;
    };

    const startTick = () => {
      const last = candles[candles.length - 1];
      lastPriceRef.current = last?.close ?? 0;
      // Synthetic tick animation runs ONLY while the market is open (the last
      // candle is current). On a closed market the chart + price stay frozen on
      // the real last close instead of showing a fake random walk.
      if (!marketOpen()) {
        onOpenChange?.(false);
        onTick?.(lastPriceRef.current);
        return;
      }
      onOpenChange?.(true);
      tickId = setInterval(() => {
        if (!seriesRef.current) return;
        if (document.hidden) return; // pause synthetic ticks in background tabs
        const cur = candles[candles.length - 1];
        const base = lastPriceRef.current;
        const next = Math.max(0.0001, base + (Math.random() - 0.5) * base * 0.003);
        const t = Math.floor(Date.now() / 1000);
        if (t - (cur.time as number) >= barSeconds) {
          const candle: CandlestickData = {
            time: t as UTCTimestamp,
            open: base,
            high: Math.max(base, next),
            low: Math.min(base, next),
            close: next,
          };
          candles.push(candle);
          if (candles.length > bars) candles.shift();
          seriesRef.current.update(candle);
        } else {
          seriesRef.current.update({
            time: cur.time,
            open: cur.open,
            high: Math.max(cur.high, next),
            low: Math.min(cur.low, next),
            close: next,
          });
          candles[candles.length - 1] = {
            ...cur,
            high: Math.max(cur.high, next),
            low: Math.min(cur.low, next),
            close: next,
          };
        }
        lastPriceRef.current = next;
        onTick?.(next);
      }, TICK_MS);
    };

    const applyData = () => {
      const series = seriesRef.current;
      const chart = chartRef.current;
      if (!series || !chart || candles.length === 0) return;
      const seen = new Set<number>();
      candles = candles.filter((c) => {
        const t = c.time as number;
        if (seen.has(t)) return false;
        seen.add(t);
        return true;
      });
      if (candles.length === 0) return;

      const symbolSwitch =
        prevKeyRef.current === null || prevKeyRef.current.split("|")[0] !== symbol;
      const prevVisible = symbolSwitch ? null : chart.timeScale().getVisibleRange();

      lastPriceRef.current = candles[candles.length - 1].close;
      try {
        series.setData(candles);
      } catch {
        series.setData([]);
        series.setData(candles);
      }

      if (symbolSwitch) {
        // New instrument: force the price axis to refit this symbol's magnitude
        // and land on the latest N bars.
        chart.priceScale("right").applyOptions({ autoScale: true });
        chart.timeScale().setVisibleLogicalRange({
          from: Math.max(0, candles.length - VISIBLE_BARS),
          to: candles.length + 4,
        });
      } else if (
        prevVisible &&
        typeof prevVisible.from === "number" &&
        typeof prevVisible.to === "number"
      ) {
        // Same instrument, different timeframe: keep the previously visible
        // time window instead of snapping to a brand-new bar range, so the
        // switch renders seamlessly without a jarring refit.
        const first = candles[0].time as number;
        const last = candles[candles.length - 1].time as number;
        const from = Math.max(prevVisible.from, first);
        const to = Math.min(prevVisible.to, last);
        if (from <= to) {
          chart.timeScale().setVisibleRange({
            from: from as UTCTimestamp,
            to: to as UTCTimestamp,
          });
        } else {
          chart.timeScale().setVisibleRange({
            from: last as UTCTimestamp,
            to: last as UTCTimestamp,
          });
        }
      } else {
        chart.timeScale().setVisibleLogicalRange({
          from: Math.max(0, candles.length - VISIBLE_BARS),
          to: candles.length + 4,
        });
      }

      prevKeyRef.current = `${symbol}|${barSeconds}`;
      onTick?.(lastPriceRef.current);
      startTick();
    };

    (async () => {
      try {
        const res = await api.get<CandleResponse>("/candles", {
          symbol,
          timeframe,
          limit: Math.min(bars, 1000),
        });
        if (cancelled) return;
        if (res.candles && res.candles.length > 1 && res.candles[0].close > 0) {
          candles = res.candles.map((c) => ({
            time: c.time as UTCTimestamp,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }));
          onProvider?.(res.provider);
        } else {
          candles = buildSimulated();
          onProvider?.("");
        }
      } catch {
        if (cancelled) return;
        candles = buildSimulated();
        onProvider?.("");
      }
      applyData();
    })();

    return () => {
      cancelled = true;
      if (tickId) clearInterval(tickId);
      lastPriceRef.current = 0;
    };
  }, [symbol, barSeconds, bars, timeframe, onTick, onProvider, onOpenChange]);

  useEffect(() => {
    if (resetKey && chartRef.current) {
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: bars - VISIBLE_BARS,
        to: bars,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey]);

  const showOverlayTooltip = (p: ChartPositionOverlay, x: number, y: number) => {
    const el = tooltipRef.current;
    if (!el || !seriesRef.current || !chartRef.current) return;
    const opened = p.opened_at ? new Date(p.opened_at).toLocaleString() : "—";
    const lot = p.lot_size != null ? Number(p.lot_size).toFixed(2) : "—";
    const last = p.last_price != null ? Number(p.last_price).toFixed(priceDecimals(symbol)) : "—";
    el.innerHTML = "";
    const rows: [string, string][] = [
      ["Direction", p.direction],
      ["Entry", p.entry_price.toFixed(priceDecimals(symbol))],
      ["SL", Number(p.sl).toFixed(priceDecimals(symbol))],
      ["Managed SL", Number(p.managed_sl).toFixed(priceDecimals(symbol))],
      ["Lot size", lot],
      ["Opened", opened],
      ["Last price", last],
    ];
    rows.forEach(([k, v], i) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between gap-3";
      const kk = document.createElement("span");
      kk.className = "text-meruem-muted";
      kk.textContent = k;
      const vv = document.createElement("span");
      vv.className = i === 0 ? "font-semibold text-meruem-gold" : "font-mono text-meruem-accent";
      vv.textContent = v;
      row.append(kk, vv);
      el.append(row);
    });
    el.style.display = "block";
    // Position the tooltip next to the hovered marker, flipped near the right edge.
    const chartEl = containerRef.current;
    const flip = chartEl ? x > chartEl.clientWidth / 2 : false;
    el.style.left = `${flip ? x - 190 : x + 12}px`;
    el.style.top = `${Math.max(8, y - 40)}px`;
  };

  return (
    <>
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute left-2 top-10 z-20 hidden rounded border border-meruem-border bg-meruem-panel/95 p-2 text-[10px] shadow-lg backdrop-blur"
      />
      <div ref={containerRef} className="h-full w-full" />
    </>
  );
}

interface ChartPanelProps {
  symbol: string;
  onSelectAsset?: (asset: Asset) => void;
}

export function ChartPanel({ symbol, onSelectAsset }: ChartPanelProps) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [tf, setTf] = useState<Timeframe>(TIMEFRAMES[2]); // 15m
  const [lookback, setLookback] = useState(DEFAULT_BARS);
  const [applied, setApplied] = useState(DEFAULT_BARS);
  const [selected, setSelected] = useState(symbol);
  const [chartHeight, setChartHeight] = useState(360);
  const [resetKey, setResetKey] = useState(0);
  const [provider, setProvider] = useState("");
  const [marketOpen, setMarketOpen] = useState(true);
  const [positions, setPositions] = useState<ChartPositionOverlay[]>([]);

  useEffect(() => {
    setSelected(symbol);
    setMarketOpen(true); // fresh state until the new symbol's candles load
  }, [symbol]);

  // Poll open positions for the charted symbol so entry/TP/SL overlay lines
  // track live trades and the managed (trailing) SL in near real-time.
  // Visibility-aware: the poll pauses while the tab is hidden.
  const loadPositions = useCallback(async () => {
    try {
      const res = await api.get<{ positions: ChartPositionOverlay[] }>("/chart/positions", {
        symbol: selected,
      });
      setPositions(res.positions ?? []);
    } catch {
      // chart overlays are best-effort; ignore transient poll failures
    }
  }, [selected]);
  usePolling(loadPositions, 5000);

  // Debounce the lookback slider so dragging doesn't regenerate every chart
  // on every pixel of movement.
  useEffect(() => {
    const id = setTimeout(() => setApplied(lookback), 300);
    return () => clearTimeout(id);
  }, [lookback]);

  useEffect(() => {
    api.get<Asset[]>("/assets").then(setAssets).catch(() => {});
  }, []);

  const pickAsset = (asset: Asset) => {
    setSelected(asset.symbol);
    onSelectAsset?.(asset);
  };

  const bars = applied;

  return (
    <Card
      className="h-full"
      title="Chart"
      action={
        <span
          className={cn(
            "flex items-center gap-1 text-[10px]",
            provider ? (marketOpen ? "text-meruem-green" : "text-meruem-gold") : "text-meruem-yellow",
          )}
        >
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              provider ? (marketOpen ? "animate-pulse bg-meruem-green" : "bg-meruem-gold") : "bg-meruem-yellow",
            )}
          />
          {provider
            ? marketOpen
              ? `LIVE · ${provider}`
              : `MARKET CLOSED · ${provider}`
            : "SIMULATED"}
        </span>
      }
    >
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-2">
        <select
          value={selected}
          onChange={(e) => {
            const a = assets.find((x) => x.symbol === e.target.value);
            if (a) pickAsset(a);
          }}
          className="rounded-md border border-meruem-border bg-meruem-bg px-2 py-1 font-mono text-sm text-meruem-gold focus:border-meruem-accent focus:outline-none"
        >
          {assets.length === 0 && <option value={selected}>{selected}</option>}
          {assets.map((a) => (
            <option key={a.id} value={a.symbol}>
              {a.symbol} — {a.asset_class}
            </option>
          ))}
        </select>

        <div className="flex flex-wrap overflow-hidden rounded-md border border-meruem-border">
          {TIMEFRAMES.map((t) => (
            <button
              key={t.label}
              onClick={() => setTf(t)}
              className={cn(
                "px-2 py-1 font-mono text-xs transition-colors",
                tf.label === t.label
                  ? "bg-meruem-accent/20 text-meruem-accent"
                  : "bg-meruem-bg text-meruem-muted hover:text-meruem-text",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-2 text-[11px] text-meruem-muted">
          Height
          <input
            type="range"
            min={200}
            max={640}
            step={20}
            value={chartHeight}
            onChange={(e) => setChartHeight(Number(e.target.value))}
            className="w-24 accent-meruem-accent"
          />
          <span className="w-8 font-mono text-meruem-text">{chartHeight}</span>
        </label>

        <button
          onClick={() => setResetKey((k) => k + 1)}
          className="rounded-md border border-meruem-border bg-meruem-bg px-2 py-1 font-mono text-xs text-meruem-muted transition-colors hover:text-meruem-accent"
        >
          Reset zoom
        </button>

        <label className="flex items-center gap-2 text-[11px] text-meruem-muted">
          Lookback
          <input
            type="range"
            min={100}
            max={MAX_BARS}
            step={100}
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            className="w-32 accent-meruem-accent"
          />
          <span className="w-16 font-mono text-meruem-text">
            {bars.toLocaleString()} bars
          </span>
        </label>

        <span className="ml-auto font-mono text-xs text-meruem-muted">
          {selected} · {tf.label}
        </span>
      </div>

      <div
        className="relative w-full overflow-hidden rounded-md border border-meruem-border"
        style={{ height: chartHeight }}
      >
        <MiniChart
          symbol={selected}
          tf={tf}
          bars={bars}
          resetKey={resetKey}
          positions={positions}
          onProvider={setProvider}
          onOpenChange={setMarketOpen}
        />
      </div>
    </Card>
  );
}

function MiniChart({
  symbol,
  tf,
  bars,
  resetKey,
  positions,
  onProvider,
  onOpenChange,
}: {
  symbol: string;
  tf: Timeframe;
  bars: number;
  resetKey: number;
  positions?: ChartPositionOverlay[];
  onProvider: (provider: string) => void;
  onOpenChange?: (open: boolean) => void;
}) {
  const priceRef = useRef<HTMLSpanElement>(null);

  // Stable callback: only changes when symbol changes (affects decimals), so
  // parent re-renders don't force a full candle rebuild.
  const onTick = useCallback(
    (p: number) => {
      if (priceRef.current) {
        priceRef.current.textContent = p.toFixed(priceDecimals(symbol));
      }
    },
    [symbol],
  );

  return (
    <>
      <div className="pointer-events-none absolute left-2 top-2 z-10 flex items-center gap-2 rounded bg-meruem-bg/80 px-1.5 py-0.5 font-mono text-[10px] text-meruem-text backdrop-blur">
        <span className="font-semibold text-meruem-gold">{symbol}</span>
        <span className="text-meruem-muted">{tf.label}</span>
        <span ref={priceRef} className="text-meruem-accent" />
      </div>
      <LiveChart
        symbol={symbol}
        barSeconds={tf.sec}
        bars={bars}
        timeframe={tf.label}
        onTick={onTick}
        onProvider={onProvider}
        onOpenChange={onOpenChange}
        resetKey={resetKey}
        positions={positions}
      />
    </>
  );
}