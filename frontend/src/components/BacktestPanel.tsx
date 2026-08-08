import { useEffect, useRef, useState } from "react";
import {
  AreaSeries,
  createChart,
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
} from "lightweight-charts";
import { api, Asset, BacktestResult, datasetCoverage, DatasetItem, RefineResult, Strategy } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1D", "1W", "1M"];

function fmt(n: number, d = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

function Metric({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "green" | "red" | "gold" }) {
  const color = tone === "green" ? "text-meruem-green" : tone === "red" ? "text-meruem-red" : tone === "gold" ? "text-meruem-gold" : "text-meruem-text";
  return (
    <div className="rounded-md border border-meruem-border bg-meruem-bg p-2">
      <div className="text-[10px] uppercase tracking-wide text-meruem-muted">{label}</div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
      {hint && <div className="text-[10px] text-meruem-muted">{hint}</div>}
    </div>
  );
}

function EquityChart({ points }: { points: { t: number; equity: number }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height: ref.current.clientHeight,
      layout: { background: { color: "transparent" }, textColor: "#7c8aa0" },
      grid: { vertLines: { color: "#1f2732" }, horzLines: { color: "#1f2732" } },
      rightPriceScale: { borderColor: "#1f2732" },
      timeScale: { borderColor: "#1f2732", timeVisible: true },
      crosshair: { mode: 0 },
    });
    const series = chart.addSeries(AreaSeries, {
      lineColor: "#38bdf8",
      topColor: "rgba(56,189,248,0.25)",
      bottomColor: "rgba(56,189,248,0.02)",
      lineWidth: 2,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });
    chartRef.current = chart;
    seriesRef.current = series;
    const onResize = () => chart.applyOptions({ width: ref.current!.clientWidth });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;
    seriesRef.current.setData(points.map((p) => ({ time: p.t as UTCTimestamp, value: p.equity })));
    chartRef.current.timeScale().fitContent();
  }, [points]);

  return <div ref={ref} className="h-44 w-full" />;
}

export function BacktestPanel({ strategy }: { strategy: Strategy }) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [symbol, setSymbol] = useState("");
  const [tf, setTf] = useState("15m");
  const [bars, setBars] = useState(1500);
  const [maxIters, setMaxIters] = useState(6);
  const [minWinrate, setMinWinrate] = useState(50);
  const [minSharpe, setMinSharpe] = useState(1.0);
  // Optional EXACT window — empty = most recent `bars` candles.
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [watchlist, setWatchlist] = useState<Asset[]>([]);

  const [result, setResult] = useState<BacktestResult | null>(null);
  const [refine, setRefine] = useState<RefineResult | null>(null);
  const [busy, setBusy] = useState<"backtest" | "refine" | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.get<Asset[]>("/assets").then((list) => {
      setAssets(list);
      if (list.length > 0) setSymbol(list[0].symbol);
    }).catch(() => {});
    // Known extracted datasets — used to hint whether this backtest will run
    // on extracted history or fall back to a live fetch.
    api.get<{ datasets: DatasetItem[] }>("/datasets")
      .then((r) => setDatasets(r.datasets || []))
      .catch(() => {});
    // Watchlist — drives the coverage indicator (which symbols still lack an
    // extracted dataset for the selected timeframe).
    api.get<Asset[]>("/assets")
      .then(setWatchlist)
      .catch(() => {});
  }, []);

  const runBacktest = async () => {
    setBusy("backtest");
    setError("");
    try {
      const res = await api.post<BacktestResult>(`/strategies/${strategy.id}/backtest`, {
        symbol, timeframe: tf, bars,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      });
      setResult(res);
      setRefine(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runRefine = async () => {
    setBusy("refine");
    setError("");
    try {
      const res = await api.post<RefineResult>(`/strategies/${strategy.id}/backtest/refine`, {
        symbol, timeframe: tf, bars,
        max_iterations: maxIters, min_winrate: minWinrate, min_sharpe: minSharpe,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      });
      setRefine(res);
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const metrics = result?.metrics ?? refine?.best?.test ?? null;

  // Has an extracted dataset been produced for the selected symbol+timeframe?
  const hasDataset = datasets.some((d) => d.symbol === symbol && d.timeframe === tf);
  const windowActive = startDate !== "" || endDate !== "";

  // Coverage of the whole watchlist for the selected timeframe.
  const coverage = datasetCoverage(watchlist, datasets, tf);
  const missingSymbols = coverage.missing.map((a) => a.symbol);

  return (
    <Card className="h-full" title={`Backtest · ${strategy.name}`}>
      <div className="mb-2 flex flex-wrap items-end gap-1.5">
        <Select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="w-28">
          {assets.map((a) => <option key={a.id} value={a.symbol}>{a.symbol}</option>)}
          {assets.length === 0 && <option value="">no assets</option>}
        </Select>
        <Select value={tf} onChange={(e) => setTf(e.target.value)} className="w-20">
          {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
        </Select>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">bars</span>
          <Input type="number" value={bars} min={100} onChange={(e) => setBars(Number(e.target.value))} className="w-20" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">from</span>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="h-7 rounded border border-meruem-border bg-meruem-panel px-1.5 text-[10px] text-meruem-text focus:outline-none focus:border-meruem-accent"
            title="Optional exact start date (uses the extracted dataset window)"
          />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">to</span>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="h-7 rounded border border-meruem-border bg-meruem-panel px-1.5 text-[10px] text-meruem-text focus:outline-none focus:border-meruem-accent"
            title="Optional exact end date (uses the extracted dataset window)"
          />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">iters</span>
          <Input type="number" value={maxIters} min={1} max={20} onChange={(e) => setMaxIters(Number(e.target.value))} className="w-14" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">minWR</span>
          <Input type="number" value={minWinrate} min={0} max={100} onChange={(e) => setMinWinrate(Number(e.target.value))} className="w-14" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-meruem-muted">minSh</span>
          <Input type="number" value={minSharpe} min={0} step={0.1} onChange={(e) => setMinSharpe(Number(e.target.value))} className="w-14" />
        </div>
        <Button size="sm" onClick={runBacktest} disabled={busy !== null}>
          {busy === "backtest" ? "Running…" : "Backtest"}
        </Button>
        <Button size="sm" variant="outline" onClick={runRefine} disabled={busy !== null} title="Run AI refinement loop (train/test, anti-overfit)">
          {busy === "refine" ? "Refining…" : "AI Refine"}
        </Button>
      </div>

      {error && <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{error}</p>}

      {/* Data source indicator: extracted dataset vs live fetch */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <Badge tone={hasDataset ? "green" : "neutral"}>
          {hasDataset
            ? `✓ extracted dataset (${symbol} ${tf})`
            : "live fetch (no extracted dataset)"}
        </Badge>
        {windowActive && (
          <span className="text-[10px] text-meruem-muted">
            window {startDate || "start"} → {endDate || "now"}
          </span>
        )}
      </div>

      {/* Watchlist coverage for the selected timeframe */}
      {watchlist.length > 0 && (
        <div className="mb-2 rounded-md border border-meruem-border/60 bg-meruem-bg/40 p-1.5">
          <div className="flex flex-wrap items-center gap-1">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
              coverage {coverage.coveredCount}/{coverage.total}
            </span>
            {watchlist.map((a) => {
              const covered = coverage.covered.some((c) => c.symbol === a.symbol);
              return (
                <span
                  key={a.id}
                  className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
                    covered
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                      : "border-amber-500/40 bg-amber-500/10 text-amber-400"
                  }`}
                  title={`${a.symbol} — ${
                    covered
                      ? `has an extracted ${tf} dataset (backtest will use it)`
                      : `NO ${tf} dataset — backtest falls back to live data. Extract it in the ML Scanner tab → Training Data.`
                  }`}
                >
                  {covered ? "✓" : "○"} {a.symbol}
                </span>
              );
            })}
          </div>
          {missingSymbols.length > 0 && (
            <p className="mt-1 flex flex-wrap items-center gap-2 text-[10px] leading-tight text-amber-300/80">
              <span>
                <span className="font-medium text-amber-300">Missing {tf} dataset:</span>{" "}
                {missingSymbols.join(", ")} — these backtests fall back to live data until you
                extract them.
              </span>
              <button
                onClick={() => window.dispatchEvent(new CustomEvent("app:go-to-ml"))}
                className="rounded border border-meruem-accent/50 bg-meruem-accent/10 px-2 py-0.5 font-medium text-meruem-accent transition-colors hover:bg-meruem-accent/20"
                title="Jump to the ML Scanner tab → Training Data section to extract datasets for these symbols"
              >
                Jump to ML Scanner ↗
              </button>
            </p>
          )}
        </div>
      )}

      {result && (
        <p className="mb-1 text-[11px] text-meruem-muted">
          {result.symbol} · {result.timeframe} · {result.bars} bars ·{" "}
          {result.provider === "dataset" ? "extracted dataset" : result.provider}
          {result.faithful && <span className="ml-1 inline-block rounded bg-meruem-accent/15 px-1 py-0.5 font-medium text-meruem-accent">faithful code</span>}
        </p>
      )}
      {refine && (
        <p className="mb-1 text-[11px] text-meruem-muted">
          {refine.faithful && <span className="mr-1 rounded bg-meruem-accent px-1 py-0.5 font-medium text-meruem-accent">faithful code</span>}
          Refine loop: <Badge tone={refine.status === "target_met" ? "green" : refine.status === "improved" ? "gold" : "red"}>{refine.status}</Badge>
          {" · "}train {refine.split.train} / test {refine.split.test} bars ·{" "}
          {refine.provider === "dataset" ? "extracted dataset" : refine.provider}
        </p>
      )}

      {metrics && (
        <div className="mb-2 grid grid-cols-5 gap-1.5">
          <Metric label="Win Rate %" value={fmt(metrics.win_rate)} tone={metrics.win_rate >= 50 ? "green" : "red"} />
          <Metric label="Sharpe" value={fmt(metrics.sharpe)} tone={metrics.sharpe >= 1 ? "green" : "gold"} />
          <Metric
            label="MC Sharpe"
            value={fmt(metrics.mc_sharpe.median)}
            hint={`p5 ${fmt(metrics.mc_sharpe.p5)} · p95 ${fmt(metrics.mc_sharpe.p95)} (${metrics.mc_sharpe.n} boot)`}
          />
          <Metric label="Max Drawdown %" value={fmt(metrics.max_drawdown)} tone={metrics.max_drawdown < -10 ? "red" : "green"} />
          <Metric label="Return %" value={fmt(metrics.return_pct)} tone={metrics.return_pct >= 0 ? "green" : "red"} />
          <Metric label="Max Win $" value={fmt(metrics.max_win)} />
          <Metric label="Max Loss $" value={fmt(metrics.max_loss)} tone="red" />
          <Metric label="Profit Factor" value={fmt(metrics.profit_factor)} />
          <Metric label="Avg Trade %" value={fmt(metrics.avg_trade_pct)} />
          <Metric label="Trades" value={String(metrics.n_trades)} />
        </div>
      )}

      {result && result.equity_curve.length > 0 && (
        <div className="mt-2 rounded-md border border-meruem-border p-1.5">
          <EquityChart points={result.equity_curve} />
        </div>
      )}

      {refine && refine.iterations.length > 0 && (
        <div className="mt-2 max-h-56 space-y-1.5 overflow-y-auto pr-1">
          {refine.iterations.map((it) => (
            <div key={it.iteration} className="rounded-md border border-meruem-border bg-meruem-bg p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">Iteration {it.iteration + 1}</span>
                <div className="flex items-center gap-1.5">
                  <Badge tone={it.accepted ? "green" : "red"}>{it.accepted ? "accepted" : "rejected"}</Badge>
                  {refine.best?.iteration === it.iteration && <Badge tone="gold">best</Badge>}
                </div>
              </div>
              <p className="mt-1 text-[11px] text-meruem-muted">
                train: WR {fmt(it.train.win_rate)}% · Sh {fmt(it.train.sharpe)} · {it.train.n_trades}t
                {"  |  "}test: WR {fmt(it.test.win_rate)}% · Sh {fmt(it.test.sharpe)} · {it.test.n_trades}t
              </p>
              <p className="mt-1 text-[11px] text-meruem-text">{it.rationale}</p>
              {it.next_rationale && (
                <p className="mt-1 text-[11px] text-meruem-accent">→ {it.next_rationale}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
