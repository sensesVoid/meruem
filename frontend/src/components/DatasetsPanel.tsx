import { useEffect, useState } from "react";
import { api, Asset, datasetCoverage, DatasetExtractResponse, DatasetItem, DatasetPreview } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Database, Trash2, Eye, RefreshCw, BarChart2, CalendarRange, AlertTriangle } from "lucide-react";

function defaultRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(start.getFullYear() - 2);
  return {
    start: start.toISOString().slice(0, 10),
    end: end.toISOString().slice(0, 10),
  };
}

// Mirrors the backend safety cap (dataset_service.MAX_BARS) so we can reject an
// obviously-unfetchable range in the UI instantly, before a server round-trip.
const MAX_BARS = 500_000;
const TIMEFRAME_SECONDS: Record<string, number> = {
  M1: 60,
  "1m": 60,
  M5: 300,
  "5m": 300,
  "15m": 900,
  "1h": 3600,
  "4h": 14400,
  "1D": 86400,
  "1d": 86400,
};

function estimateBars(startDate: string, endDate: string, timeframe: string): number {
  if (!startDate || !endDate) return 0;
  const spanSec = Math.max(
    0,
    (Date.parse(`${endDate}T00:00:00Z`) - Date.parse(`${startDate}T00:00:00Z`)) / 1000,
  );
  const secs = TIMEFRAME_SECONDS[timeframe] ?? 900;
  return Math.floor(spanSec / secs);
}

function formatBars(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

export function DatasetsPanel() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [watchlist, setWatchlist] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Form state — extraction follows the watchlist + an EXACT date window.
  const [timeframe, setTimeframe] = useState("15m");
  const [range] = useState(defaultRange); // lazy initializer (computed once)
  const [startDate, setStartDate] = useState(range.start);
  const [endDate, setEndDate] = useState(range.end);

  // Coverage of the watchlist for the selected timeframe (shared helper).
  const coverage = datasetCoverage(watchlist, datasets, timeframe);

  // Live estimate of how many bars the selected range would fetch, so an
  // unfetchable range is flagged in the UI immediately.
  const estBars = estimateBars(startDate, endDate, timeframe);
  const overCap = estBars > MAX_BARS;

  // Preview state
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewingId, setPreviewingId] = useState<string | null>(null);

  const fetchDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ datasets: DatasetItem[] }>("/datasets");
      setDatasets(res.datasets || []);
    } catch (err: any) {
      setError(err?.message || "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  const fetchWatchlist = async () => {
    try {
      setWatchlist(await api.get<Asset[]>("/assets"));
    } catch {
      setWatchlist([]);
    }
  };

  useEffect(() => {
    fetchDatasets();
    fetchWatchlist();
  }, []);

  const handleExtract = async (e: React.FormEvent) => {
    e.preventDefault();
    if (startDate >= endDate) {
      setError("Start date must be before end date.");
      return;
    }
    const est = estimateBars(startDate, endDate, timeframe);
    if (est > MAX_BARS) {
      setError(
        `This range would fetch ~${formatBars(est)} ${timeframe} bars — over the ` +
          `${MAX_BARS.toLocaleString()} bar limit. Shorten the date range or choose a ` +
          `higher timeframe (e.g. 1h / 4h / 1D).`,
      );
      return;
    }
    setExtracting(true);
    setError(null);
    setInfo(null);
    try {
      // No symbol -> the backend extracts EVERY watchlist asset automatically.
      const res = await api.post<DatasetExtractResponse>("/datasets/extract", {
        timeframe,
        start_date: startDate,
        end_date: endDate,
      });
      const errors = res.errors ?? [];
      const warns = res.datasets.flatMap((d) =>
        (d.warnings ?? []).map((w) => `${d.symbol}: ${w.error}`),
      );
      if (res.datasets.length > 0) {
        const src = res.datasets
          .map((d) => `${d.symbol} (${d.provider || "unknown"})`)
          .join(", ");
        setInfo(
          `Extracted ${res.datasets.length} dataset(s) — ${src} — for ${startDate} → ${endDate}.` +
            (errors.length > 0 ? ` ${errors.length} symbol(s) failed (see below).` : "") +
            (warns.length > 0 ? ` Warnings: ${warns.join(" | ")}` : ""),
        );
      } else if (errors.length > 0) {
        setError(
          `No datasets extracted. ${errors.map((er) => `${er.symbol}: ${er.error}`).join(" | ")}`,
        );
      } else {
        setError("No datasets extracted — the watchlist is empty?");
      }
      await fetchDatasets();
    } catch (err: any) {
      setError(err?.message || "Extraction failed");
    } finally {
      setExtracting(false);
    }
  };

  const handlePreview = async (id: string) => {
    if (previewingId === id) {
      setPreviewingId(null);
      setPreview(null);
      return;
    }
    setPreviewingId(id);
    try {
      const res = await api.get<DatasetPreview>(`/datasets/${id}`);
      setPreview(res);
    } catch (err: any) {
      setError(err?.message || "Failed to fetch preview");
      setPreviewingId(null);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete dataset ${id}.parquet?`)) return;
    try {
      await api.del(`/datasets/${id}`);
      if (previewingId === id) {
        setPreviewingId(null);
        setPreview(null);
      }
      await fetchDatasets();
    } catch (err: any) {
      setError(err?.message || "Delete failed");
    }
  };

  return (
    <div className="flex flex-col gap-3 rounded-md border border-meruem-border bg-meruem-panel p-3 text-meruem-text">
      <div className="flex items-center justify-between border-b border-meruem-border/60 pb-2">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-meruem-accent" />
          <h3 className="text-sm font-semibold">Historical Data Extraction (.parquet)</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-meruem-muted">
            {datasets.length} dataset(s) saved
          </span>
          <Button size="sm" variant="outline" onClick={fetchDatasets} disabled={loading}>
            <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      <p className="text-[11px] text-meruem-muted leading-tight">
        Pull historical candles from MT5 terminal / Binance / Yahoo Finance into high-performance
        <strong className="text-meruem-text"> Apache Parquet</strong> datasets for an{" "}
        <strong className="text-meruem-text">exact date range</strong>. Extraction follows the{" "}
        <strong className="text-meruem-text">watchlist</strong> — every asset listed is extracted
        automatically. Extracted datasets are now used automatically by{" "}
        <strong className="text-meruem-text">backtests</strong> and <strong className="text-meruem-text">ML training</strong>{" "}
        (live data is the fallback when a dataset is missing).
      </p>

      {/* Extraction Form */}
      <form onSubmit={handleExtract} className="space-y-2 rounded border border-meruem-border/40 bg-meruem-bg/50 p-2.5">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
            <CalendarRange className="h-3.5 w-3.5 text-meruem-accent" />
            Extraction scope — watchlist assets ({watchlist.length})
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {watchlist.length > 0 ? (
            watchlist.map((a) => {
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
                      ? `has an extracted ${timeframe} dataset (used by backtests/ML)`
                      : `no ${timeframe} dataset yet — backtests fall back to live data`
                  }`}
                >
                  {covered ? "✓" : "○"} {a.symbol}
                </span>
              );
            })
          ) : (
            <span className="text-[10px] text-meruem-muted">
              Watchlist is empty — add assets to the watchlist first.
            </span>
          )}
        </div>
        {watchlist.length > 0 && (
          <p className="text-[10px] text-meruem-muted">
            {coverage.coveredCount}{" of "}
            {coverage.total} watchlist assets have an extracted{" "}
            <span className="font-mono text-meruem-accent">{timeframe}</span> dataset —{" "}
            <span className="text-meruem-text">✓ = backtest/ML use it, ○ = live fallback</span>
          </p>
        )}
        {coverage.missing.length > 0 && (
          <p className="text-[10px] text-amber-300/80">
            <span className="font-medium text-amber-300">Missing:</span>{" "}
            {coverage.missing.map((a) => a.symbol).join(", ")} — extracting now will cover
            them for the <span className="font-mono text-meruem-accent">{timeframe}</span> timeframe.
          </p>
        )}

        <div className="grid gap-2 sm:grid-cols-4 items-end">
          <div>
            <label className="block text-[10px] text-meruem-muted mb-1 font-medium">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              disabled={extracting}
              className="h-8 w-full rounded border border-meruem-border bg-meruem-panel px-2 text-xs text-meruem-text focus:outline-none disabled:opacity-50 disabled:pointer-events-none"
            >
              <option value="M1">M1 (1-min)</option>
              <option value="M5">M5 (5-min)</option>
              <option value="15m">15m (15-min)</option>
              <option value="1h">1h (1-hour)</option>
              <option value="4h">4h (4-hour)</option>
              <option value="1D">1D (1-day)</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-meruem-muted mb-1 font-medium">Start date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              disabled={extracting}
              className="h-8 w-full rounded border border-meruem-border bg-meruem-panel px-2 text-xs text-meruem-text focus:outline-none disabled:opacity-50"
              required
            />
          </div>
          <div>
            <label className="block text-[10px] text-meruem-muted mb-1 font-medium">End date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              disabled={extracting}
              className="h-8 w-full rounded border border-meruem-border bg-meruem-panel px-2 text-xs text-meruem-text focus:outline-none disabled:opacity-50"
              required
            />
          </div>
          <div>
            <Button type="submit" size="sm" className="w-full h-8 flex items-center justify-center gap-1.5" disabled={extracting}>
              {extracting ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
              {extracting ? "Extracting…" : "Extract to Parquet"}
            </Button>
          </div>
        </div>

        {/* Live bar estimate — rejects unfetchable ranges before we hit the server */}
        <div
          className={`flex items-center gap-1.5 rounded border px-1.5 py-1 text-[10px] ${
            overCap
              ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
              : "border-meruem-border/40 bg-meruem-bg/40 text-meruem-muted"
          }`}
        >
          <BarChart2 className={`h-3 w-3 ${overCap ? "text-rose-400" : "text-meruem-accent"}`} />
          <span>
            Range would fetch{" "}
            <span className="font-mono">{formatBars(estBars)}</span> {timeframe} bars
            {overCap ? (
              <> — over the {MAX_BARS.toLocaleString()} bar limit. Shorten the range or use a higher timeframe.</>
            ) : (
              <> (max {MAX_BARS.toLocaleString()}). Large ranges can take several minutes.</>
            )}
          </span>
        </div>

        {/* Processing lock — shown for the entire duration of the fetch */}
        {extracting && (
          <div className="flex items-center gap-2 rounded border border-sky-500/40 bg-sky-500/10 px-2 py-1.5 text-[11px] text-sky-300">
            <RefreshCw className="h-3.5 w-3.5 animate-spin text-sky-400" />
            <span>
              <strong>Fetching historical data…</strong> pulling{" "}
              <span className="font-mono">{formatBars(estBars)}</span> {timeframe} bars per symbol
              from MT5 → Binance → Yahoo. The form is locked — hold on.
            </span>
          </div>
        )}
      </form>

      {info && (
        <div className="rounded border border-emerald-500/30 bg-emerald-500/10 p-2 text-xs text-emerald-300">
          {info}
        </div>
      )}

      {error && (
        <div className="rounded border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Datasets Table */}
      <div className="overflow-x-auto rounded border border-meruem-border">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-meruem-border bg-meruem-panel/60 text-meruem-muted">
              <th className="py-2 px-2.5">Dataset ID</th>
              <th className="py-2 px-2.5">Symbol</th>
              <th className="py-2 px-2.5">Timeframe</th>
              <th className="py-2 px-2.5">Source</th>
              <th className="py-2 px-2.5 text-right">Bars</th>
              <th className="py-2 px-2.5">Date Range</th>
              <th className="py-2 px-2.5 text-right">Size</th>
              <th className="py-2 px-2.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-meruem-border/50">
            {datasets.length > 0 ? (
              datasets.map((d) => (
                <tr key={d.id} className="hover:bg-meruem-accent/5">
                  <td className="py-2 px-2.5 font-mono text-[11px] font-medium text-meruem-text">
                    {d.id}.parquet
                  </td>
                  <td className="py-2 px-2.5 font-semibold">{d.symbol}</td>
                  <td className="py-2 px-2.5">
                    <Badge tone="blue">{d.timeframe}</Badge>
                  </td>
                  <td className="py-2 px-2.5">
                    <Badge tone={d.provider === "mt5" ? "gold" : d.provider === "binance" ? "green" : d.provider === "yahoo" ? "blue" : "neutral"}>
                      {d.provider || "—"}
                    </Badge>
                    {d.warnings && d.warnings.length > 0 && (
                      <span
                        className="ml-1 inline-flex items-center text-amber-300/80 cursor-help"
                        title={d.warnings.map((w) => w.error).join("\n")}
                      >
                        <AlertTriangle className="h-3 w-3" />
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-2.5 text-right font-mono font-medium">
                    {d.bar_count.toLocaleString()}
                  </td>
                  <td className="py-2 px-2.5 text-[10px] text-meruem-muted font-mono">
                    {d.start_time ? `${d.start_time.slice(0, 10)} -> ${d.end_time.slice(0, 10)}` : "-"}
                  </td>
                  <td className="py-2 px-2.5 text-right font-mono text-[11px]">
                    {d.file_size_mb} MB
                  </td>
                  <td className="py-2 px-2.5 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handlePreview(d.id)}
                        className="h-6 px-1.5 text-[10px]"
                        title="Preview dataset stats & sample rows"
                      >
                        <Eye className="h-3 w-3 mr-1" />
                        {previewingId === d.id ? "Hide" : "Preview"}
                      </Button>
                      <button
                        onClick={() => handleDelete(d.id)}
                        className="p-1 text-meruem-muted hover:text-rose-400 transition-colors"
                        title="Delete dataset"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={8} className="py-4 text-center text-xs text-meruem-muted">
                  No historical datasets stored yet. Extract one above!
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Dataset Preview Section */}
      {previewingId && preview && (
        <div className="mt-1 rounded-md border border-indigo-500/30 bg-indigo-500/5 p-3 space-y-2">
          <div className="flex items-center justify-between border-b border-indigo-500/20 pb-1.5">
            <span className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
              <BarChart2 className="h-4 w-4" />
              Preview: {preview.id}.parquet ({preview.stats.bar_count.toLocaleString()} bars)
            </span>
            <span className="text-[10px] text-meruem-muted font-mono">
              {preview.stats.start_time.slice(0, 10)} to {preview.stats.end_time.slice(0, 10)}
            </span>
          </div>

          {/* Quick Stats Grid */}
          <div className="grid grid-cols-3 gap-2 text-[11px]">
            <div className="rounded bg-meruem-bg/60 p-1.5 border border-meruem-border/30">
              <span className="text-meruem-muted block text-[10px]">Min Close</span>
              <span className="font-mono font-semibold">{preview.stats.min_close}</span>
            </div>
            <div className="rounded bg-meruem-bg/60 p-1.5 border border-meruem-border/30">
              <span className="text-meruem-muted block text-[10px]">Max Close</span>
              <span className="font-mono font-semibold">{preview.stats.max_close}</span>
            </div>
            <div className="rounded bg-meruem-bg/60 p-1.5 border border-meruem-border/30">
              <span className="text-meruem-muted block text-[10px]">Avg Volume</span>
              <span className="font-mono font-semibold">{preview.stats.avg_volume.toFixed(2)}</span>
            </div>
          </div>

          {/* Sample Rows Table */}
          <div className="max-h-48 overflow-y-auto rounded border border-meruem-border/40">
            <table className="w-full text-left text-[10px] font-mono border-collapse">
              <thead>
                <tr className="border-b border-meruem-border bg-meruem-panel/80 text-meruem-muted">
                  <th className="py-1 px-2">Timestamp</th>
                  <th className="py-1 px-2 text-right">Open</th>
                  <th className="py-1 px-2 text-right">High</th>
                  <th className="py-1 px-2 text-right">Low</th>
                  <th className="py-1 px-2 text-right">Close</th>
                  <th className="py-1 px-2 text-right">Volume</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-meruem-border/30">
                {preview.sample.map((row, i) => (
                  <tr key={i} className="hover:bg-indigo-500/10">
                    <td className="py-1 px-2 text-meruem-muted">{row.timestamp}</td>
                    <td className="py-1 px-2 text-right">{row.open}</td>
                    <td className="py-1 px-2 text-right text-emerald-400">{row.high}</td>
                    <td className="py-1 px-2 text-right text-rose-400">{row.low}</td>
                    <td className="py-1 px-2 text-right font-bold">{row.close}</td>
                    <td className="py-1 px-2 text-right text-meruem-muted">{row.volume}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
