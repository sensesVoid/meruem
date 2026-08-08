import { useMemo } from "react";
import { cn } from "@/lib/utils";

export interface CurvePoint {
  ts: string | null;
  equity: number;
  pnl: number;
}

export interface CurveStats {
  baseline: number;
  equity: number;
  total_closed: number;
  wins: number;
  losses: number;
  win_rate: number;
  profit_factor: number | null;
  gross_win: number;
  gross_loss: number;
  avg_win: number;
  avg_loss: number;
  best: number;
  worst: number;
  max_drawdown_pct: number;
}

const fmtMoney = (v: number, digits = 0) =>
  v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });

function pnlClass(v: number) {
  return v > 0 ? "text-meruem-green" : v < 0 ? "text-meruem-red" : "text-meruem-muted";
}

function Pnl({ v }: { v: number }) {
  return (
    <span className={cn("font-mono", pnlClass(v))}>
      {v > 0 ? "+" : ""}
      {fmtMoney(v, v % 1 ? 2 : 0)}
    </span>
  );
}

export function EquityCurve({ curve, stats }: { curve: CurvePoint[]; stats: CurveStats | null }) {
  const path = useMemo(() => {
    const pts = curve.length >= 2 ? curve : null;
    if (!pts) return null;
    const w = 560;
    const h = 150;
    const pad = 6;
    const min = Math.min(...pts.map((p) => p.equity));
    const max = Math.max(...pts.map((p) => p.equity));
    const range = max - min || 1;
    const x = (i: number) => pad + (i * (w - pad * 2)) / (pts.length - 1);
    const y = (v: number) => pad + ((max - v) * (h - pad * 2)) / range;
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join(" ");
    const area = `${line} L${x(pts.length - 1).toFixed(1)},${(h - pad).toFixed(1)} L${pad.toFixed(1)},${(h - pad).toFixed(1)} Z`;
    return { line, area, max, min, pad, w, h };
  }, [curve]);

  return (
    <div className="flex flex-col gap-3 lg:flex-row">
      <div className="min-w-0 flex-1">
        {path ? (
          <svg viewBox={`0 0 ${path.w} ${path.h}`} className="h-40 w-full">
            <defs>
              <linearGradient id="eq-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgb(139 92 246 / 0.35)" />
                <stop offset="100%" stopColor="rgb(139 92 246 / 0.02)" />
              </linearGradient>
            </defs>
            <path d={path.area} fill="url(#eq-fill)" />
            <path
              d={path.line}
              fill="none"
              stroke="rgb(139 92 246)"
              strokeWidth={1.8}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <text x={path.pad} y={path.h - path.pad} className="text-[10px] text-meruem-muted">
              {fmtMoney(path.min)}
            </text>
            <text x={path.pad} y={12} className="text-[10px] text-meruem-muted">
              {fmtMoney(path.max)}
            </text>
          </svg>
        ) : (
          <p className="py-10 text-center text-xs text-meruem-muted">
            No closed trades yet — the equity curve appears once trades are realised.
          </p>
        )}
      </div>

      {stats && (
        <div className="grid shrink-0 grid-cols-2 gap-2 text-[11px] sm:grid-cols-3 lg:w-56 lg:grid-cols-2">
          <Stat label="Running equity" value={fmtMoney(stats.equity)} accent />
          <Stat label="Net P&L" value={<Pnl v={stats.equity - stats.baseline} />} />
          <Stat label="Win rate" value={`${stats.win_rate.toFixed(1)}%`} sub={`${stats.wins}W / ${stats.losses}L`} />
          <Stat
            label="Profit factor"
            value={stats.profit_factor == null ? "∞" : stats.profit_factor.toFixed(2)}
          />
          <Stat label="Avg win" value={<Pnl v={stats.avg_win} />} />
          <Stat label="Avg loss" value={<Pnl v={-stats.avg_loss} />} />
          <Stat label="Max drawdown" value={`-${stats.max_drawdown_pct.toFixed(2)}%`} />
          <Stat label="Best / Worst" value={fmtMoney(stats.best)} sub={fmtMoney(stats.worst)} />
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  accent?: boolean;
  sub?: string;
}) {
  return (
    <div className="rounded-md border border-meruem-border bg-meruem-bg px-2 py-1.5">
      <div className="text-[10px] uppercase text-meruem-muted">{label}</div>
      <div className={cn("font-mono text-sm", accent && "text-meruem-gold")}>{value}</div>
      {sub && <div className="text-[10px] leading-none text-meruem-muted">{sub}</div>}
    </div>
  );
}
