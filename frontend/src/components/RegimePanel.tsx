import { useEffect, useState } from "react";
import { api, RegimeRead, RegimeState } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

function toneFor(label: string): "green" | "red" | "gold" | "neutral" | "blue" {
  if (label.includes("UP") || label === "BULLISH_REGIME" || label === "BROAD-UP") return "green";
  if (label.includes("DOWN") || label === "BEARISH_REGIME" || label === "BROAD-DOWN") return "red";
  if (label.includes("HIGH") || label.includes("EXHAUST")) return "gold";
  if (label === "MOMENTUM_UP") return "green";
  if (label === "MOMENTUM_DOWN") return "red";
  return "neutral";
}

function overallTone(overall: string): "green" | "red" | "gold" {
  if (overall === "BULLISH_REGIME") return "green";
  if (overall === "BEARISH_REGIME") return "red";
  return "gold";
}

export function RegimePanel({ symbol }: { symbol?: string }) {
  const [state, setState] = useState<RegimeState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.get<RegimeState>("/regime");
      if (res.error) setError(res.error);
      setState(res);
    } catch {
      setError("Backend unreachable — is it running?");
      setState(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 5 * 60 * 1000); // refresh every 5 minutes
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  const assets = state?.assets ? Object.values(state.assets) : [];
  const reads: RegimeRead[] = symbol
    ? assets.filter((a) => a.symbol === symbol)
    : assets;

  return (
    <Card
      className="h-full"
      title="Market Regime Agent"
      action={
        <div className="flex items-center gap-2">
          {symbol && <Badge tone="blue">{symbol}</Badge>}
          {state?.overall && (
            <Badge tone={overallTone(state.overall)}>{state.overall}</Badge>
          )}
          <button
            onClick={load}
            className="text-[11px] text-meruem-muted hover:text-meruem-text"
            disabled={loading}
          >
            {loading ? "..." : "Refresh"}
          </button>
        </div>
      }
    >
      {error && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 px-2 py-1 text-[11px] text-meruem-red">
          {error}
        </p>
      )}

      {state?.breadth && (
        <div className="mb-2 flex items-center gap-2 text-[10px] text-meruem-muted">
          <span>Breadth: </span>
          <span className="font-mono text-meruem-green">↑ {state.breadth.up_pct}%</span>
          <span className="font-mono text-meruem-red">↓ {state.breadth.down_pct}%</span>
          <Badge tone={overallTone(state.overall)}>{state.breadth.label}</Badge>
        </div>
      )}

      <div className="max-h-[520px] space-y-1.5 overflow-y-auto pr-1">
        {(symbol ? reads : assets).map((a) => (
          <div
            key={a.symbol}
            className="rounded-md border border-meruem-border bg-meruem-bg p-2"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs font-semibold text-meruem-gold">
                {a.symbol}
              </span>
              <Badge tone={toneFor(a.trend_label)}>{a.trend_label}</Badge>
            </div>
            <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-meruem-muted">
              <span>
                Trend <span className="text-meruem-text">ADX {a.trend_strength}</span>
              </span>
              <span>
                Vol <span className="text-meruem-text">{a.volatility_regime}</span>
              </span>
              <span>
                RSI <span className="text-meruem-text">{a.rsi}</span>
              </span>
              <span>
                Mom{" "}
                <span className={cn("text-meruem-text", a.exhausted && "text-meruem-gold")}>
                  {a.momentum_label}
                </span>
              </span>
            </div>
            {(a.garch || a.markov) && (
              <div className="mt-1.5 flex flex-wrap items-center gap-1 border-t border-meruem-border/60 pt-1.5">
                {a.markov && (
                  <Badge tone={a.markov.state === "TURBULENT" ? "gold" : "blue"}>
                    {a.markov.state} {Math.round(a.markov.prob_high * 100)}%
                  </Badge>
                )}
                {a.garch_label && (
                  <Badge
                    tone={
                      a.garch_label === "GARCH_EXPANDING" ? "gold"
                      : a.garch_label === "GARCH_COMPRESSING" ? "blue" : "neutral"
                    }
                  >
                    {a.garch_label} {a.garch?.vol_ratio ? `×${a.garch.vol_ratio.toFixed(2)}` : ""}
                  </Badge>
                )}
              </div>
            )}
          </div>
        ))}
        {!loading && (symbol ? reads : assets).length === 0 && !error && (
          <p className="py-6 text-center text-xs text-meruem-muted">
            No regime data yet — the 5-minute scheduler will populate it.
          </p>
        )}
      </div>
    </Card>
  );
}
