import { useCallback, useEffect, useState } from "react";
import { api, NewsAgentRead } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { usePolling } from "@/lib/utils";

function biasTone(bias: string): "green" | "red" | "neutral" {
  if (bias === "BULLISH") return "green";
  if (bias === "BEARISH") return "red";
  return "neutral";
}

function ago(iso: string): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const mins = Math.floor((Date.now() - t) / 60000);
  if (mins < 60) return `${Math.max(mins, 0)}m ago`;
  if (mins < 1440) return `${Math.floor(mins / 60)}h ago`;
  return `${Math.floor(mins / 1440)}d ago`;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export function NewsPanel({ symbol }: { symbol?: string }) {
  const [reads, setReads] = useState<Record<string, NewsAgentRead>>({});
  const [refreshedAt, setRefreshedAt] = useState<string | null>(null);
  const [before, setBefore] = useState(30);
  const [after, setAfter] = useState(45);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      // GET /news-agent/state auto-refreshes the cache on first call if empty.
      const res = await api.get<{
        assets: Record<string, NewsAgentRead>;
        refreshed_at: string | null;
        error?: string | null;
      }>("/news-agent/state");
      if (res.error) setError(res.error);
      setReads(res.assets ?? {});
      setRefreshedAt(res.refreshed_at);
    } catch {
      setError("Backend unreachable — is it running?");
      setReads({});
    } finally {
      setLoading(false);
    }
  }, []);

  const loadWindow = async () => {
    try {
      const s = await api.get<Record<string, unknown>>("/settings");
      const b = Number(s["news_block_before_min"] ?? 30);
      const a = Number(s["news_block_after_min"] ?? 45);
      if (Number.isFinite(b)) setBefore(clamp(b, 15, 60));
      if (Number.isFinite(a)) setAfter(clamp(a, 15, 60));
    } catch {
      /* keep defaults */
    }
  };

  const saveWindow = async () => {
    setSaving(true);
    setError("");
    try {
      await api.put("/settings", { key: "news_block_before_min", value: clamp(before, 15, 60) });
      await api.put("/settings", { key: "news_block_after_min", value: clamp(after, 15, 60) });
      await load(); // reflect the new window live
    } catch {
      setError("Failed to save block window.");
    } finally {
      setSaving(false);
    }
  };

  usePolling(load, 10 * 60 * 1000); // refresh every 10 minutes (pauses when tab hidden)
  useEffect(() => {
    loadWindow();
  }, []);

  const all = Object.values(reads);
  const shown = symbol ? all.filter((r) => r.symbol === symbol) : all;

  return (
    <Card
      className="h-full"
      title="News & Economic Calendar"
      action={
        <div className="flex items-center gap-2">
          {symbol && <Badge tone="blue">{symbol}</Badge>}
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
      {refreshedAt && (
        <p className="mb-2 text-[10px] text-meruem-muted">
          AI insights refreshed <span className="text-meruem-accent">{ago(refreshedAt)}</span>
        </p>
      )}
      {error && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 px-2 py-1 text-[11px] text-meruem-red">
          {error}
        </p>
      )}
      <div className="mb-3 rounded-md border border-meruem-border bg-meruem-bg p-2 text-[11px]">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-semibold text-meruem-muted">Trade-block window</span>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1 text-meruem-muted">
              Before
              <input
                type="number"
                min={15}
                max={60}
                value={before}
                onChange={(e) => setBefore(Number(e.target.value) || 15)}
                className="w-14 rounded border border-meruem-border bg-meruem-bg px-1 py-0.5 text-center font-mono text-meruem-text"
              />
              m
            </label>
            <label className="flex items-center gap-1 text-meruem-muted">
              After
              <input
                type="number"
                min={15}
                max={60}
                value={after}
                onChange={(e) => setAfter(Number(e.target.value) || 15)}
                className="w-14 rounded border border-meruem-border bg-meruem-bg px-1 py-0.5 text-center font-mono text-meruem-text"
              />
              m
            </label>
            <button
              onClick={saveWindow}
              disabled={saving}
              className="rounded bg-meruem-accent px-2 py-0.5 text-[11px] font-semibold text-meruem-bg disabled:opacity-50"
            >
              {saving ? "..." : "Save"}
            </button>
          </div>
        </div>
        <p className="mt-1 text-[10px] text-meruem-muted">
          Auto-scan & trades are blocked [before–after] each high-impact news event (15–60 min).
        </p>
      </div>
      <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
        {shown.map((r) => (
          <div
            key={r.symbol}
            className="rounded-md border border-meruem-border bg-meruem-bg p-2"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs font-semibold text-meruem-gold">
                {r.symbol}
              </span>
              <Badge tone={biasTone(r.fundamental_bias)}>
                {r.fundamental_bias}
              </Badge>
            </div>
            {r.trade_block && (
              <div className="mt-1.5 rounded border border-meruem-red/40 bg-meruem-red/10 px-2 py-1 text-[11px] font-semibold text-meruem-red">
                ⛔ TRADE BLOCKED — {r.block_reason || "high-impact news inbound"}
                {r.block_start && r.block_end && (
                  <span className="block font-normal text-meruem-red/80">
                    Until {new Date(r.block_end).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                )}
              </div>
            )}
            <p className="mt-1 text-xs leading-snug text-meruem-text">
              {r.summary || "No AI summary available yet."}
            </p>
            {r.key_events && r.key_events.length > 0 && (
              <ul className="mt-1.5 space-y-0.5">
                {r.key_events.slice(0, 3).map((ev, i) => (
                  <li key={i} className="flex items-start gap-1 text-[11px] text-meruem-muted">
                    <span className="text-meruem-accent">▸</span>
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            )}
            {r.precision_notes && (
              <p className="mt-1 text-[10px] italic text-meruem-muted">{r.precision_notes}</p>
            )}
            <p className="mt-1 flex items-center justify-between text-[10px] text-meruem-muted">
              <span>
                {r.provider} · {r.item_count} sources
              </span>
            </p>
          </div>
        ))}
        {!loading && shown.length === 0 && !error && (
          <p className="py-6 text-center text-xs text-meruem-muted">
            No AI news insights yet — the scheduler builds them every 4 hours. Click Refresh to generate now.
          </p>
        )}
      </div>
    </Card>
  );
}
