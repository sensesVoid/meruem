import { useCallback, useRef, useState } from "react";
import { AcceptResult, api, AutomationState, Signal, SignalsResponse } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, usePolling } from "@/lib/utils";

const REFRESH_MS = 5000;

function fmt(price?: number): string {
  if (price === undefined || price === null || Number.isNaN(price)) return "-";
  const decimals = Math.abs(price) >= 1000 ? 2 : 5;
  return price.toFixed(decimals);
}

export function SignalsPanel() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [automation, setAutomation] = useState<AutomationState>({
    enabled: true,
    interval_seconds: 300,
    auto_execution_enabled: false,
  });
  const [interval, setIntervalVal] = useState(300);
  const [scanning, setScanning] = useState(false);
  const [accepted, setAccepted] = useState<Record<string, string>>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeRef = useRef<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await api.get<SignalsResponse>("/signals");
      setSignals(res.signals);
      setAutomation(res.automation);
      setIntervalVal(res.automation.interval_seconds);
      if (res.signals.length > 0 && !activeRef.current) {
        activeRef.current = res.signals[0].scan_id;
        setActiveId(res.signals[0].scan_id);
      }
    } catch {
      /* backend offline */
    }
  }, []);

  usePolling(load, REFRESH_MS);

  const scanNow = async () => {
    setScanning(true);
    try {
      const res = await api.post<SignalsResponse>("/signals/scan", { symbol: null });
      setSignals(res.signals);
      if (res.signals.length > 0) {
        activeRef.current = res.signals[0].scan_id;
        setActiveId(res.signals[0].scan_id);
      }
    } finally {
      setScanning(false);
    }
  };

  const toggleAutoScan = async (enabled: boolean) => {
    await api.post("/signals/automation", { enabled, interval_seconds: interval });
    await load();
  };

  const saveInterval = async () => {
    await api.post("/signals/automation", { enabled: automation.enabled, interval_seconds: interval });
    await load();
  };

  const toggleAutoExec = async (enabled: boolean) => {
    await api.put("/settings", { key: "auto_execution_enabled", value: enabled });
    setAutomation((a) => ({ ...a, auto_execution_enabled: enabled }));
  };

  const accept = async (sig: Signal) => {
    try {
      const res = await api.post<AcceptResult>(`/signals/${sig.scan_id}/accept`);
      setAccepted((m) => ({
        ...m,
        [sig.scan_id]:
          res.status === "executed"
            ? `Trade executed — ${res.plans?.length ?? 0} account(s) fanned out`
            : res.reason ?? res.status,
      }));
      load();
    } catch (err) {
      setAccepted((m) => ({ ...m, [sig.scan_id]: String(err) }));
    }
  };

  const active = signals.find((s) => s.scan_id === activeId) ?? signals[0];

  return (
    <div className="grid h-full min-h-0 gap-3 lg:grid-cols-[300px_1fr]">
      {/* Signal list */}
      <Card className="flex min-h-0 flex-col" title="Signals">
        <div className="space-y-1">
          {signals.map((s) => (
            <button
              key={s.scan_id}
              onClick={() => {
                activeRef.current = s.scan_id;
                setActiveId(s.scan_id);
              }}
              className={cn(
                "flex w-full items-center justify-between rounded-md border px-2 py-1.5 text-left transition-colors",
                active?.scan_id === s.scan_id
                  ? "border-meruem-accent/60 bg-meruem-accent/10"
                  : "border-meruem-border/60 hover:bg-meruem-panel",
              )}
            >
              <span className="font-mono text-xs text-meruem-text">{s.asset}</span>
              <SignalTone s={s} />
            </button>
          ))}
          {signals.length === 0 && (
            <p className="py-6 text-center text-xs text-meruem-muted">
              No scans yet — run a scan or wait for auto-scan.
            </p>
          )}
        </div>
      </Card>

      {/* AI analysis output window */}
      <Card
        className="min-h-0"
        title="AI Signal Analysis"
        action={
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="outline" onClick={scanNow} disabled={scanning}>
              {scanning ? "Scanning…" : "Scan now"}
            </Button>
            <label className="flex items-center gap-1 text-[11px] text-meruem-muted">
              Auto-scan
              <input
                type="checkbox"
                checked={automation.enabled}
                onChange={(e) => toggleAutoScan(e.target.checked)}
                className="accent-meruem-accent"
              />
            </label>
            <input
              type="number"
              min={10}
              step={30}
              value={interval}
              onChange={(e) => setIntervalVal(Number(e.target.value))}
              onBlur={saveInterval}
              className="w-20 rounded-md border border-meruem-border bg-meruem-bg px-2 py-1 font-mono text-xs text-meruem-text focus:outline-none"
              title="Scan interval (seconds)"
            />
            <label className="flex items-center gap-1 text-[11px] text-meruem-muted">
              Auto-execute
              <input
                type="checkbox"
                checked={automation.auto_execution_enabled}
                onChange={(e) => toggleAutoExec(e.target.checked)}
                className="accent-meruem-accent"
              />
            </label>
          </div>
        }
      >
        {active ? (
          <SignalDetail sig={active} acceptedMsg={accepted[active.scan_id]} onAccept={accept} />
        ) : (
          <div className="flex h-64 items-center justify-center text-xs text-meruem-muted">
            Waiting for the first scan…
          </div>
        )}
      </Card>
    </div>
  );
}

function SignalTone({ s }: { s: Signal }) {
  const verdict = s.validation?.verdict;
  if (s.executed) return <Badge tone="green">executed</Badge>;
  if (!s.has_trade) return <Badge tone="neutral">no trade</Badge>;
  if (verdict === "ACCEPTABLE") return <Badge tone="green">TRADE</Badge>;
  if (verdict === "NOT_ACCEPTABLE") return <Badge tone="red">rejected</Badge>;
  return <Badge tone="gold">review</Badge>;
}

function SignalDetail({
  sig,
  acceptedMsg,
  onAccept,
}: {
  sig: Signal;
  acceptedMsg?: string;
  onAccept: (s: Signal) => void;
}) {
  const verdict = sig.validation?.verdict;
  const levels = sig.validation?.levels ?? {};
  const c = sig.candidate;
  const actionable = sig.has_trade && verdict === "ACCEPTABLE" && !sig.executed;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-lg text-meruem-text">{sig.asset}</span>
          {sig.candidate.timeframe && (
            <Badge>{sig.candidate.timeframe}</Badge>
          )}
          <SignalTone s={sig} />
          {sig.validation?.provider && (
            <Badge tone="blue">validated by {sig.validation.provider}</Badge>
          )}
        </div>
        <span className="font-mono text-[10px] text-meruem-muted">
          {new Date(sig.timestamp).toLocaleString()}
        </span>
      </div>

      {/* Verdict banner */}
      <div
        className={cn(
          "rounded-md border px-3 py-2 text-sm",
          !sig.has_trade
            ? "border-meruem-yellow/40 bg-meruem-yellow/10 text-meruem-yellow"
            : verdict === "ACCEPTABLE"
              ? "border-meruem-green/40 bg-meruem-green/10 text-meruem-green"
              : verdict === "NOT_ACCEPTABLE"
                ? "border-meruem-red/40 bg-meruem-red/10 text-meruem-red"
                : "border-meruem-border bg-meruem-panel text-meruem-text",
        )}
      >
        {!sig.has_trade
          ? "NO TRADE — no setup found. Rescanning the market…"
          : verdict === "ACCEPTABLE"
            ? "TRADE — setup confirmed by AI validator"
            : verdict === "NOT_ACCEPTABLE"
              ? "NO TRADE — setup rejected by AI validator"
              : verdict === "PAUSED_LLM_LIMIT"
                ? "NO TRADE — all validation providers unavailable (PAUSED_LLM_LIMIT)"
                : "TRADE — waiting for AI validation…"}
      </div>

      {/* Candidate rationale */}
      <div>
        <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-meruem-accent">
          Scanner Analysis (Ollama · local)
        </h5>
        <p className="text-xs text-meruem-text">
          {c.rationale || "No rationale provided."}
        </p>
        {c.confidence !== undefined && (
          <p className="mt-1 text-[11px] text-meruem-muted">
            Confidence: {(c.confidence * 100).toFixed(0)}% · Strategy refs:{" "}
            {c.strategy_refs?.length ? c.strategy_refs.join(", ") : "none"}
          </p>
        )}
      </div>

      {/* Validator analysis */}
      <div>
        <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-meruem-accent">
          AI Validation (cloud · based on Strategy Library)
        </h5>
        <p className="text-xs text-meruem-text">
          {sig.validation?.rationale || "Not yet validated."}
        </p>
      </div>

      {/* Levels */}
      <div>
        <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-meruem-accent">
          Trade Plan
        </h5>
        {sig.has_trade ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <LevelBox label="Direction" value={c.direction ?? "?"} tone={c.direction === "BUY" ? "green" : "red"} />
            <LevelBox label="Entry" value={fmt(levels.entry ?? c.entry_zone?.[0])} />
            <LevelBox label="Invalidation / SL" value={fmt(levels.sl ?? c.invalidation)} tone="red" />
            <LevelBox label="TP1" value={fmt(levels.tp1)} tone="green" />
            <LevelBox label="TP2" value={fmt(levels.tp2)} tone="green" />
            <LevelBox label="TP3" value={fmt(levels.tp3)} tone="green" />
            <LevelBox label="TP4" value={fmt(levels.tp4)} tone="green" />
            <LevelBox label="Entry Zone" value={`${fmt(c.entry_zone?.[0])} – ${fmt(c.entry_zone?.[1])}`} />
          </div>
        ) : (
          <p className="text-xs text-meruem-muted">No trade plan — nothing to display.</p>
        )}
      </div>

      {/* Accept / status */}
      <div className="flex items-center gap-2">
        {actionable && (
          <Button variant="success" onClick={() => onAccept(sig)}>
            Accept & Execute Trade
          </Button>
        )}
        {sig.executed && <Badge tone="green">Executed — trade(s) sent</Badge>}
        {acceptedMsg && (
          <span className={cn("text-[11px]", acceptedMsg.includes("executed") ? "text-meruem-green" : "text-meruem-red")}>
            {acceptedMsg}
          </span>
        )}
      </div>
    </div>
  );
}

function LevelBox({ label, value, tone }: { label: string; value: string; tone?: "green" | "red" }) {
  return (
    <div className="rounded-md border border-meruem-border/60 p-2">
      <div className="text-[10px] uppercase tracking-wider text-meruem-muted">{label}</div>
      <div className={cn("font-mono text-sm", tone === "green" ? "text-meruem-green" : tone === "red" ? "text-meruem-red" : "text-meruem-text")}>
        {value}
      </div>
    </div>
  );
}