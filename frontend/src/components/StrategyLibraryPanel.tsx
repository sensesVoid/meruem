import { useEffect, useState } from "react";
import { api, Asset, Strategy, StrategyPipelineJob } from "@/lib/api";
import {
  setFinalizeHandler,
  startPoll,
  stopPoll,
  subscribeJob,
} from "@/lib/pipelineJob";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select, Textarea } from "@/components/ui/input";
import { SynthesisDiagnostics } from "@/components/SynthesisDiagnostics";
import { MLArtifactUpload } from "@/components/MLArtifactUpload";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1D", "1W", "1M"];

const CONCEPT_TEMPLATE = `Concepts / Thesis:
  Trade the buy-side liquidity sweep that leaves a fair value gap, in the direction of the higher-timeframe trend.

Definition of Terms:
  - FVG = Fair Value Gap (imbalance between consecutive candles)
  - CSD = Change in State of Delivery

Entry Toolset:
  Volume Profile, VWAP, FVG, CHoCH/MSS, Bullish Engulfing, RSI

Entry Model:
  Liquidity sweep followed by FVG and CSD; enter at the FVG retest, target opposing liquidity

Invalidation / Stop:
  Close below the sweep low

Target:
  Opposing liquidity pool, 2R`;

const STATUS_TONE: Record<string, "green" | "gold" | "blue" | "red" | "neutral"> = {
  active: "green",
  draft: "gold",
  archived: "blue",
  failed: "red",
  reference: "blue",
};

const PIPELINE_STEPS = ["Synthesis", "QA", "Finalize"] as const;

function stepStates(job: StrategyPipelineJob): ("done" | "active" | "pending" | "error")[] {
  const stage = job.stage ?? "";
  if (job.status === "error") {
    // QA gate failure surfaces as a FAIL on the merged gate; everything before the
    // failing step is done, the failing step is red, the rest untouched.
    const failedIdx = stage === "synth" ? 0 : stage === "qa" ? 1 : 2;
    return PIPELINE_STEPS.map((_, i) =>
      i < failedIdx ? "done" : i === failedIdx ? "error" : "pending",
    );
  }
  if (job.status === "queued") return PIPELINE_STEPS.map(() => "pending");
  if (stage === "synth") return ["active", "pending", "pending"];
  if (stage === "qa") return ["done", "active", "pending"];
  if (stage === "finalize") return ["done", "done", "active"];
  return ["done", "done", "done"];
}

function PipelineSteps({ job }: { job: StrategyPipelineJob }) {
  const states = stepStates(job);
  return (
    <div className="mt-2 flex items-center gap-1">
      {PIPELINE_STEPS.map((label, i) => {
        const state = states[i];
        const tone =
          state === "done" ? "text-meruem-green border-meruem-green"
            : state === "error" ? "text-meruem-red border-meruem-red"
              : state === "active" ? "text-meruem-accent border-meruem-accent"
                : "text-meruem-muted border-meruem-border";
        return (
          <div key={label} className="flex items-center gap-1">
            {i > 0 && <div className={`h-px w-3 ${state === "pending" ? "bg-meruem-border" : "bg-meruem-accent/60"}`} />}
            <span className={`rounded border px-1.5 py-0.5 text-[10px] font-medium ${tone}`}>
              {state === "done" ? "✓ " : state === "error" ? "✗ " : ""}
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function StrategyLibraryPanel() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [name, setName] = useState("");
  const [concept, setConcept] = useState("");
  const [body, setBody] = useState("");
  const [savedId, setSavedId] = useState<string | null>(null);
  const [job, setJob] = useState<StrategyPipelineJob | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editConcept, setEditConcept] = useState("");
  const [editBody, setEditBody] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("5m");

  const clearForm = () => {
    setName("");
    setConcept("");
    setBody("");
  };

  const useTemplate = () => {
    setConcept(CONCEPT_TEMPLATE);
    setInfo("Template filled — replace the example content with your own concept. All sections are optional and are fed to the Synthesizer as structured context.");
  };

  const load = async () => {
    try {
      setStrategies(await api.get<Strategy[]>("/strategies"));
    } catch {
      setError("Could not load strategies (backend offline?).");
    }
  };
  useEffect(() => {
    load();
  }, []);
  useEffect(() => {
    api.get<Asset[]>("/assets").then((list) => {
      setAssets(list);
    }).catch(() => {});
  }, []);
  // Polling lives at module scope so synthesis keeps running (and is re-observable)
  // when the user switches tabs. Subscribe here to mirror the active job into state;
  // give the module a live finalize hook so the form clears + list refreshes exactly
  // when the job finishes while this panel is mounted.
  useEffect(() => {
    const unsub = subscribeJob((s) => {
      setJob(s);
      setRunning(s !== null && s.status !== "done" && s.status !== "error");
      if (s && (s.status === "done" || (s.status === "error" && s.strategy_id))) {
        setSavedId(s.strategy_id ?? null);
      }
    });
    return unsub;
  }, []);

  useEffect(() => {
    setFinalizeHandler((s) => {
      if (s.status === "done" || (s.status === "error" && s.strategy_id)) {
        setSavedId(s.strategy_id ?? null);
        clearForm();
        load();
      }
      // Plain error (no strategy persisted): keep the user's input so it can
      // be fixed and resubmitted — handled by the commit above not firing.
    });
    return () => setFinalizeHandler(null);
  }, []);

  const submit = async (strategyId?: string) => {
    const hasReferences = strategies.some((s) => s.status === "reference");
    if (!name.trim() && !strategyId && !hasReferences) {
      setError("Enter a strategy name first (or save reference documents to synthesize).");
      return;
    }
    if (!concept.trim() && !hasReferences) {
      setError("Enter a trading concept first (or save reference documents to synthesize).");
      return;
    }
    const finalName = name.trim() || (strategies.find((s) => s.id === strategyId)?.name) ||
      `Synthesized ${new Date().toISOString().slice(0, 10)}`;
    setError("");
    setInfo("");
    setSavedId(null);
    try {
      const { job_id } = await api.post<{ job_id: string }>("/strategies/submit", {
        name: finalName,
        concept_markdown: concept,
        body_markdown: body,
        strategy_id: strategyId ?? null,
        symbol,
        timeframe: tf,
      });
      startPoll(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const stopPolling = () => {
    stopPoll();
    setJob(null);
  };

  const remove = async (id: string) => {
    // Instant optimistic update
    setStrategies((prev) => prev.filter((s) => s.id !== id));
    try {
      await api.del(`/strategies/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      await load();
    }
  };

  const setStatus = async (id: string, status: string) => {
    try {
      if (status === "active") {
        await api.post(`/strategies/${id}/approve`);
      } else {
        await api.put(`/strategies/${id}`, { status });
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const startEdit = (s: Strategy) => {
    setEditing(s.id);
    setEditName(s.name);
    setEditConcept(s.concept_markdown ?? "");
    setEditBody(s.body_markdown);
  };

  const saveEdit = async (s: Strategy) => {
    try {
      await api.put(`/strategies/${s.id}`, {
        name: editName,
        concept_markdown: editConcept,
        body_markdown: editBody,
      });
      setEditing(null);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const resubmitExisting = async (s: Strategy) => {
    setName(s.name);
    setConcept(s.concept_markdown ?? "");
    setBody(s.body_markdown);
    setSavedId(null);
    await submit(s.id);
  };

  const saveReference = async () => {
    if (!name.trim() || !concept.trim()) {
      setError("Enter a strategy name and a trading concept first.");
      return;
    }
    setError("");
    setInfo("");
    setSavedId(null);
    try {
      await api.post("/strategies/reference", { name, concept_markdown: concept });
      setInfo("Saved as a reference document. References are combined by the Synthesizer when you click Synthesize & QA.");
      clearForm();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Card
      className="h-full"
      title="Strategy Library"
      action={
        <div className="flex items-center gap-1.5">
          <Button
            size="sm"
            variant="outline"
            onClick={saveReference}
            title="Save the concept as a REFERENCE doc in the library; synthesis combines references into one entry"
          >
            Save Reference
          </Button>
          {running && (
            <Button size="sm" variant="outline" onClick={stopPolling}>
              Stop
            </Button>
          )}
        </div>
      }
    >
      <>
          <div className="mb-3 space-y-1.5 rounded-md border border-meruem-border p-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Strategy name" />
            <div className="flex flex-wrap items-end gap-1.5">
              <Select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="w-24" title="Symbol hint for synthesis (entry criteria are kept symbol-agnostic)">
                {assets.map((a) => <option key={a.id} value={a.symbol}>{a.symbol}</option>)}
                {assets.length === 0 && <option value="EURUSD">EURUSD</option>}
              </Select>
              <Select value={tf} onChange={(e) => setTf(e.target.value)} className="w-20" title="Timeframe hint for synthesis">
                {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">Trading concept</p>
              <Button
                size="sm"
                variant="outline"
                onClick={useTemplate}
                title="Fill the structured concept template (Thesis / Terms / Toolset / Model / Invalidation / Target). All sections optional — replace the example with your own concept."
              >
                Use template
              </Button>
            </div>
            <Textarea
              value={concept}
              onChange={(e) => setConcept(e.target.value)}
              placeholder="Trading Concept — your general idea / document (ingested as context; the Synthesizer Agent builds the entry criteria from this..."
              rows={4}
              className="resize-y min-h-[88px]"
            />
        <p className="text-[11px] text-meruem-muted">
          The trading concept is a REFERENCE document — save it first, then
          <span className="text-meruem-accent"> Synthesize &amp; QA</span> compiles all saved
          references (plus the RAG library) into ONE exact, executable entry criteria. Price-action
          mechanics (ICT: liquidity sweeps, FVG/imbalance, CHOCH/MSS, retests) and order-flow
          approximations are honored; synthesis builds the machine-actionable trade from the concepts
          (no concept is rejected for being conceptual). The merged QA gate verdict comes from the
          synthesis call chain itself — no backtest — and a PASS + grounded entry activates it.
        </p>
      </div>

      {savedId && !running && (
        <p className="mb-2 rounded border border-meruem-green/40 bg-meruem-green/10 p-2 text-xs text-meruem-green">
          Strategy saved! (
          {strategies.find((s) => s.id === savedId)?.status ??
            (job?.status === "done" ? "see status below" : "see status below")}
          ).
        </p>
      )}

      {job && (
        <div className="mb-2 rounded-md border border-meruem-border p-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">Processing</span>
            <Badge
              tone={
                job.status === "done" ? "green"
                  : job.status === "error" ? "red"
                    : "gold"
              }
            >
              {job.status}
            </Badge>
          </div>
          <p className="mt-1 text-[11px] text-meruem-text">{job.message || "Working…"}</p>
          <PipelineSteps job={job} />
          {job.status === "done" && (
            <p className="mt-1 text-[10px] text-meruem-accent">
              {job.grounded ? "Entry grounded on the ingested concepts — QA verdict: " : ""}
              {job.diagnostics?.synth?.qa?.verdict ?? "see diagnostics"}
            </p>
          )}
          {job.diagnostics && (job.status === "done" || job.status === "error") && (
            <SynthesisDiagnostics
              diagnostics={job.diagnostics}
              failed={job.status === "error"}
            />
          )}
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded bg-meruem-bg">
            <div
              className="h-full bg-meruem-accent transition-all"
              style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {info && (
        <p className="mb-2 rounded border border-meruem-green/40 bg-meruem-green/10 p-2 text-xs text-meruem-green">{info}</p>
      )}

      {error && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{error}</p>
      )}

      <div className="mb-2 flex items-center justify-between gap-2 rounded-md border border-meruem-accent/40 bg-meruem-accent/5 p-2">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-meruem-accent">
            Synthesize saved references
          </p>
          <p className="text-[10px] text-meruem-muted">
            Compiles all saved reference documents (plus the RAG library) into one
            executable strategy and runs the merged QA gate on it.
          </p>
        </div>
        {running ? (
          <Button size="sm" variant="outline" onClick={stopPolling}>
            Stop
          </Button>
        ) : (
          <Button size="sm" onClick={() => { setEditing(null); submit(); }}>
            Synthesize &amp; QA
          </Button>
        )}
      </div>

      <div className="max-h-[380px] space-y-1.5 overflow-y-auto">
        {strategies.map((s) => (
          <div key={s.id} className="rounded-md border border-meruem-border bg-meruem-bg p-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{s.name}</span>
              <div className="flex items-center gap-1.5">
                <Badge tone={STATUS_TONE[s.status] ?? "gold"}>{s.status}</Badge>
                {s.status === "reference" ? (
                  <Button size="sm" variant="outline" onClick={() => resubmitExisting(s)} title="Compile this reference (with all saved references) into a machine-actionable entry">
                    Synthesize
                  </Button>
                ) : (
                  <>
                    {s.status !== "active" && (
                      <Button size="sm" variant="success" onClick={() => setStatus(s.id, "active")} title="Promote to ACTIVE (usable by Scanner/Validator)">
                        Approve
                      </Button>
                    )}
                    {s.status !== "draft" && (
                      <Button size="sm" variant="outline" onClick={() => setStatus(s.id, "draft")} title="Send back to DRAFT">
                        Revise
                      </Button>
                    )}
                  </>
                )}
                <button className="text-meruem-red" onClick={() => remove(s.id)}>x</button>
              </div>
            </div>
            <p className="mt-1 line-clamp-2 text-[11px] text-meruem-muted">{s.body_markdown}</p>
            {s.concept_markdown && (
              <p className="mt-1 line-clamp-1 text-[10px] text-meruem-accent">Concept: {s.concept_markdown}</p>
            )}
            <p className="mt-1 text-[10px] text-meruem-muted">
              {s.created_at?.slice(0, 10)}
            </p>
            {s.status === "failed" && s.qa_notes && (
              <p className="mt-1 rounded bg-meruem-red/10 p-1.5 text-[10px] text-meruem-red">
                QA: {s.qa_notes}
              </p>
            )}

            {editing === s.id ? (
              <div className="mt-1.5 space-y-1.5 rounded border border-meruem-border p-1.5">
                <Input value={editName} onChange={(e) => setEditName(e.target.value)} />
                <Textarea
                  value={editConcept}
                  onChange={(e) => setEditConcept(e.target.value)}
                  placeholder="Trading Concept"
                  rows={2}
                  className="resize-y"
                />
                <Textarea
                  value={editBody}
                  onChange={(e) => setEditBody(e.target.value)}
                  rows={4}
                  className="resize-y"
                />
                <div className="flex gap-1.5">
                  <Button size="sm" variant="success" onClick={() => saveEdit(s)}>Save</Button>
                  <Button size="sm" variant="outline" onClick={() => setEditing(null)}>Cancel</Button>
                  <Button size="sm" variant="outline" onClick={() => resubmitExisting(s)} title="Re-run synthesis + QA with this body">
                    Resubmit
                  </Button>
                </div>
              </div>
            ) : (
              (s.status === "failed" || s.status === "draft") && (
                <div className="mt-1.5 flex gap-1.5">
                  <Button size="sm" variant="outline" onClick={() => startEdit(s)}>Edit</Button>
                </div>
              )
            )}
          </div>
        ))}
        {strategies.length === 0 && (
          <p className="py-4 text-center text-xs text-meruem-muted">No strategies saved</p>
        )}
      </div>

      <MLArtifactUpload />
      </>
    </Card>
  );
}
