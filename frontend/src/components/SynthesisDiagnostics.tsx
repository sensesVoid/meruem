import { useEffect, useState } from "react";
import {
  api,
  PipelineDiagnostics,
  PipelineThinkingStep,
  PrimitiveProposal,
  PrimitiveRegistryResponse,
  RegisterPrimitivesResult,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const REC_TONE: Record<string, string> = {
  code: "text-meruem-gold border-meruem-gold/40 bg-meruem-gold/10",
  references: "text-meruem-accent border-meruem-accent/40 bg-meruem-accent/10",
  entries: "text-meruem-accent border-meruem-accent/40 bg-meruem-accent/10",
  overfit: "text-meruem-red border-meruem-red/40 bg-meruem-red/10",
  error: "text-meruem-red border-meruem-red/40 bg-meruem-red/10",
  general: "text-meruem-muted border-meruem-border bg-meruem-bg",
};

const REC_ICON: Record<string, string> = {
  code: "⚠",
  references: "+",
  entries: "◎",
  overfit: "↯",
  error: "✗",
  general: "→",
};

function fmt(v: number | undefined, digits = 2, suffix = ""): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}${suffix}`;
}

function McBar({ value, benchmark = 1.0 }: { value?: { p5: number; median: number; p95: number }; benchmark?: number }) {
  if (!value) return <span className="text-meruem-muted">—</span>;
  const { p5, median, p95 } = value;
  const max = Math.max(p95, benchmark * 1.3, 0.1);
  const lo = Math.min(p5, benchmark, 0);
  const span = Math.max(max - lo, 0.001);
  const mark = (x: number) => `${Math.max(0, Math.min(100, ((x - lo) / span) * 100))}%`;
  return (
    <div className="relative h-3 w-full rounded bg-meruem-bg">
      {/* benchmark line */}
      <div
        className="absolute top-0 bottom-0 w-px bg-meruem-gold/70"
        style={{ left: mark(benchmark) }}
        title={`Benchmark ${benchmark}`}
      />
      <div
        className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded bg-meruem-accent/80"
        style={{ left: mark(p5), width: `calc(${mark(p95)} - ${mark(p5)})` }}
      />
      <div
        className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded bg-meruem-text"
        style={{ left: mark(median) }}
        title={`median ${fmt(median, 3)}`}
      />
    </div>
  );
}

function ThinkingStep({ step, open }: { step: PipelineThinkingStep; open: boolean }) {
  const [expanded, setExpanded] = useState(open);
  const hasData = step.data !== undefined && step.data !== null && step.data !== "";
  return (
    <div className="rounded border border-meruem-border bg-meruem-bg">
      <button
        className="flex w-full items-center justify-between px-2 py-1.5 text-left"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="flex items-center gap-1.5 text-[11px] font-medium text-meruem-text">
          <span className="text-meruem-accent">{expanded ? "▾" : "▸"}</span>
          {step.label}
          {hasData && <span className="text-meruem-muted">·</span>}
          <span className="text-[10px] font-normal text-meruem-muted">{step.detail}</span>
        </span>
      </button>
      {expanded && hasData && (
        <pre className="max-h-40 overflow-auto border-t border-meruem-border/60 px-2 py-1.5 text-[10px] text-meruem-text">
          {typeof step.data === "string"
            ? step.data
            : JSON.stringify(step.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

function PrimitiveCard({
  proposal,
  registered,
  busy,
  onRegister,
}: {
  proposal: PrimitiveProposal;
  registered: boolean;
  busy: boolean;
  onRegister: (p: PrimitiveProposal) => void;
}) {
  const params = Object.entries(proposal.parameters ?? {});
  return (
    <div className="rounded border border-meruem-gold/30 bg-meruem-bg p-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] font-semibold text-meruem-text">
          {proposal.name}
        </span>
        {registered ? (
          <Badge tone="green">✓ registered</Badge>
        ) : (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => onRegister(proposal)}
            title="Approve: write the canonical registry function into ir_primitives.py and live-reload the tool schema"
          >
            {busy ? "Writing…" : "Register"}
          </Button>
        )}
      </div>
      <p className="mt-0.5 text-[10px] text-meruem-muted">{proposal.description}</p>
      {params.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {params.map(([k, v]) => (
            <span
              key={k}
              className="rounded bg-meruem-panel px-1 py-0.5 font-mono text-[9px] text-meruem-accent"
            >
              {k}: {v.type}
              {v.default !== undefined && v.default !== null
                ? ` = ${typeof v.default === "string" ? `"${v.default}"` : String(v.default)}`
                : ""}
            </span>
          ))}
        </div>
      )}
      {proposal.formula && (
        <pre className="mt-1 max-h-20 overflow-auto rounded bg-meruem-panel p-1 text-[9px] leading-relaxed text-meruem-muted">
          {proposal.formula}
        </pre>
      )}
    </div>
  );
}

export function SynthesisDiagnostics({
  diagnostics,
  failed,
}: {
  diagnostics: PipelineDiagnostics;
  failed?: boolean;
}) {
  const [showScores, setShowScores] = useState(true);
  const [showTrace, setShowTrace] = useState(failed === true);
  const [regNames, setRegNames] = useState<Set<string>>(new Set());
  const [regBusy, setRegBusy] = useState<string | null>(null);
  const [regErr, setRegErr] = useState("");
  const [regInfo, setRegInfo] = useState("");
  const synth = diagnostics.synth ?? {};
  const scores = diagnostics.scores;
  const recs = diagnostics.recommendations ?? [];
  const thinking = diagnostics.thinking ?? [];
  const proposals = synth.new_primitives ?? [];

  // Pre-mark tools that are already in the registry (registered earlier, or
  // by another run) so their cards read as done instead of asking twice.
  useEffect(() => {
    api
      .get<PrimitiveRegistryResponse>("/strategies/primitives")
      .then((r) => setRegNames(new Set(r.primitives.map((p) => p.name))))
      .catch(() => {});
  }, []);

  const applyRegistered = (names: string[]) => {
    if (!names.length) return;
    setRegNames((prev) => {
      const next = new Set(prev);
      names.forEach((n) => next.add(n));
      return next;
    });
  };

  const registerOne = async (p: PrimitiveProposal) => {
    setRegBusy(p.name);
    setRegErr("");
    setRegInfo("");
    try {
      const res = await api.post<RegisterPrimitivesResult>(
        "/strategies/primitives/register",
        { primitives: [p] },
      );
      applyRegistered(res.registered);
      setRegInfo(
        `${p.name} is now in the active registry — available to the next synthesis and the IR compiler.`,
      );
      if (res.errors.length) setRegErr(res.errors.join("; "));
    } catch (e) {
      setRegErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRegBusy(null);
    }
  };

  const registerAll = async () => {
    setRegBusy("all");
    setRegErr("");
    setRegInfo("");
    try {
      const res = await api.post<RegisterPrimitivesResult>(
        "/strategies/primitives/register",
        { primitives: proposals },
      );
      applyRegistered(res.registered);
      setRegInfo(
        `Registered ${res.registered.join(", ") || "none"} — new tools are now in the active registry.`,
      );
      if (res.errors.length) setRegErr(res.errors.join("; "));
    } catch (e) {
      setRegErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRegBusy(null);
    }
  };

  const iterations = scores?.iterations ?? [];
  const accepted = iterations.filter((i) => i.accepted);
  const last =
    iterations.length > 0
      ? (accepted.length > 0 ? accepted[accepted.length - 1] : iterations[iterations.length - 1])
      : null;
  const rows = last
    ? [
        { label: "Trades", train: last.train.n_trades, test: last.test.n_trades, bench: "≥ 5", ok: (v: number) => v >= 5 },
        { label: "Win rate", train: last.train.win_rate, test: last.test.win_rate, bench: "≥ 50%", ok: (v: number) => v >= 50 },
        { label: "Sharpe", train: last.train.sharpe, test: last.test.sharpe, bench: "≥ 1.0", ok: (v: number) => v >= 1.0 },
        { label: "Max DD", train: last.train.max_drawdown, test: last.test.max_drawdown, bench: "—", ok: () => true },
        { label: "Profit factor", train: last.train.profit_factor, test: last.test.profit_factor, bench: "—", ok: () => true },
        { label: "Avg trade %", train: last.train.avg_trade_pct, test: last.test.avg_trade_pct, bench: "—", ok: () => true },
      ]
    : [];

  const tone = failed ? "border-meruem-red/40" : "border-meruem-border";

  return (
    <div className={`mb-2 rounded-md border ${tone} bg-meruem-bg/40`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-meruem-border/60 px-2 py-1.5">
        <span className="flex items-center gap-1.5 text-[11px] font-semibold text-meruem-text">
          {failed ? "✗ Synthesis OK — backtest / refine did not pass" : "✓ Synthesis report"}
        </span>
        <div className="flex items-center gap-1.5">
          <Badge tone="blue">{synth.provider ?? "unknown"}</Badge>
          {synth.grounding && (
            <Badge
              tone={
                synth.grounding.verdict === "GROUNDED" ? "green"
                  : synth.grounding.verdict === "PARTIAL" ? "gold" : "red"
              }
            >
              grounding {synth.grounding.verdict.toLowerCase()} · {Math.round(synth.grounding.score * 100)}%
            </Badge>
          )}
        </div>
      </div>

      {/* The actual synthesized strategy */}
      <div className="border-b border-meruem-border/60 px-2 py-2">
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
          Synthesized entry criteria {synth.summary ? `— ${synth.summary}` : ""}
        </p>
        <pre className="max-h-52 overflow-auto rounded border border-meruem-border bg-meruem-bg p-2 text-[10px] leading-relaxed text-meruem-text">
          {synth.entry_markdown || "(no entry criteria)"}
        </pre>
        {synth.rag_docs && synth.rag_docs.length > 0 && (
          <p className="mt-1 text-[10px] text-meruem-muted">
            Grounded on {synth.rag_docs.length} retrieved concept(s):{" "}
            {synth.rag_docs.map((d) => d.split("\n")[0]).join(" · ")}
          </p>
        )}
      </div>

      {/* New tools detected — register cards */}
      {proposals.length > 0 && (
        <div className="border-b border-meruem-border/60 px-2 py-2">
          <div className="mb-1 flex items-center justify-between gap-2">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-meruem-gold">
              New tools detected — register them to extend the primitive registry
            </p>
            {proposals.length > 1 && (
              <Button
                size="sm"
                variant="outline"
                onClick={registerAll}
                disabled={regBusy !== null}
              >
                Register all
              </Button>
            )}
          </div>
          <div className="space-y-1">
            {proposals.map((p) => (
              <PrimitiveCard
                key={p.name}
                proposal={p}
                registered={regNames.has(p.name)}
                busy={regBusy === p.name || regBusy === "all"}
                onRegister={registerOne}
              />
            ))}
          </div>
          {regInfo && <p className="mt-1 text-[10px] text-meruem-green">{regInfo}</p>}
          {regErr && <p className="mt-1 text-[10px] text-meruem-red">{regErr}</p>}
          {synth.new_primitive_errors && synth.new_primitive_errors.length > 0 && (
            <p className="mt-1 rounded bg-meruem-bg p-1 text-[9px] text-meruem-muted">
              {synth.new_primitive_errors.length} proposal(s) rejected by the safety
              gate: {synth.new_primitive_errors.join("; ")}
            </p>
          )}
        </div>
      )}

      {/* Scores incl. Monte Carlo benchmark */}
      {last ? (
        <div className="border-b border-meruem-border/60 px-2 py-2">
          <button
            className="flex w-full items-center justify-between text-left"
            onClick={() => setShowScores(!showScores)}
          >
            <span className="text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
              Scores · iteration {last.iteration} {last.accepted ? "(accepted)" : "(rejected)"}
            </span>
            <span className="text-meruem-accent">{showScores ? "▾" : "▸"}</span>
          </button>
          {showScores && (
            <div className="mt-1.5 space-y-1">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="text-left text-meruem-muted">
                    <th className="py-0.5 pr-2 font-medium">Metric</th>
                    <th className="py-0.5 pr-2 font-medium">Train</th>
                    <th className="py-0.5 pr-2 font-medium">OOS test</th>
                    <th className="py-0.5 font-medium">Benchmark</th>
                  </tr>
                </thead>
                <tbody className="text-meruem-text">
                  {rows.map((r) => (
                    <tr key={r.label} className="border-t border-meruem-border/40">
                      <td className="py-0.5 pr-2 text-meruem-muted">{r.label}</td>
                      <td className={`py-0.5 pr-2 ${r.ok(r.train) ? "text-meruem-green" : "text-meruem-gold"}`}>
                        {fmt(r.train)}
                      </td>
                      <td className={`py-0.5 pr-2 font-medium ${r.ok(r.test) ? "text-meruem-green" : "text-meruem-red"}`}>
                        {fmt(r.test)}
                      </td>
                      <td className="py-0.5 text-meruem-muted">{r.bench}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Monte Carlo Sharpe benchmark */}
              <div className="mt-1.5 rounded border border-meruem-border bg-meruem-bg p-1.5">
                <p className="text-[10px] font-semibold text-meruem-accent">
                  Monte Carlo Sharpe benchmark{" "}
                  <span className="font-normal text-meruem-muted">(1500 bootstrap resamples)</span>
                </p>
                <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                  <div>
                    <p className="text-meruem-muted">Train · p5 / median / p95</p>
                    <p className="text-meruem-text">
                      {fmt(last.train.mc_sharpe?.p5, 3)} / {fmt(last.train.mc_sharpe?.median, 3)} / {fmt(last.train.mc_sharpe?.p95, 3)}
                    </p>
                    <McBar value={last.train.mc_sharpe} />
                  </div>
                  <div>
                    <p className="text-meruem-muted">OOS test · p5 / median / p95</p>
                    <p className="text-meruem-text">
                      {fmt(last.test.mc_sharpe?.p5, 3)} / {fmt(last.test.mc_sharpe?.median, 3)} / {fmt(last.test.mc_sharpe?.p95, 3)}
                    </p>
                    <McBar value={last.test.mc_sharpe} />
                  </div>
                </div>
                {last.train.mc_costs && last.test.mc_costs && (
                  <div className="mt-1.5 border-t border-meruem-border/60 pt-1.5">
                    <p className="text-[10px] font-semibold text-meruem-gold">
                      Cost-adjusted benchmark{" "}
                      <span className="font-normal text-meruem-muted">
                        (randomized spread {last.train.mc_costs.spread_bps} bps · slippage {last.train.mc_costs.slippage_bps} bps per trade)
                      </span>
                    </p>
                    <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
                      <div>
                        <p className="text-meruem-muted">Train · Sharpe p5 / median / p95</p>
                        <p className="text-meruem-text">
                          {fmt(last.train.mc_costs.sharpe_p5, 3)} / {fmt(last.train.mc_costs.sharpe_median, 3)} / {fmt(last.train.mc_costs.sharpe_p95, 3)}
                        </p>
                        <McBar value={{ p5: last.train.mc_costs.sharpe_p5, median: last.train.mc_costs.sharpe_median, p95: last.train.mc_costs.sharpe_p95 }} />
                      </div>
                      <div>
                        <p className="text-meruem-muted">OOS test · Sharpe p5 / median / p95</p>
                        <p className="text-meruem-text">
                          {fmt(last.test.mc_costs.sharpe_p5, 3)} / {fmt(last.test.mc_costs.sharpe_median, 3)} / {fmt(last.test.mc_costs.sharpe_p95, 3)}
                        </p>
                        <McBar value={{ p5: last.test.mc_costs.sharpe_p5, median: last.test.mc_costs.sharpe_median, p95: last.test.mc_costs.sharpe_p95 }} />
                      </div>
                    </div>
                  </div>
                )}
                <p className="mt-1 text-[9px] text-meruem-muted">
                  Gold tick = 1.0 benchmark. Band = 90% of bootstrapped Sharpe outcomes; a median below the tick means the edge likely does not survive cost randomization.
                </p>
              </div>

              {iterations.length > 1 && (
                <p className="text-[9px] text-meruem-muted">
                  {iterations.length} refinement iteration(s) — best accepted kept.
                </p>
              )}
            </div>
          )}
        </div>
      ) : (
        <p className="px-2 py-1.5 text-[10px] text-meruem-muted">
          No backtest scores available — refinement never produced a runnable iteration.
        </p>
      )}

      {/* Recommendations */}
      {recs.length > 0 && (
        <div className="border-b border-meruem-border/60 px-2 py-2">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
            Recommendations
          </p>
          <div className="space-y-1">
            {recs.map((r, i) => (
              <p
                key={i}
                className={`rounded border px-1.5 py-1 text-[10px] ${REC_TONE[r.type] ?? REC_TONE.general}`}
              >
                <span className="mr-1 font-bold">{REC_ICON[r.type] ?? "•"}</span>
                {r.text}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Thinking process */}
      {thinking.length > 0 && (
        <div className="px-2 py-2">
          <button
            className="flex w-full items-center justify-between text-left"
            onClick={() => setShowTrace(!showTrace)}
          >
            <span className="text-[10px] font-semibold uppercase tracking-wide text-meruem-muted">
              Thinking process — how the synthesis was built
            </span>
            <span className="text-meruem-accent">{showTrace ? "▾" : "▸"}</span>
          </button>
          {showTrace && (
            <div className="mt-1.5 space-y-1">
              {thinking.map((s, i) => (
                <ThinkingStep key={s.step ?? i} step={s} open={i === 0} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
