import { useCallback, useEffect, useState } from "react";
import { api, BenchmarkResult, ExportSummary, QiMeraStatus, JuliaStatus } from "./api";

export function App() {
  const [status, setStatus] = useState<QiMeraStatus | null>(null);
  const [julia, setJulia] = useState<JuliaStatus | null>(null);
  const [error, setError] = useState("");
  const [asset, setAsset] = useState("EURUSD");
  const [busy, setBusy] = useState<"export" | "inject" | null>(null);
  const [message, setMessage] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [manifest, setManifest] = useState<string>("");

  const [bars, setBars] = useState(20000);
  const [lookback, setLookback] = useState(200);
  const [bench, setBench] = useState<BenchmarkResult | null>(null);
  const [benchBusy, setBenchBusy] = useState(false);

  const load = useCallback(() => {
    api.get<QiMeraStatus>("/status").then(setStatus).catch(() => setStatus(null));
    api.get<JuliaStatus>("/julia/status").then(setJulia).catch(() => setJulia(null));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  const run = async (kind: "export" | "inject") => {
    setBusy(kind);
    setError("");
    setMessage("");
    try {
      if (kind === "export") {
        const res = await api.post<{ manifest: { asset: string; artifact_id: string } }>("/export", {
          asset,
          regime: null,
          note: "skeleton export",
        });
        setMessage(`Exported v-manifest for ${res.manifest.asset}`);
      } else {
        const res = await api.post<{ injected: boolean; filename: string }>("/inject", { asset });
        setMessage(`Injected ${res.filename} into Meruem's ML-artifact store`);
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runBenchmark = async () => {
    setBenchBusy(true);
    setError("");
    try {
      setBench(await api.post<BenchmarkResult>("/benchmark", { bars, lookback, runs: 3 }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBenchBusy(false);
    }
  };

  const selectAsset = async (name: string) => {
    setSelected(name);
    setError("");
    setManifest("");
    try {
      setManifest(JSON.stringify(await api.get(`/export/${name}`), null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div>
      <header>
        <h1>
          <span className="dot" />
          QIMERA
        </h1>
        <span className="muted">
          Feature &amp; parameter research · {status?.app} v{status?.version}
        </span>
      </header>

      <main>
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className={`badge ${status ? "green" : "red"}`}>
              {status ? "API ONLINE" : "API OFFLINE"}
            </span>
            <span className="muted">Export store: {status?.export_dir ?? "—"}</span>
          </div>
        </div>

        <div className="card">
          <h2>Produce an artifact</h2>
          <div className="row">
            <input value={asset} onChange={(e) => setAsset(e.target.value)} placeholder="Symbol e.g. EURUSD" />
            <button className="primary" disabled={busy !== null} onClick={() => run("export")}>
              {busy === "export" ? "Exporting…" : "Generate sample export"}
            </button>
            <button disabled={busy !== null} onClick={() => run("inject")}>
              {busy === "inject" ? "Injecting…" : "Inject into Meruem scanner context"}
            </button>
          </div>
          {error && <p style={{ color: "var(--red)" }}>{error}</p>}
          {message && <p className="muted">{message}</p>}
        </div>

        <div className="card">
          <h2>Exported artifacts</h2>
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Version</th>
                <th>Champions</th>
                <th>Generated</th>
                <th>Artifact ID</th>
              </tr>
            </thead>
            <tbody>
              {(status?.exports ?? []).map((e: ExportSummary) => (
                <tr key={e.asset}>
                  <td>
                    <a
                      href="#"
                      onClick={(ev) => {
                        ev.preventDefault();
                        selectAsset(e.asset);
                      }}
                    >
                      {e.asset}
                    </a>
                  </td>
                  <td>{e.version ?? "—"}</td>
                  <td>{e.champions}</td>
                  <td>{e.generated_at ? new Date(e.generated_at).toLocaleString() : "—"}</td>
                  <td className="muted">{(e.artifact_id ?? "").slice(0, 12)}</td>
                </tr>
              ))}
              {(status?.exports ?? []).length === 0 && (
                <tr>
                  <td colSpan={5} className="muted">No exports yet — generate one above.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Julia compute layer (§6.4 / M2)</h2>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className={`badge ${julia?.available ? "green" : "red"}`}>
              {julia?.available ? "JULIA READY" : "JULIA UNAVAILABLE"}
            </span>
            <span className="muted">
              {julia?.julia_version ? `Julia ${julia.julia_version}` : ""}{" "}
              {julia?.module_loaded ? `· QiMera loaded (${(julia.functions ?? []).join(", ")})` : ""}
              {julia?.error ? ` · ${julia.error}` : ""}
            </span>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <label className="muted">bars</label>
            <input
              type="number"
              value={bars}
              min={1000}
              step={1000}
              onChange={(e) => setBars(Math.max(1000, Number(e.target.value)))}
            />
            <label className="muted">lookback</label>
            <input
              type="number"
              value={lookback}
              min={10}
              step={10}
              onChange={(e) => setLookback(Math.max(10, Number(e.target.value)))}
            />
            <button className="primary" disabled={benchBusy || !julia?.available} onClick={runBenchmark}>
              {benchBusy ? "Running…" : "Benchmark Julia vs Python"}
            </button>
          </div>
          {bench && julia?.available && (
            <table style={{ marginTop: 10 }}>
              <tbody>
                <tr>
                  <td className="muted">Sweep {bench.bars} bars × lookback {bench.lookback}</td>
                  <td>Julia {bench.julia_sweep_s}s</td>
                  <td>Python {bench.python_sweep_s}s</td>
                  <td>
                    <b>{bench.speedup}x speed-up</b>
                  </td>
                  <td className="muted">
                    parity: fitness {bench.parity_fitness ? "✓" : "✗"}, sweep {bench.parity_sweep ? "✓" : "✗"}
                  </td>
                </tr>
                {bench.fitness && (
                  <tr>
                    <td className="muted">fitness</td>
                    <td colSpan={4}>
                      net {bench.fitness.net_pnl.toFixed(2)} · win {bench.fitness.win_rate.toFixed(3)} ·
                      PF {bench.fitness.profit_factor.toFixed(3)} · dd {bench.fitness.max_drawdown_pct.toFixed(2)}% ·
                      {bench.fitness.trades} trades
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        {selected && (
          <div className="card">
            <h2>Manifest — {selected}</h2>
            {manifest ? <pre>{manifest}</pre> : <p className="muted">Loading…</p>}
          </div>
        )}
      </main>
    </div>
  );
}