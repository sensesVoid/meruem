const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) }),
};

export interface ExportSummary {
  asset: string;
  version?: number | null;
  generated_at?: string | null;
  champions: number;
  artifact_id?: string;
}

export interface QiMeraStatus {
  app: string;
  version: string;
  export_dir: string;
  exports: ExportSummary[];
  meruem: { api: string };
}

export interface JuliaStatus {
  available: boolean;
  project: string;
  error?: string;
  julia_version?: string;
  module_loaded?: boolean;
  functions?: string[];
}

export interface BenchmarkResult {
  bars: number;
  lookback: number;
  runs: number;
  parity_fitness: boolean | null;
  parity_sweep: boolean;
  julia_sweep_s: number | null;
  python_sweep_s: number | null;
  speedup: number | null;
  fitness: {
    net_pnl: number;
    win_rate: number;
    profit_factor: number;
    max_drawdown_pct: number;
    trades: number;
  } | null;
}