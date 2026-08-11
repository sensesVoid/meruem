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

function withParams(path: string, params?: Record<string, string | number>) {
  if (!params) return path;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) qs.set(k, String(v));
  return `${path}?${qs.toString()}`;
}

export const api = {
  get: <T>(path: string, params?: Record<string, string | number>) =>
    request<T>(withParams(path, params)),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body ?? {}) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

export interface SystemStatus {
  system_status: string;
  auto_execution_enabled: boolean;
  risk: { allowed: boolean; reason: string; detail: Record<string, unknown> };
}

export interface Account {
  id: string;
  broker: string;
  account_type: string;
  nickname: string;
  risk_scaling_mode: string;
  scale_factor: number;
  encrypted_ref: string;
  login?: string;
  base_currency?: string;
  balance?: number | null;
  enabled?: boolean;
}

export interface AccountEquity {
  account_id: string;
  equity: number;
  connected: boolean;
  reason?: string;
}

export interface AccountStatus {
  account_id: string;
  nickname: string;
  broker: string;
  account_type: string;
  login: string;
  enabled: boolean;
  connected: boolean;
  reason: string;
}

export interface AccountPerformance {
  account_id: string;
  nickname: string;
  account_type: string;
  broker: string;
  enabled: boolean;
  connected: boolean;
  reason?: string;
  funded: number | null;
  running_capital: number | null;
  equity: number | null;
  gain: number | null;
  gain_pct: number | null;
  open_trades: number;
  closed_trades: number;
  realized_pnl: number;
  is_master: boolean;
}

export interface BrokerStatus {
  /** Per-broker credential presence, e.g. mt5 / binance / mexc (Phase 10 — universal connector). */
  [broker: string]: { configured: boolean } | AccountStatus[] | string | null;
  accounts: AccountStatus[];
  master_id: string | null;
}

export interface BrokerConnectData {
  broker: string;
  account_type: string;
  nickname?: string;
  login?: string;
  server?: string;
  password?: string;
  api_key?: string;
  secret?: string;
  risk_scaling_mode?: string;
  scale_factor?: number;
}

export interface BrokerConnectResult {
  ok: boolean;
  detail?: string;
  connected?: boolean;
  reason?: string;
  account?: Account;
}

export interface DashboardSummary {
  master: AccountPerformance | null;
  slaves: AccountPerformance[];
  total_accounts: number;
  master_id: string | null;
  performance?: {
    curve: { ts: string | null; equity: number; pnl: number }[];
    stats: {
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
    };
  } | null;
}

export interface VaultSlot {
  namespace: string;
  username: string;
  configured: boolean;
  redacted?: string;
}

export interface LLMProvider {
  id: string;
  name: string;
  base_url: string;
  model: string;
  enabled: boolean;
  key_ref: string;
}

export interface VaultStatus {
  slots: VaultSlot[];
  providers: LLMProvider[];
}

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  engine: string;
  model: string;
  ready: boolean;
  status: string;
  note: string;
}

export interface AgentStatus {
  ollama: {
    url: string;
    reachable: boolean;
    models: string[];
    latency_ms: number;
    error: string;
    scanner_model: string;
    validator_model: string;
  };
  providers: (LLMProvider & { key_set: boolean; ready: boolean })[];
  agents: AgentInfo[];
  cloud_ready: boolean;
}

export interface OllamaModels {
  models: string[];
  error?: string;
}

export interface OllamaTest {
  ok: boolean;
  error?: string;
  models?: number;
}

export interface ProviderTest {
  ok: boolean;
  provider: string;
  model: string;
  latency_ms?: number;
  reply?: string;
  error?: string;
}

export interface ConnResult {
  ok: boolean;
  detail?: string;
  credential?: string;
}

export interface LotPlan {
  account_id: string;
  account_type: string;
  lot_size: number;
  mode: string;
  detail: string;
  equity?: number | null;
  connected?: boolean;
}
export interface CopierPlan {
  plans: LotPlan[];
  master?: { id: string; equity: number; account_type: string };
}

export interface SignalValidation {
  verdict: string;
  rationale: string;
  levels: { entry?: number; sl?: number; tp1?: number; tp2?: number; tp3?: number; tp4?: number };
  provider: string;
}

export interface SignalCandidate {
  asset?: string;
  timeframe?: string;
  direction?: string | null;
  entry_zone?: number[];
  invalidation?: number;
  strategy_refs?: string[];
  confidence?: number;
  rationale?: string;
}

export interface Signal {
  scan_id: string;
  asset: string;
  timestamp: string;
  candidate: SignalCandidate;
  has_trade: boolean;
  validation: SignalValidation | null;
  executed: boolean;
}

export interface AutomationState {
  enabled: boolean;
  interval_seconds: number;
  auto_execution_enabled?: boolean;
}

export interface SignalsResponse {
  signals: Signal[];
  automation: AutomationState;
  auto_execution_enabled?: boolean;
}

export interface AcceptResult {
  status: string;
  reason?: string;
  plans?: LotPlan[];
}

export interface Asset {
  id: string;
  symbol: string;
  asset_class: string;
  timeframe_watchlist: string;
}

export interface Strategy {
  id: string;
  name: string;
  concept_markdown?: string;
  body_markdown: string;
  status: string;
  created_at: string;
  params_json?: string;
  qa_notes?: string;
}

export function parseStrategyParams(s: Strategy): Record<string, number> | null {
  if (!s.params_json) return null;
  try {
    const parsed = JSON.parse(s.params_json);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export interface PrimitiveParam {
  type: string;
  default?: string | number | boolean | null;
}

export interface PrimitiveProposal {
  name: string;
  description: string;
  parameters: Record<string, PrimitiveParam>;
  formula?: string;
}

export interface PrimitiveRegistryEntry {
  name: string;
  description: string;
  parameters: Record<string, PrimitiveParam>;
}

export interface PrimitiveRegistryResponse {
  primitives: PrimitiveRegistryEntry[];
}

export interface RegisterPrimitivesResult {
  registered: string[];
  errors: string[];
}

export interface PipelineThinkingStep {
  step: string;
  label: string;
  detail: string;
  data?: unknown;
}

export interface PipelineRecommendation {
  type: string;
  text: string;
}

export interface PipelineDiagnostics {
  synth: {
    concept?: string;
    entry_markdown?: string;
    provider?: string;
    approximations?: string[];
    grounded_on?: string[];
    summary?: string;
    params?: Record<string, number>;
    self_check?: {
      verdict: string;
      quality_score: number;
      contamination: boolean;
      issues: string[];
    } | null;
    qa?: {
      verdict: string;
      quality_score: number;
      contamination: boolean;
      issues: string[];
      provider?: string;
    } | null;
    grounding?: {
      score: number;
      verdict: string;
      ungrounded_rules: string[];
      source_count: number;
    } | null;
    ir?: {
      status?: string;
      errors?: string[];
      blueprint_attempts?: number;
      compiled_blueprint?: { rules: unknown[] } | null;
    } | null;
    new_primitives?: PrimitiveProposal[];
    new_primitive_errors?: string[];
    rag_docs?: string[];
  };
  recommendations?: PipelineRecommendation[];
  thinking?: PipelineThinkingStep[];
}

export interface StrategyPipelineJob {
  status: "queued" | "running" | "done" | "error" | "unknown";
  stage?: string;
  progress?: number;
  message?: string;
  strategy_id?: string;
  error?: string;
  params?: Record<string, number>;
  grounded?: boolean;
  grounding?: {
    score: number;
    verdict: string;
    ungrounded_rules: string[];
    source_count: number;
  } | null;
  ir?: {
    status?: string;
    errors?: string[];
    blueprint_attempts?: number;
    compiled_blueprint?: { rules: unknown[] } | null;
  } | null;
  diagnostics?: PipelineDiagnostics;
}

export interface StrategyReview {
  verdict: "PASS" | "REVISE" | "FAIL";
  quality_score: number;
  contamination: boolean;
  issues: string[];
  rewritten_markdown: string;
  summary: string;
  provider: string;
}

export interface Trade {
  id: string;
  account_id: string;
  entry_price: number;
  sl: number;
  tp1?: number;
  tp2?: number;
  tp3?: number;
  tp4?: number;
  direction: string;
  lot_size: number;
  opened_at: string;
  closed_at?: string;
  pnl?: number;
  mode: string;
  symbol?: string;
  broker?: string;
  broker_order_id?: string | null;
  broker_error?: string;
  exit_price?: number | null;
  close_reason?: string;
  loss_reason?: string;
  ltr_review?: string;
  ltr_reviewed?: number;
}

export interface LTRResult {
  count: number;
  lessons: string;
  reviewed: {
    trade_id: string;
    symbol: string;
    direction: string;
    pnl: number;
    loss_reason: string;
    review: string;
  }[];
}


export interface Validation {
  id: string;
  scan_id: string;
  verdict: string;
  rationale: string;
  suggested_levels_json: string;
  provider_used: string;
}

export interface RegimeRead {
  symbol: string;
  asset_class?: string;
  last_close?: number;
  trend_direction: number;
  trend_label: string;
  trend_strength: number;
  volatility_regime: string;
  volatility_pct: number;
  atr_ratio: number;
  momentum_pct: number;
  rsi: number;
  exhausted: boolean;
  momentum_label: string;
  garch_label?: string;
  garch?: {
    cond_vol: number;
    long_run_vol: number;
    vol_ratio: number;
    alpha: number;
    beta: number;
    omega: number;
    note?: string;
  } | null;
  markov?: { state: string; prob_high: number; vol_calm: number; vol_turb: number; persistence: number; note?: string } | null;
  label?: string;
  note?: string;
}

export interface RegimeBreadth {
  up_pct: number;
  down_pct: number;
  label: string;
}

export interface RegimeState {
  refreshed_at: string | null;
  assets: Record<string, RegimeRead>;
  breadth: RegimeBreadth;
  overall: string;
  error?: string | null;
}

export interface NewsAgentRead {
  symbol: string;
  asset_class?: string;
  summary: string;
  fundamental_bias: string;
  key_events: string[];
  precision_notes: string;
  provider: string;
  item_count: number;
  trade_block?: boolean;
  block_reason?: string;
  block_start?: string | null;
  block_end?: string | null;
}

export interface NewsAgentState {
  refreshed_at: string | null;
  assets: Record<string, NewsAgentRead>;
  error?: string | null;
}

export interface NewsItem {
  title: string;
  url: string;
  source: string;
  published: string;
  summary: string;
}

export interface AgentCatalogEntry {
  id: string;
  name: string;
  role: string;
  engine: string;
  refresh: string;
  default_model: string;
  recommended_model: string;
  tip: string;
  assigned_model: string;
}

export interface AgentCatalog {
  agents: AgentCatalogEntry[];
}

export interface ProviderModelList {
  provider: string;
  models: string[];
  error?: string;
}

export interface RiskDecision {
  allowed: boolean;
  reason: string;
  detail: Record<string, unknown>;
}

export interface RiskAccount {
  id: string;
  type: string;
  nickname: string;
  broker: string;
  mode: string;
  scale: number;
  equity: number;
}

export interface RiskLotSize {
  account_id: string;
  type: string;
  lot: number;
  detail: string;
}

export interface RiskExposure {
  open_trades: number;
  open_auto: number;
  slots_free: number;
  max_concurrent: number;
}

export interface RiskPlan {
  generated_at: string;
  decision: RiskDecision;
  exposure: RiskExposure;
  equity: number;
  risk_pct: number;
  lot_mode: string;
  accounts: RiskAccount[];
  lots: RiskLotSize[];
  advisory: string;
}

export interface RiskClearanceResult {
  allowed: boolean;
  reason: string;
  detail: {
    lots?: RiskLotSize[];
    sl_distance?: number;
    [key: string]: unknown;
  };
}

export interface MLArtifact {
  id: string;
  filename: string;
  symbol?: string | null;
  size: number;
  created_at?: number;
  preview?: string;
}

export interface MLArtifactUploadInput {
  filename: string;
  symbol?: string | null;
  content: string;
}

export async function listMLArtifacts(
  symbol?: string | null,
): Promise<MLArtifact[]> {
  const res = await api.get<{ artifacts: MLArtifact[] }>("/strategies/ml-artifacts", {
    ...(symbol ? { symbol } : {}),
  });
  return res.artifacts;
}

export async function uploadMLArtifact(
  input: MLArtifactUploadInput,
): Promise<MLArtifact> {
  return api.post<MLArtifact>("/strategies/ml-artifacts", {
    filename: input.filename,
    symbol: input.symbol ?? null,
    content: input.content,
  });
}

export async function getMLArtifact(id: string): Promise<MLArtifact> {
  return api.get<MLArtifact>(`/strategies/ml-artifacts/${id}`);
}

export async function deleteMLArtifact(id: string): Promise<{ ok: boolean }> {
  return api.del<{ ok: boolean }>(`/strategies/ml-artifacts/${id}`);
}