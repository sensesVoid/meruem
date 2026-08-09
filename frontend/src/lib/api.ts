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
  asset_class: string;
  timeframe: string;
  status: string;
  created_at: string;
  params_json?: string;
  qa_notes?: string;
  strategy_code?: string;
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

export interface PipelineScores {
  iterations: {
    iteration: number;
    accepted: boolean;
    rationale: string;
    train: BacktestMetrics;
    test: BacktestMetrics;
  }[];
}

export interface PipelineDiagnostics {
  synth: {
    concept?: string;
    entry_markdown?: string;
    provider?: string;
    approximations?: string[];
    grounded_on?: string[];
    summary?: string;
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
  scores?: PipelineScores | null;
  recommendations?: PipelineRecommendation[];
  thinking?: PipelineThinkingStep[];
}

export interface StrategyPipelineJob {
  status: "queued" | "running" | "done" | "error" | "unknown";
  stage?: string;
  progress?: number;
  message?: string;
  strategy_id?: string;
  green?: boolean;
  faithful?: boolean;
  error?: string;
  spec?: Record<string, number>;
  gate?: {
    n_trades_train: number;
    n_trades_test: number;
    win_rate_train: number;
    win_rate_test: number;
    sharpe_train: number;
    sharpe_test: number;
    max_drawdown_train: number;
    max_drawdown_test: number;
    profit_factor_train: number;
    profit_factor_test: number;
  };
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

export interface McSharpe {
  p5: number;
  median: number;
  p95: number;
  n: number;
}

export interface BacktestMetrics {
  n_trades: number;
  win_rate: number;
  sharpe: number;
  max_drawdown: number;
  return_pct: number;
  profit_factor: number;
  max_win: number;
  max_loss: number;
  avg_trade_pct: number;
  mc_sharpe: McSharpe;
  mc_costs?: {
    spread_bps: number;
    slippage_bps: number;
    sharpe_p5: number;
    sharpe_median: number;
    sharpe_p95: number;
    ret_p5: number;
    ret_median: number;
    ret_p95: number;
    n: number;
  };
}

export interface EquityPoint {
  t: number;
  equity: number;
  dd: number;
}

export interface BacktestTrade {
  size: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  ret_pct: number;
  entry_time?: string;
  exit_time?: string;
  duration?: string;
}

export interface BacktestResult {
  symbol: string;
  timeframe: string;
  bars: number;
  provider: string;
  spec: Record<string, number>;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: BacktestTrade[];
  status: string;
  faithful?: boolean;
}

export interface BacktestIteration {
  iteration: number;
  accepted: boolean;
  spec: Record<string, number>;
  train: BacktestMetrics;
  test: BacktestMetrics;
  rationale: string;
  next_rationale?: string;
  predicted_improvement?: string;
}

export interface RefineResult {
  strategy_id: string;
  symbol: string;
  timeframe: string;
  bars: number;
  split: { train: number; test: number };
  provider: string;
  status: string;
  faithful?: boolean;
  best: {
    iteration: number;
    spec: Record<string, number>;
    train: BacktestMetrics;
    test: BacktestMetrics;
  } | null;
  iterations: BacktestIteration[];
}

export interface BacktestHistoryItem {
  id: string;
  iteration: number;
  symbol: string;
  timeframe: string;
  spec: Record<string, number>;
  metrics: { train: BacktestMetrics; test: BacktestMetrics };
  rationale: string;
  accepted: boolean;
  created_at: string;
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

export interface DatasetWarn {
  provider: string;
  ok: boolean;
  error: string;
}

export interface DatasetItem {
  id: string;
  symbol: string;
  timeframe: string;
  bar_count: number;
  start_time: string;
  end_time: string;
  file_size_bytes: number;
  file_size_mb: number;
  file_path: string;
  provider?: string;
  warnings?: DatasetWarn[];
  updated_at: string;
}

export interface DatasetExtractError {
  symbol: string;
  error: string;
}

export interface DatasetExtractResponse {
  datasets: DatasetItem[];
  errors: DatasetExtractError[];
}

export interface DatasetPreview {
  id: string;
  stats: {
    min_close: number;
    max_close: number;
    avg_volume: number;
    bar_count: number;
    start_time: string;
    end_time: string;
  };
  sample: {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
}

/**
 * Coverage of the watchlist against the extracted Parquet datasets for a
 * given timeframe. Used by the Backtest and Datasets panels so both render
 * the same ✓ covered / ○ missing picture.
 */
export function datasetCoverage(
  watchlist: Asset[],
  datasets: DatasetItem[],
  timeframe: string,
): { covered: Asset[]; missing: Asset[]; coveredCount: number; total: number } {
  const covered = watchlist.filter((a) =>
    datasets.some((d) => d.symbol === a.symbol && d.timeframe === timeframe),
  );
  const coveredCount = covered.length;
  const missing = watchlist.filter(
    (a) => !datasets.some((d) => d.symbol === a.symbol && d.timeframe === timeframe),
  );
  return { covered, missing, coveredCount, total: watchlist.length };
}