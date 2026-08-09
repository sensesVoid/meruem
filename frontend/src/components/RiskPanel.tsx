import { useEffect, useState } from "react";
import {
  api,
  RiskPlan,
  RiskClearanceResult,
} from "@/lib/api";
import {
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  Zap,
  Calculator,
  Sliders,
  DollarSign,
  TrendingDown,
  Layers,
  Settings2,
  ChevronDown,
} from "lucide-react";

function CollapsibleCard({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: React.ReactNode;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-meruem-border bg-meruem-panel p-4">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <h3 className="text-sm font-semibold flex items-center gap-2">
          {icon}
          {title}
        </h3>
        <ChevronDown
          className={`h-4 w-4 text-meruem-muted shrink-0 transition-transform ${open ? "" : "-rotate-90"}`}
        />
      </button>
      {open && <div className="mt-3">{children}</div>}
    </div>
  );
}

export function RiskPanel() {
  const [plan, setPlan] = useState<RiskPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Clearance test form
  const [testSymbol, setTestSymbol] = useState("XAUUSD");
  const [testDirection, setTestDirection] = useState("BUY");
  const [testEntry, setTestEntry] = useState("2650.00");
  const [testSL, setTestSL] = useState("2640.00");
  const [clearanceResult, setClearanceResult] = useState<RiskClearanceResult | null>(null);
  const [clearing, setClearing] = useState(false);

  // Settings state
  const [maxConcurrent, setMaxConcurrent] = useState<number>(2);
  const [dailyLossCap, setDailyLossCap] = useState<number>(-3.0);
  const [maxDrawdown, setMaxDrawdown] = useState<number>(-8.0);
  const [masterRiskPct, setMasterRiskPct] = useState<number>(1.0);
  const [lotMode, setLotMode] = useState<string>("risk_pct");
  const [savingSettings, setSavingSettings] = useState(false);

  // Trading & position-management settings (merged from the retired Settings tab)
  const [autoExecution, setAutoExecution] = useState(false);
  const [manualLot, setManualLot] = useState(0.1);
  const [breakevenEnabled, setBreakevenEnabled] = useState(true);
  const [beTrigger, setBeTrigger] = useState("tp1");
  const [beRMultiple, setBeRMultiple] = useState(1.0);
  const [trailingEnabled, setTrailingEnabled] = useState(true);
  const [trailStepPips, setTrailStepPips] = useState(10);

  const fetchPlan = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RiskPlan>("/risk/plan");
      setPlan(data);
    } catch (err: any) {
      setError(err?.message || "Failed to load risk plan");
    } finally {
      setLoading(false);
    }
  };

  const fetchSettings = async () => {
    try {
      const settings = await api.get<Record<string, any>>("/settings");
      if (settings.max_concurrent_auto_trades !== undefined) {
        setMaxConcurrent(Number(settings.max_concurrent_auto_trades));
      }
      if (settings.daily_loss_cap_pct !== undefined) {
        setDailyLossCap(Number(settings.daily_loss_cap_pct));
      }
      if (settings.max_drawdown_pct !== undefined) {
        setMaxDrawdown(Number(settings.max_drawdown_pct));
      }
      if (settings.master_risk_pct !== undefined) {
        setMasterRiskPct(Number(settings.master_risk_pct));
      }
      if (settings.master_lot_sizing_mode) {
        setLotMode(String(settings.master_lot_sizing_mode));
      }
      if (settings.auto_execution_enabled !== undefined) {
        setAutoExecution(Boolean(settings.auto_execution_enabled));
      }
      if (settings.master_manual_lot !== undefined) {
        setManualLot(Number(settings.master_manual_lot));
      }
      if (settings.auto_breakeven_enabled !== undefined) {
        setBreakevenEnabled(Boolean(settings.auto_breakeven_enabled));
      }
      if (settings.auto_be_trigger) {
        setBeTrigger(String(settings.auto_be_trigger));
      }
      if (settings.auto_be_r_multiple !== undefined) {
        setBeRMultiple(Number(settings.auto_be_r_multiple));
      }
      if (settings.trailing_stop_enabled !== undefined) {
        setTrailingEnabled(Boolean(settings.trailing_stop_enabled));
      }
      if (settings.trailing_stop_step_pips !== undefined) {
        setTrailStepPips(Number(settings.trailing_stop_step_pips));
      }
    } catch (e) {
      // best-effort
    }
  };

  // Immediate-write for the trading/position settings (each change persists).
  const setSetting = async (key: string, value: unknown) => {
    try {
      await api.put("/settings", { key, value });
    } catch (err: any) {
      alert("Failed to save " + key + ": " + (err?.message || "Error"));
    }
  };

  useEffect(() => {
    fetchPlan();
    fetchSettings();
  }, []);

  const handleTestClearance = async (e: React.FormEvent) => {
    e.preventDefault();
    setClearing(true);
    setClearanceResult(null);
    try {
      const res = await api.post<RiskClearanceResult>("/risk/clear", {
        symbol: testSymbol,
        direction: testDirection,
        entry_price: parseFloat(testEntry) || 0,
        sl: parseFloat(testSL) || 0,
      });
      setClearanceResult(res);
    } catch (err: any) {
      setClearanceResult({
        allowed: false,
        reason: err?.message || "Clearance check failed",
        detail: {},
      });
    } finally {
      setClearing(false);
    }
  };

  const handleSaveSettings = async () => {
    setSavingSettings(true);
    try {
      await api.put("/settings", { key: "max_concurrent_auto_trades", value: maxConcurrent });
      await api.put("/settings", { key: "daily_loss_cap_pct", value: dailyLossCap });
      await api.put("/settings", { key: "max_drawdown_pct", value: maxDrawdown });
      await api.put("/settings", { key: "master_risk_pct", value: masterRiskPct });
      await api.put("/settings", { key: "master_lot_sizing_mode", value: lotMode });
      await fetchPlan();
    } catch (err: any) {
      alert("Failed to save settings: " + (err?.message || "Error"));
    } finally {
      setSavingSettings(false);
    }
  };

  const isAllowed = plan?.decision?.allowed ?? false;

  return (
    <div className="flex flex-col gap-4 p-4 text-meruem-text">
      {/* Top Header Card */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-meruem-border bg-meruem-panel p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-full ${
              isAllowed
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                : "bg-rose-500/10 text-rose-400 border border-rose-500/30"
            }`}
          >
            {isAllowed ? <ShieldCheck className="h-6 w-6" /> : <ShieldAlert className="h-6 w-6" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold">Risk Management Agent</h2>
              <span
                className={`rounded px-2 py-0.5 text-xs font-semibold ${
                  isAllowed
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                }`}
              >
                {isAllowed ? "TRADING ALLOWED" : "TRADING BLOCKED"}
              </span>
            </div>
            <p className="text-xs text-meruem-muted">
              {plan?.decision?.reason || "Evaluating risk guardrails..."}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {plan?.generated_at && (
            <span className="text-[11px] text-meruem-muted">
              Updated {new Date(plan.generated_at).toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchPlan}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-md border border-meruem-border bg-meruem-panel/80 px-3 py-1.5 text-xs font-medium text-meruem-text hover:bg-meruem-accent/20 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh Plan
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-400">
          {error}
        </div>
      )}

      {/* AI Risk Advisory Banner */}
      {plan?.advisory && (
        <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 mb-1">
            <Zap className="h-4 w-4" />
            AI Risk Manager Narrative Advisory
          </div>
          <p className="text-xs text-meruem-text/90 leading-relaxed">{plan.advisory}</p>
        </div>
      )}

      {/* Key Risk Metrics Cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {/* Master Equity */}
        <div className="rounded-lg border border-meruem-border bg-meruem-panel p-3.5">
          <div className="flex items-center justify-between text-xs text-meruem-muted mb-1">
            <span>Master Equity</span>
            <DollarSign className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-xl font-bold text-meruem-text">
            ${plan?.equity ? plan.equity.toLocaleString() : "0"}
          </div>
          <div className="mt-1 text-[11px] text-meruem-muted">
            Risk Cap: {plan?.risk_pct ?? 1.0}% (${(((plan?.equity ?? 0) * (plan?.risk_pct ?? 1.0)) / 100).toFixed(2)})
          </div>
        </div>

        {/* System Exposure */}
        <div className="rounded-lg border border-meruem-border bg-meruem-panel p-3.5">
          <div className="flex items-center justify-between text-xs text-meruem-muted mb-1">
            <span>Auto Trade Slots</span>
            <Layers className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-xl font-bold text-meruem-text">
            {plan?.exposure?.open_auto ?? 0} / {plan?.exposure?.max_concurrent ?? 2}
          </div>
          <div className="mt-1 text-[11px] text-meruem-muted">
            {plan?.exposure?.slots_free ?? 0} free slot(s) remaining
          </div>
        </div>

        {/* Daily Loss Cap */}
        <div className="rounded-lg border border-meruem-border bg-meruem-panel p-3.5">
          <div className="flex items-center justify-between text-xs text-meruem-muted mb-1">
            <span>Daily Loss Cap</span>
            <TrendingDown className="h-4 w-4 text-rose-400" />
          </div>
          <div className="text-xl font-bold text-rose-400">{dailyLossCap}%</div>
          <div className="mt-1 text-[11px] text-meruem-muted">
            Halts auto-trading if daily PnL drops below cap
          </div>
        </div>

        {/* Max Drawdown */}
        <div className="rounded-lg border border-meruem-border bg-meruem-panel p-3.5">
          <div className="flex items-center justify-between text-xs text-meruem-muted mb-1">
            <span>Max Drawdown Ceiling</span>
            <ShieldAlert className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-xl font-bold text-amber-400">{maxDrawdown}%</div>
          <div className="mt-1 text-[11px] text-meruem-muted">
            Hard system stop on peak-to-trough drop
          </div>
        </div>
      </div>

      {/* Grid Layout: Left = Account Lot Sizing, Right = Pre-Clearance & Settings */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Left Column (2 cols): Account Lot Sizing Table */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="rounded-lg border border-meruem-border bg-meruem-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Calculator className="h-4 w-4 text-meruem-accent" />
                Per-Account Lot Sizing Plan
              </h3>
              <span className="text-xs text-meruem-muted">
                Sizing Mode: <strong className="text-meruem-text">{plan?.lot_mode ?? lotMode}</strong>
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-meruem-border bg-meruem-panel/60 text-meruem-muted">
                    <th className="py-2 px-3">Account</th>
                    <th className="py-2 px-3">Type</th>
                    <th className="py-2 px-3">Broker</th>
                    <th className="py-2 px-3">Equity</th>
                    <th className="py-2 px-3">Mode</th>
                    <th className="py-2 px-3 text-right">Target Lot</th>
                    <th className="py-2 px-3">Math Detail</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-meruem-border/50">
                  {plan?.accounts && plan.accounts.length > 0 ? (
                    plan.accounts.map((acc) => {
                      const lotItem = plan?.lots?.find((l) => l.account_id === acc.id);
                      return (
                        <tr key={acc.id} className="hover:bg-meruem-accent/5">
                          <td className="py-2.5 px-3 font-medium">
                            {acc.nickname || acc.id}
                          </td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`rounded px-1.5 py-0.5 text-[10px] uppercase font-semibold ${
                                acc.type === "master"
                                  ? "bg-amber-500/20 text-amber-300"
                                  : "bg-blue-500/20 text-blue-300"
                              }`}
                            >
                              {acc.type}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-meruem-muted">{acc.broker}</td>
                          <td className="py-2.5 px-3 font-mono">
                            ${acc.equity?.toLocaleString() ?? "0"}
                          </td>
                          <td className="py-2.5 px-3 text-meruem-muted">
                            {acc.mode} {acc.scale !== 1.0 ? `(×${acc.scale})` : ""}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono font-bold text-emerald-400">
                            {lotItem?.lot ?? "0.00"}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[10px] text-meruem-muted truncate max-w-[200px]" title={lotItem?.detail}>
                            {lotItem?.detail || "-"}
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-meruem-muted">
                        No enabled accounts found. Add or enable accounts in Brokers tab.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column (1 col): Clearance Calculator & Quick Settings */}
        <div className="flex flex-col gap-4">
          {/* Trade Clearance Calculator */}
          <CollapsibleCard
            title="Pre-Trade Clearance Check"
            icon={<Zap className="h-4 w-4 text-emerald-400" />}
          >
            <p className="text-xs text-meruem-muted mb-3">
              Test if a trade passes all risk guardrails and get lot sizes before placing it.
            </p>

            <form onSubmit={handleTestClearance} className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">Symbol</label>
                  <input
                    type="text"
                    value={testSymbol}
                    onChange={(e) => setTestSymbol(e.target.value)}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text focus:outline-none focus:border-meruem-accent"
                    required
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">Direction</label>
                  <select
                    value={testDirection}
                    onChange={(e) => setTestDirection(e.target.value)}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text focus:outline-none focus:border-meruem-accent"
                  >
                    <option value="BUY">BUY</option>
                    <option value="SELL">SELL</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">Entry Price</label>
                  <input
                    type="number"
                    step="any"
                    value={testEntry}
                    onChange={(e) => setTestEntry(e.target.value)}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text focus:outline-none focus:border-meruem-accent"
                    required
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">Stop Loss (SL)</label>
                  <input
                    type="number"
                    step="any"
                    value={testSL}
                    onChange={(e) => setTestSL(e.target.value)}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text focus:outline-none focus:border-meruem-accent"
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={clearing}
                className="mt-1 flex items-center justify-center gap-1.5 rounded bg-meruem-accent px-3 py-2 text-xs font-semibold text-meruem-panel hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                {clearing ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                Check Clearance & Calculate Sizing
              </button>
            </form>

            {clearanceResult && (
              <div
                className={`mt-3 rounded border p-3 text-xs ${
                  clearanceResult.allowed
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                    : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                }`}
              >
                <div className="font-semibold mb-1 flex items-center gap-1.5">
                  {clearanceResult.allowed ? (
                    <ShieldCheck className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <ShieldAlert className="h-4 w-4 text-rose-400" />
                  )}
                  {clearanceResult.allowed ? "CLEARED: Trade Approved" : "BLOCKED: Trade Rejected"}
                </div>
                <p className="text-[11px] opacity-90">{clearanceResult.reason}</p>

                {clearanceResult.allowed && clearanceResult.detail?.lots && (
                  <div className="mt-2.5 border-t border-emerald-500/20 pt-2 flex flex-col gap-1">
                    <div className="text-[11px] font-semibold text-emerald-400">
                      Calculated Lots (SL Dist: {clearanceResult.detail.sl_distance}):
                    </div>
                    {(clearanceResult.detail.lots as any[]).map((l, i) => (
                      <div key={i} className="flex items-center justify-between text-[11px] font-mono">
                        <span>{l.account_id}:</span>
                        <span className="font-bold">{l.lot} lot</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CollapsibleCard>

          {/* Quick Settings Form */}
          <CollapsibleCard
            title="Risk Guardrail Settings"
            icon={<Sliders className="h-4 w-4 text-indigo-400" />}
          >
            <div className="flex flex-col gap-3 text-xs">
              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Max Concurrent Auto-Trades
                </label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={maxConcurrent}
                  onChange={(e) => setMaxConcurrent(parseInt(e.target.value) || 1)}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Daily Loss Cap (%)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={dailyLossCap}
                  onChange={(e) => setDailyLossCap(parseFloat(e.target.value) || 0)}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Max Drawdown Ceiling (%)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={maxDrawdown}
                  onChange={(e) => setMaxDrawdown(parseFloat(e.target.value) || 0)}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Master Risk per Trade (%)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={masterRiskPct}
                  onChange={(e) => setMasterRiskPct(parseFloat(e.target.value) || 0.1)}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Master Lot Sizing Mode
                </label>
                <select
                  value={lotMode}
                  onChange={(e) => setLotMode(e.target.value)}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                >
                  <option value="risk_pct">risk_pct (% of equity)</option>
                  <option value="manual">manual (fixed lot)</option>
                </select>
              </div>

              <button
                onClick={handleSaveSettings}
                disabled={savingSettings}
                className="mt-1 rounded border border-meruem-border bg-meruem-panel/80 py-2 text-xs font-semibold text-meruem-text hover:bg-meruem-accent/20 transition-colors disabled:opacity-50"
              >
                {savingSettings ? "Saving..." : "Save Guardrail Settings"}
              </button>
            </div>
          </CollapsibleCard>

          {/* Trading & Position Settings (merged from the Settings tab) */}
          <CollapsibleCard
            title="Trading &amp; Position Settings"
            icon={<Settings2 className="h-4 w-4 text-emerald-400" />}
          >
            <div className="flex flex-col gap-3 text-xs">
              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Auto-Execution
                </label>
                <select
                  value={autoExecution ? "1" : "0"}
                  onChange={(e) => {
                    setAutoExecution(e.target.value === "1");
                    setSetting("auto_execution_enabled", e.target.value === "1");
                  }}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                >
                  <option value="0">Manual (recommendation cards)</option>
                  <option value="1">Auto</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  Manual Master Lot
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={manualLot}
                  onChange={(e) => {
                    setManualLot(parseFloat(e.target.value) || 0);
                    setSetting("master_manual_lot", parseFloat(e.target.value) || 0);
                  }}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">
                    Auto-Breakeven
                  </label>
                  <select
                    value={breakevenEnabled ? "1" : "0"}
                    onChange={(e) => {
                      setBreakevenEnabled(e.target.value === "1");
                      setSetting("auto_breakeven_enabled", e.target.value === "1");
                    }}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                  >
                    <option value="0">Off</option>
                    <option value="1">On</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">
                    Auto-BE Trigger
                  </label>
                  <select
                    value={beTrigger}
                    onChange={(e) => {
                      setBeTrigger(e.target.value);
                      setSetting("auto_be_trigger", e.target.value);
                    }}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                  >
                    <option value="tp1">On TP1 Hit</option>
                    <option value="r_multiple">On R-Multiple</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[11px] text-meruem-muted mb-1">
                  BE R-Multiple
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={beRMultiple}
                  onChange={(e) => {
                    setBeRMultiple(parseFloat(e.target.value) || 0);
                    setSetting("auto_be_r_multiple", parseFloat(e.target.value) || 0);
                  }}
                  className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">
                    Trailing Stop
                  </label>
                  <select
                    value={trailingEnabled ? "1" : "0"}
                    onChange={(e) => {
                      setTrailingEnabled(e.target.value === "1");
                      setSetting("trailing_stop_enabled", e.target.value === "1");
                    }}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                  >
                    <option value="0">Off</option>
                    <option value="1">On</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] text-meruem-muted mb-1">
                    Trail Step (pips)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={trailStepPips}
                    onChange={(e) => {
                      setTrailStepPips(parseFloat(e.target.value) || 0);
                      setSetting("trailing_stop_step_pips", parseFloat(e.target.value) || 0);
                    }}
                    className="w-full rounded border border-meruem-border bg-meruem-panel/80 px-2.5 py-1.5 text-xs text-meruem-text"
                  />
                </div>
              </div>
            </div>
            <p className="mt-2 text-[10px] text-meruem-muted">
              Changes persist immediately. Credentials, API keys and model selection live in the
              <span className="text-meruem-accent"> Vault</span> tab.
            </p>
          </CollapsibleCard>
        </div>
      </div>
    </div>
  );
}
