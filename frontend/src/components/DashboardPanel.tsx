import { useCallback, useState } from "react";
import { Account, api, CopierPlan, DashboardSummary, LotPlan } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import { EquityCurve } from "@/components/EquityCurve";
import { cn, usePolling } from "@/lib/utils";

const EMPTY = {
  master_lot: 0.1,
  master_equity: 10000,
  sl_distance_pips: 50,
  pip_value: 10,
  entry_price: 0,
  sl: 0,
  tp1: 0,
  tp2: 0,
  tp3: 0,
  tp4: 0,
  direction: "BUY",
};

const fmtMoney = (v: number | null | undefined, digits = 0) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });

function PnLBadge({ value, prefix = "" }: { value: number | null | undefined; prefix?: string }) {
  if (value == null) return <span className="text-meruem-muted">—</span>;
  const cls = value > 0 ? "text-meruem-green" : value < 0 ? "text-meruem-red" : "text-meruem-muted";
  return <span className={cn("font-mono", cls)}>{prefix}{value > 0 ? "+" : ""}{value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>;
}

export function DashboardPanel() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [masterId, setMasterId] = useState("");
  const [form, setForm] = useState(EMPTY);
  const [plan, setPlan] = useState<LotPlan[]>([]);
  const [masterEquity, setMasterEquity] = useState<number | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      setSummary(await api.get<DashboardSummary>("/dashboard"));
      const a = await api.get<Account[]>("/accounts");
      setAccounts(a);
      const s = await api.get<Record<string, unknown>>("/settings");
      const selected = String(s.selected_master_account_id ?? "");
      const masters = a.filter((x) => x.account_type === "master" && x.enabled !== false);
      if (selected && masters.some((m) => m.id === selected)) setMasterId(selected);
      else if (masters.length) setMasterId(masters[0].id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  usePolling(load, 10000);

  const set = (k: keyof typeof EMPTY, v: string) => setForm({ ...form, [k]: k === "direction" ? v : Number(v) });

  const run = async () => {
    if (!masterId) {
      setMsg({ ok: false, text: "Add and select a master account in the Brokers tab first." });
      return;
    }
    setMsg(null);
    try {
      const res = await api.post<CopierPlan>("/copier/plan", { master_account_id: masterId, ...form, master_equity: null });
      setPlan(res.plans);
      setMasterEquity(res.master?.equity ?? null);
      setMsg({ ok: true, text: `Simulated fan-out across ${res.plans.length} account(s).` });
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
  };

  const master = summary?.master ?? null;
  const slaves = summary?.slaves ?? [];

  return (
    <div className="space-y-3">
      {err && <p className="rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{err}</p>}

      {/* Master / main account performance */}
      <Card className="h-full" title="Main Account Performance">
        {master ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="gold">Master</Badge>
              <span className="font-mono text-sm">{master.nickname}</span>
              <Badge>{master.broker}</Badge>
              {master.connected && <Badge tone="green">● live</Badge>}
              {!master.connected && <Badge>pre-live</Badge>}
              {master.reason && <span className="text-[11px] text-meruem-muted">{master.reason}</span>}
            </div>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              <Metric label="Funded Capital" value={fmtMoney(master.funded)} />
              <Metric label="Running Capital" value={fmtMoney(master.running_capital)} accent="gold" />
              <Metric label="Gain / Loss $" value={<PnLBadge value={master.gain} />} />
              <Metric
                label="Gain / Loss %"
                value={
                  <span className={cn("font-mono", (master.gain_pct ?? 0) > 0 ? "text-meruem-green" : (master.gain_pct ?? 0) < 0 ? "text-meruem-red" : "text-meruem-text")}>
                    {master.gain_pct == null ? "—" : `${master.gain_pct > 0 ? "+" : ""}${master.gain_pct.toFixed(2)}%`}
                  </span>
                }
              />
            </div>
            <div className="flex flex-wrap gap-4 text-[11px] text-meruem-muted">
              <span>Open trades: <span className="text-meruem-text">{master.open_trades}</span></span>
              <span>Closed: <span className="text-meruem-text">{master.closed_trades}</span></span>
              <span>Realized PnL: <PnLBadge value={master.realized_pnl} /></span>
              <span>Lot mode: <span className="text-meruem-text">{accounts.find((x) => x.id === master.account_id)?.risk_scaling_mode ?? "—"}</span></span>
            </div>
          </div>
        ) : (
          <p className="py-4 text-center text-xs text-meruem-muted">
            No master account configured. Add one in the <span className="text-meruem-text">Brokers</span> tab and set it as master.
          </p>
        )}
      </Card>

      {/* Performance curve */}
      <Card className="h-full" title="Equity Curve · Performance">
        {summary?.performance ? (
          <EquityCurve
            curve={summary.performance.curve}
            stats={summary.performance.stats}
          />
        ) : (
          <p className="py-4 text-center text-xs text-meruem-muted">
            No master account configured.
          </p>
        )}
      </Card>

      {/* Slave accounts performance */}
      <Card className="h-full" title="Copier Accounts · Running Capital & Gain/Loss">
        <p className="mb-2 text-[11px] text-meruem-muted">
          Slaves copy the main account. Equity resolves live when connected, else uses funded balance.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-meruem-border text-[11px] uppercase text-meruem-muted">
                <th className="py-1.5 pr-2">Account</th>
                <th className="py-1.5 pr-2">Broker</th>
                <th className="py-1.5 pr-2 text-right">Funded</th>
                <th className="py-1.5 pr-2 text-right">Running Capital</th>
                <th className="py-1.5 pr-2 text-right">Gain/Loss $</th>
                <th className="py-1.5 pr-2 text-right">Gain/Loss %</th>
                <th className="py-1.5 pr-2 text-right">Realized</th>
                <th className="py-1.5">Status</th>
              </tr>
            </thead>
            <tbody>
              {slaves.map((s) => (
                <tr key={s.account_id} className="border-b border-meruem-border/50">
                  <td className="py-1.5 pr-2 font-mono text-xs">{s.nickname}</td>
                  <td className="py-1.5 pr-2"><Badge>{s.broker}</Badge></td>
                  <td className="py-1.5 pr-2 text-right font-mono text-xs">{fmtMoney(s.funded)}</td>
                  <td className="py-1.5 pr-2 text-right font-mono text-xs">
                    {fmtMoney(s.running_capital)}
                    {s.connected && <span className="ml-1 text-meruem-green">●</span>}
                  </td>
                  <td className="py-1.5 pr-2 text-right"><PnLBadge value={s.gain} /></td>
                  <td className="py-1.5 pr-2 text-right">
                    {s.gain_pct == null ? (
                      <span className="text-meruem-muted">—</span>
                    ) : (
                      <span className={cn("font-mono", s.gain_pct > 0 ? "text-meruem-green" : s.gain_pct < 0 ? "text-meruem-red" : "text-meruem-muted")}>
                        {s.gain_pct > 0 ? "+" : ""}{s.gain_pct.toFixed(2)}%
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs"><PnLBadge value={s.realized_pnl} /></td>
                  <td className="py-1.5">
                    <Badge tone={s.enabled === false ? "red" : "green"}>{s.enabled === false ? "disabled" : "active"}</Badge>
                  </td>
                </tr>
              ))}
              {slaves.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-4 text-center text-xs text-meruem-muted">
                    No slave accounts yet — add them in the Brokers tab.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Copier fan-out */}
      <Card
        className="h-full"
        title="Trade Copier · Master → Slave Fan-Out"
        action={
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-meruem-muted">Master account</span>
            <Select className="w-56" value={masterId} onChange={(e) => setMasterId(e.target.value)}>
              {accounts.filter((a) => a.account_type === "master" && a.enabled !== false).map((a) => (
                <option key={a.id} value={a.id}>{a.nickname} ({a.broker})</option>
              ))}
            </Select>
          </div>
        }
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">Master Order</h4>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Master Lot">
                <Input type="number" step="0.01" value={form.master_lot} onChange={(e) => set("master_lot", e.target.value)} />
              </Field>
              <Field label="Master Equity">
                <Input type="number" value={masterEquity ?? form.master_equity} readOnly />
              </Field>
              <Field label="Direction">
                <Select value={form.direction} onChange={(e) => set("direction", e.target.value)}>
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </Select>
              </Field>
              <Field label="SL distance (pips)">
                <Input type="number" value={form.sl_distance_pips} onChange={(e) => set("sl_distance_pips", e.target.value)} />
              </Field>
              <Field label="Pip value ($)">
                <Input type="number" value={form.pip_value} onChange={(e) => set("pip_value", e.target.value)} />
              </Field>
              <Field label="Entry Price">
                <Input type="number" step="0.0001" value={form.entry_price} onChange={(e) => set("entry_price", e.target.value)} />
              </Field>
              <Field label="Stop Loss">
                <Input type="number" step="0.0001" value={form.sl} onChange={(e) => set("sl", e.target.value)} />
              </Field>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {(["tp1", "tp2", "tp3", "tp4"] as const).map((k) => (
                <Field key={k} label={k.toUpperCase()}>
                  <Input type="number" step="0.0001" value={form[k]} onChange={(e) => set(k, e.target.value)} />
                </Field>
              ))}
            </div>
            <Button onClick={run}>Simulate Fan-Out</Button>
            {msg && <p className={`text-[11px] ${msg.ok ? "text-meruem-green" : "text-meruem-red"}`}>{msg.text}</p>}
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">Resulting Lot Plan</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-meruem-border text-[11px] uppercase text-meruem-muted">
                    <th className="py-1.5 pr-2">Account</th>
                    <th className="py-1.5 pr-2">Type</th>
                    <th className="py-1.5 pr-2 text-right">Lot</th>
                    <th className="py-1.5 pr-2 text-right">Equity</th>
                    <th className="py-1.5 pr-2">Mode</th>
                    <th className="py-1.5">Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {plan.map((p) => (
                    <tr key={p.account_id} className="border-b border-meruem-border/50">
                      <td className="py-1.5 pr-2 font-mono text-[11px]">{p.account_id.slice(-10)}</td>
                      <td className="py-1.5 pr-2">
                        <Badge tone={p.account_type === "master" ? "gold" : "blue"}>{p.account_type}</Badge>
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono">{p.lot_size.toFixed(2)}</td>
                      <td className="py-1.5 pr-2 text-right font-mono text-[11px]">
                        {p.equity != null ? p.equity.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"}
                        {p.connected && <span className="ml-1 text-meruem-green">●</span>}
                      </td>
                      <td className="py-1.5 pr-2 text-[11px]">{p.mode}</td>
                      <td className="py-1.5 text-[10px] text-meruem-muted">{p.detail}</td>
                    </tr>
                  ))}
                  {plan.length === 0 && !msg && (
                    <tr>
                      <td colSpan={6} className="py-4 text-center text-xs text-meruem-muted">
                        Fill the master order and simulate to see the lot distribution.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: React.ReactNode; accent?: "gold" | "green" | "red" }) {
  const accentCls =
    accent === "gold" ? "text-meruem-gold" : accent === "green" ? "text-meruem-green" : accent === "red" ? "text-meruem-red" : "text-meruem-text";
  return (
    <div className="rounded-md border border-meruem-border bg-meruem-bg p-2">
      <div className="text-[10px] uppercase text-meruem-muted">{label}</div>
      <div className={`font-mono text-lg ${accentCls}`}>{value}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-1">
      <span className="text-[11px] text-meruem-muted">{label}</span>
      {children}
    </label>
  );
}
