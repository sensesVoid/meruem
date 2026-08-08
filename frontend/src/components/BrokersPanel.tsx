import { useEffect, useState } from "react";
import { Account, api, BrokerConnectResult, BrokerStatus } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import { cn } from "@/lib/utils";

const MODES = [
  ["proportional", "Proportional (equity)"],
  ["fixed", "Fixed multiplier"],
  ["risk_pct", "Risk % of equity"],
];

type ExchangeId = "MT5" | "Binance" | "MEXC";

const defaultForm = {
  exchange: "MT5" as ExchangeId,
  account_type: "slave",
  nickname: "",
  login: "",
  server: "",
  password: "",
  api_key: "",
  secret: "",
  risk_scaling_mode: "proportional",
  scale_factor: 1.0,
  balance: "" as string,
};

export function BrokersPanel() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [statusMap, setStatusMap] = useState<Record<string, { connected: boolean; reason: string }>>({});
  const [selectedMaster, setSelectedMaster] = useState("");
  const [form, setForm] = useState(defaultForm);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [err, setErr] = useState("");

const load = async () => {
    try {
      const b = await api.get<BrokerStatus>("/brokers/status");
      setStatusMap(Object.fromEntries(b.accounts.map((a) => [a.account_id, { connected: a.connected, reason: a.reason }])));
      setAccounts(await api.get<Account[]>("/accounts"));
      setSelectedMaster(b.master_id ?? "");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  useEffect(() => {
    load();
  }, []);

  const set = (k: keyof typeof defaultForm, v: string) =>
    setForm({ ...form, [k]: k === "scale_factor" ? Number(v) : k === "exchange" ? (v as ExchangeId) : v });

  const testConnection = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const res = await api.post<BrokerConnectResult>("/brokers/test", {
        broker: form.exchange,
        account_type: form.account_type,
        login: form.login,
        server: form.server,
        password: form.password,
        api_key: form.api_key,
        secret: form.secret,
      });
      setMsg({ ok: res.ok, text: res.detail || (res.ok ? "connection OK" : "failed") });
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  const connect = async () => {
    if (!form.exchange || !form.account_type) return;
    setBusy(true);
    setErr("");
    setMsg(null);
    try {
      const res = await api.post<BrokerConnectResult>("/brokers/connect", {
        broker: form.exchange,
        account_type: form.account_type,
        nickname: form.nickname.trim() || undefined,
        login: form.login,
        server: form.server,
        password: form.password,
        api_key: form.api_key,
        secret: form.secret,
        risk_scaling_mode: form.risk_scaling_mode,
        scale_factor: form.scale_factor,
      });
      if (res.ok && res.connected) {
        setMsg({ ok: true, text: `Connected · ${res.reason || "OK"}` });
        setForm({ ...defaultForm, exchange: form.exchange, account_type: form.account_type });
      } else {
        const reason = res.reason || res.detail || "not connected";
        setMsg({ ok: false, text: reason });
      }
      await load();
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  const setMaster = async (id: string) => {
    await api.put("/settings", { key: "selected_master_account_id", value: id });
    setSelectedMaster(id);
    await load();
  };

  const patchAccount = async (id: string, patch: Record<string, unknown>) => {
    await api.put(`/accounts/${id}`, patch);
    await load();
  };

  const removeAccount = async (id: string) => {
    await api.del(`/accounts/${id}`);
    await load();
  };

  const toggleEnabled = async (a: Account) => patchAccount(a.id, { enabled: !(a.enabled ?? true) });

  // API-key exchanges (Binance / MEXC / future ccxt brokers) share the key+secret form.
  const isMT5 = form.exchange === "MT5";

  return (
    <Card className="h-full" title="Brokers, Exchanges & Accounts">
      {err && <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{err}</p>}

      {/* ===== Accounts view (on top) ===== */}
      <div className="mb-5">
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wider text-meruem-accent">Connected Accounts</h4>
        <p className="mb-2 text-[11px] text-meruem-muted">
          Choose one master (drives the dashboard, direct account details & chart data). Slaves copy it. Green = good connection, red = problem.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-meruem-border text-[11px] uppercase text-meruem-muted">
                <th className="py-1.5 pr-2">Nickname</th>
                <th className="py-1.5 pr-2">Broker</th>
                <th className="py-1.5 pr-2">Role</th>
                <th className="py-1.5 pr-2">Lot Mode</th>
                <th className="py-1.5 pr-2 text-right">Scale</th>
                <th className="py-1.5 pr-2">Connection</th>
                <th className="py-1.5 pr-2">Status</th>
                <th className="py-1.5 pr-2">Act as Master</th>
                <th className="py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {accounts.map((a) => {
                const st = statusMap[a.id];
                const isMaster = selectedMaster === a.id || a.account_type === "master";
                return (
                  <tr key={a.id} className="border-b border-meruem-border/50">
                    <td className="py-1.5 pr-2 font-mono text-xs">{a.nickname}</td>
                    <td className="py-1.5 pr-2"><Badge>{a.broker}</Badge></td>
                    <td className="py-1.5 pr-2">
                      <Badge tone={a.account_type === "master" ? "gold" : "blue"}>{a.account_type}</Badge>
                    </td>
                    <td className="py-1.5 pr-2 text-xs">{a.risk_scaling_mode}</td>
                    <td className="py-1.5 pr-2 text-right font-mono text-xs">
                      <Input
                        type="number"
                        step="0.05"
                        className="w-20"
                        value={a.scale_factor}
                        onChange={(e) => patchAccount(a.id, { scale_factor: Number(e.target.value) })}
                      />
                    </td>
                    <td className="py-1.5 pr-2">
                      <span className="flex items-center gap-1.5">
                        <span
                          className={cn(
                            "h-2 w-2 shrink-0 rounded-full",
                            st ? (st.connected ? "bg-meruem-green" : "bg-meruem-red") : "bg-meruem-muted",
                          )}
                        />
                        <span className="text-[11px] text-meruem-muted">
                          {st ? (st.connected ? "good" : "bad") : "—"}
                        </span>
                      </span>
                      {st && (
                        <p className="max-w-64 text-[10px] text-meruem-muted">{st.reason}</p>
                      )}
                    </td>
                    <td className="py-1.5 pr-2">
                      <Button size="sm" variant={a.enabled === false ? "danger" : "outline"} onClick={() => toggleEnabled(a)}>
                        {a.enabled === false ? "disabled" : "enabled"}
                      </Button>
                    </td>
                    <td className="py-1.5 pr-2">
                      {a.account_type === "master" ? (
                        <Button size="sm" variant={isMaster ? "success" : "outline"} onClick={() => setMaster(a.id)}>
                          {isMaster ? "Master ✓" : "Set Master"}
                        </Button>
                      ) : (
                        <span className="text-[11px] text-meruem-muted">—</span>
                      )}
                    </td>
                    <td className="py-1.5">
                      <Button size="sm" variant="danger" onClick={() => removeAccount(a.id)}>Delete</Button>
                    </td>
                  </tr>
                );
              })}
              {accounts.length === 0 && (
                <tr>
                  <td colSpan={9} className="py-4 text-center text-xs text-meruem-muted">
                    No accounts yet — connect one below.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ========= Connection form (below the view) ========= */}
      <div className="rounded-md border border-meruem-border/60 bg-meruem-bg/30 p-3">
        <h4 className="mb-1 text-xs font-semibold uppercase tracking-wider text-meruem-accent">Connect an Account</h4>
        <p className="mb-3 text-[11px] text-meruem-muted">
          Pick an exchange, then a role, then enter credentials. Test first to see if it works before saving.
        </p>

        {/* Step 1 — exchange */}
        <div className="mb-3 flex flex-wrap gap-2">
          {(["MT5", "Binance", "MEXC"] as ExchangeId[]).map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => set("exchange", ex)}
              className={cn(
                "min-w-24 rounded-md border px-4 py-2 text-sm font-medium transition-colors",
                form.exchange === ex
                  ? "border-meruem-accent bg-meruem-accent/15 text-meruem-accent"
                  : "border-meruem-border bg-meruem-bg text-meruem-muted hover:text-meruem-text",
              )}
            >
              {ex === "MT5" ? "MetaTrader 5" : ex === "Binance" ? "Binance" : "MEXC"}
            </button>
          ))}
        </div>

        {/* Step 2 — role */}
        <div className="mb-3 flex flex-wrap gap-2">
          <span className="self-center text-[11px] text-meruem-muted">Role:</span>
          <button
            type="button"
            onClick={() => set("account_type", "master")}
            className={cn(
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              form.account_type === "master"
                ? "border-meruem-gold bg-meruem-gold/15 text-meruem-gold"
                : "border-meruem-border bg-meruem-bg text-meruem-muted hover:text-meruem-text",
            )}
          >
            Master
          </button>
          <button
            type="button"
            onClick={() => set("account_type", "slave")}
            className={cn(
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              form.account_type === "slave"
                ? "border-meruem-accent bg-meruem-accent/15 text-meruem-accent"
                : "border-meruem-border bg-meruem-bg text-meruem-muted hover:text-meruem-text",
            )}
          >
            Slave
          </button>
          {form.account_type === "master" && (
            <span className="self-center text-[11px] text-meruem-gold">Only one master — adding this makes it the main account.</span>
          )}
        </div>

        {/* Step 3 — credentials */}
        <div className="grid gap-2 md:grid-cols-3">
          <Field label="Nickname">
            <Input value={form.nickname} onChange={(e) => set("nickname", e.target.value)} placeholder={form.account_type} />
          </Field>
          {isMT5 ? (
            <>
              <Field label="Login">
                <Input value={form.login} onChange={(e) => set("login", e.target.value)} />
              </Field>
              <Field label="Server">
                <Input value={form.server} onChange={(e) => set("server", e.target.value)} placeholder="e.g. ICMarkets-Live02" />
              </Field>
              <Field label="Password">
                <Input type="password" value={form.password} onChange={(e) => set("password", e.target.value)} />
              </Field>
            </>
          ) : (
            <>
              <Field label="API Key">
                <Input value={form.api_key} onChange={(e) => set("api_key", e.target.value)} />
              </Field>
              <Field label="API Secret">
                <Input type="password" value={form.secret} onChange={(e) => set("secret", e.target.value)} />
              </Field>
            </>
          )}
          <Field label="Lot Mode">
            <Select value={form.risk_scaling_mode} onChange={(e) => set("risk_scaling_mode", e.target.value)}>
              {MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </Select>
          </Field>
          <Field label={form.risk_scaling_mode === "proportional" ? "Equity scale" : "Multiplier / Risk %"}>
            <Input type="number" step="0.05" value={form.scale_factor} onChange={(e) => set("scale_factor", e.target.value)} />
          </Field>
        </div>

        <div className="mt-2 flex items-center gap-2">
          <label className="flex items-center gap-1 text-[11px] text-meruem-muted">
            Funded balance (pre-live)
            <Input type="number" className="w-32" value={form.balance} onChange={(e) => set("balance", e.target.value)} placeholder="auto / live" />
          </label>
          <Button size="sm" variant="outline" onClick={testConnection} disabled={busy}>Test Connection</Button>
          <Button size="sm" variant="success" onClick={connect} disabled={busy}>
            {busy ? "Working…" : "Save Connection"}
          </Button>
        </div>
        {msg && (
          <p className={cn("mt-2 text-[11px]", msg.ok ? "text-meruem-green" : "text-meruem-red")}>
            {msg.ok ? "● " : "● "}{msg.text}
          </p>
        )}
      </div>
    </Card>
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