import { useEffect, useState } from "react";
import {
  AgentCatalogEntry,
  AgentStatus,
  api,
  LLMProvider,
  OllamaModels,
  ProviderTest,
  VaultStatus,
} from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type Settings = Record<string, unknown>;

export function VaultPanel() {
  const [status, setStatus] = useState<VaultStatus>({ slots: [], providers: [] });
  const [settings, setSettings] = useState<Settings>({});
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [msgs, setMsgs] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [test, setTest] = useState<Record<string, { running: boolean; ok: boolean | null; error?: string }>>(
    {},
  );
  const [ollamaUrl, setOllamaUrl] = useState("http://127.0.0.1:11434");
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaError, setOllamaError] = useState("");
  const [ollamaMsg, setOllamaMsg] = useState("");
  const [agents, setAgents] = useState<AgentStatus | null>(null);
  const [catalog, setCatalog] = useState<AgentCatalogEntry[]>([]);
  const [agentModels, setAgentModels] = useState<Record<string, string>>({});
  const [agentDraft, setAgentDraft] = useState<Record<string, string>>({});
  const [agentLocked, setAgentLocked] = useState<Record<string, boolean>>({});
  const [agentModelSrc, setAgentModelSrc] = useState<Record<string, string[]>>({});
  const [agentSaving, setAgentSaving] = useState(false);
  const [agentMsg, setAgentMsg] = useState("");

  const [draft, setDraft] = useState({
    id: "", name: "", base_url: "https://api.openai.com/v1", model: "", enabled: true,
  });

  const load = async () => {
    try {
      const s = await api.get<Settings>("/settings");
      setSettings(s);
      const provs = (s.llm_providers as LLMProvider[]) ?? [];
      const vs = await api.get<VaultStatus>("/vault/status");
      setStatus({ slots: vs.slots, providers: provs });
      setOllamaUrl(String(s.ollama_url ?? "http://127.0.0.1:11434"));
    } catch (e) {
      setErrors((m) => ({ ...m, _load: e instanceof Error ? e.message : String(e) }));
    }
  };

  const loadAgents = async () => {
    try {
      setAgents(await api.get<AgentStatus>("/agents/status"));
    } catch (e) {
      setErrors((m) => ({ ...m, _agents: e instanceof Error ? e.message : String(e) }));
    }
  };

  const loadCatalog = async () => {
    try {
      const cat = await api.get<{ agents: AgentCatalogEntry[] }>("/agents/catalog");
      setCatalog(cat.agents);
      const models: Record<string, string[]> = {};
      try {
        const o = await api.get<{ models: string[] }>("/providers/ollama/models");
        models.ollama = o.models;
      } catch {
        models.ollama = [];
      }
      setAgentModelSrc(models);
      const saved = Object.fromEntries(cat.agents.map((a) => [a.id, a.assigned_model ?? ""]));
      setAgentModels(saved);
      setAgentDraft({ ...saved });
      // Fields start locked — edit only after explicitly unlocking.
      setAgentLocked(Object.fromEntries(cat.agents.map((a) => [a.id, true])));
    } catch (e) {
      setErrors((m) => ({ ...m, _catalog: e instanceof Error ? e.message : String(e) }));
    }
  };

  /** Save ALL agent model drafts in one explicit action (no autosave). */
  const saveAgentModels = async () => {
    setAgentSaving(true);
    setAgentMsg("");
    try {
      await api.put("/settings", { key: "agent_models", value: { ...agentDraft } });
      setAgentModels({ ...agentDraft });
      // Lock everything back up after a successful save.
      setAgentLocked(Object.fromEntries(Object.keys(agentDraft).map((id) => [id, true])));
      setAgentMsg("Model assignments saved — take effect on next agent run");
      setTimeout(() => setAgentMsg(""), 3500);
    } catch (e) {
      setErrors((m) => ({ ...m, _agent: e instanceof Error ? e.message : String(e) }));
    } finally {
      setAgentSaving(false);
    }
  };

  /** Save ONE agent's field immediately (still requires unlocking first). */
  const saveAgentModel = async (_id: string) => {
    setAgentSaving(true);
    setAgentMsg("");
    try {
      await api.put("/settings", { key: "agent_models", value: { ...agentDraft } });
      setAgentModels({ ...agentDraft });
      setAgentMsg("Model assignment saved — take effect on next agent run");
      setTimeout(() => setAgentMsg(""), 3000);
    } catch (e) {
      setErrors((m) => ({ ...m, _agent: e instanceof Error ? e.message : String(e) }));
    } finally {
      setAgentSaving(false);
    }
  };

  useEffect(() => {
    load();
    loadAgents();
    loadCatalog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const configured = (keyRef: string) => {
    const [ns, user] = keyRef.split(":");
    return status.slots.some((s) => s.namespace === ns && s.username === user && s.configured);
  };

  const storedSlot = (keyRef: string) => {
    const [ns, user] = keyRef.split(":");
    return status.slots.find((s) => s.namespace === ns && s.username === user && s.configured);
  };

  const deleteKey = async (keyRef: string, id: string) => {
    const [ns, user] = keyRef.split(":");
    if (!window.confirm(`Delete the stored API key for "${id}" from the OS keychain?`)) return;
    try {
      await api.del(`/vault/${ns}/${user}`);
      setKeys((k) => ({ ...k, [id]: "" }));
      setTest((t) => ({ ...t, [id]: { running: false, ok: null, error: "" } }));
      setMsgs((m) => ({ ...m, [id]: "Key deleted from OS keychain" }));
      setTimeout(() => setMsgs((m) => ({ ...m, [id]: "" })), 2500);
      load();
    } catch (e) {
      setErrors((m) => ({ ...m, _save: e instanceof Error ? e.message : String(e) }));
    }
  };

  const saveProviders = async (providers: LLMProvider[]) => {
    await api.put("/settings", { key: "llm_providers", value: providers });
    setStatus((s) => ({ ...s, providers }));
    setSettings((s) => ({ ...s, llm_providers: providers }));
  };

  const updateProvider = (id: string, patch: Partial<LLMProvider>) => {
    const providers = status.providers.map((p) => (p.id === id ? { ...p, ...patch } : p));
    saveProviders(providers).catch((e) =>
      setErrors((m) => ({ ...m, _save: e instanceof Error ? e.message : String(e) })));
  };

  const removeProvider = async (id: string) => {
    await saveProviders(status.providers.filter((p) => p.id !== id));
  };

  const moveProvider = async (index: number, dir: -1 | 1) => {
    const list = [...status.providers];
    const target = index + dir;
    if (target < 0 || target >= list.length) return;
    [list[index], list[target]] = [list[target], list[index]];
    await saveProviders(list);
  };

  const addProvider = async () => {
    if (!draft.id.trim() || !draft.name.trim() || !draft.base_url.trim()) return;
    const id = draft.id.trim().toLowerCase().replace(/\s+/g, "-");
    const keyRef = `llm:${draft.id.trim()}`;
    const provider: LLMProvider = {
      id,
      name: draft.name.trim(),
      base_url: draft.base_url.trim(),
      model: draft.model.trim(),
      enabled: draft.enabled,
      key_ref: keyRef,
    };
    const providers = [...status.providers, provider];
    await saveProviders(providers).catch(() => {});
    setDraft({ id: "", name: "", base_url: "https://api.openai.com/v1", model: "", enabled: true });
  };

  const saveKey = async (keyRef: string, id: string) => {
    const [ns, user] = keyRef.split(":");
    const secret = keys[id];
    if (!secret) return;
    await api.post("/vault", { namespace: ns, username: user, secret });
    setMsgs((m) => ({ ...m, [id]: "Key saved to OS keychain" }));
    setTimeout(() => setMsgs((m) => ({ ...m, [id]: "" })), 2500);
    load();
  };

  const testApiKey = async (id: string) => {
    setTest((t) => ({ ...t, [id]: { running: true, ok: null } }));
    try {
      const res = await api.post<ProviderTest>(`/providers/${id}/test`);
      setTest((t) => ({
        ...t,
        [id]: { running: false, ok: res.ok, error: res.error },
      }));
    } catch (e) {
      setTest((t) => ({
        ...t,
        [id]: { running: false, ok: false, error: e instanceof Error ? e.message : String(e) },
      }));
    }
  };

  const testAllKeys = async () => {
    const withKeys = status.providers.filter((p) => configured(p.key_ref));
    await Promise.all(withKeys.map((p) => testApiKey(p.id)));
  };

  const saveOllamaUrl = async () => {
    await api.put("/settings", { key: "ollama_url", value: ollamaUrl });
    setSettings((s) => ({ ...s, ollama_url: ollamaUrl }));
    setOllamaMsg("Ollama endpoint saved");
    setTimeout(() => setOllamaMsg(""), 2500);
  };

  const setModel = async (key: string, value: string) => {
    setSettings((s) => ({ ...s, [key]: value }));
    await api.put("/settings", { key, value });
  };

  const pullModels = async () => {
    setOllamaError("");
    setOllamaMsg("Listing models…");
    const res = await api.get<OllamaModels>("/ollama/models");
    if (res.error) {
      setOllamaError(res.error);
      setOllamaMsg("Could not reach Ollama");
      return;
    }
    setOllamaModels(res.models);
    setOllamaMsg(`${res.models.length} model(s) found`);
  };

  return (
    <Card className="h-full" title="Vault · Credentials & Models">
      {errors._load && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{errors._load}</p>
      )}
      {errors._save && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{errors._save}</p>
      )}
      {errors._agents && (
        <p className="mb-2 rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{errors._agents}</p>
      )}

      {/* ===== AI Agents activity ===== */}
      <div className="mb-4 space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">AI Agents Active</h4>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={loadAgents}>Refresh status</Button>
            <Badge tone={agents?.cloud_ready ? "green" : "neutral"}>
              {agents?.cloud_ready ? "cloud ready" : "cloud: key missing"}
            </Badge>
          </div>
        </div>

        {/* Agent cards */}
        {agents && (
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
            {agents.agents.map((a) => (
              <div key={a.id} className="rounded-md border border-meruem-border/60 p-2">
                <div className="flex items-center gap-1.5">
                  <span
                    className={cn(
                      "h-2 w-2 shrink-0 rounded-full",
                      a.ready ? "bg-meruem-green" : "bg-meruem-red",
                    )}
                  />
                  <span className="text-xs font-medium text-meruem-text">{a.name}</span>
                  <Badge tone={a.ready ? "green" : "red"}>{a.ready ? "running" : "blocked"}</Badge>
                </div>
                <p className="mt-1 text-[10px] text-meruem-muted">{a.role}</p>
                <p className="mt-1 truncate font-mono text-[11px] text-meruem-accent" title={a.model}>
                  {a.model || "—"}
                </p>
                <p className="mt-1 text-[10px] text-meruem-muted">{a.engine}</p>
                {a.note && <p className="mt-1 text-[10px] text-meruem-muted">{a.note}</p>}
              </div>
            ))}
          </div>
        )}

        {/* Provider + Ollama readout */}
        {agents && (
          <div className="grid gap-2 md:grid-cols-2">
            {/* Cloud providers */}
            <div className="rounded-md border border-meruem-border/60 p-2">
              <p className="mb-1 text-[11px] font-medium text-meruem-text">Cloud providers</p>
              <ul className="space-y-0.5">
                {agents.providers.map((p) => (
                  <li key={p.id} className="flex items-center gap-1.5 text-[11px]">
                    <span
                      className={cn(
                        "h-1.5 w-1.5 shrink-0 rounded-full",
                        p.ready ? "bg-meruem-green" : p.enabled ? "bg-meruem-yellow" : "bg-meruem-muted",
                      )}
                    />
                    <span className="text-meruem-text">{p.name}</span>
                    <span className="truncate font-mono text-meruem-muted">{p.model}</span>
                    <span className="ml-auto text-meruem-muted">
                      {p.enabled ? (p.key_set ? "key set" : "no key") : "off"}
                    </span>
                  </li>
                ))}
                {agents.providers.length === 0 && (
                  <li className="text-[11px] text-meruem-muted">No providers configured.</li>
                )}
              </ul>
            </div>

            {/* Ollama */}
            <div className="rounded-md border border-meruem-border/60 p-2">
              <p className="mb-1 flex items-center gap-1.5 text-[11px] font-medium text-meruem-text">
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    agents.ollama.reachable ? "bg-meruem-green" : "bg-meruem-red",
                  )}
                />
                Ollama (local)
                <span className="ml-auto font-normal text-meruem-muted">
                  {agents.ollama.reachable
                    ? `${agents.ollama.models.length} model(s) · ${agents.ollama.latency_ms}ms`
                    : "unreachable"}
                </span>
              </p>
              {agents.ollama.reachable ? (
                <>
                  <p className="text-[11px] text-meruem-muted">
                    Scanner: <span className="font-mono text-meruem-accent">{agents.ollama.scanner_model || "—"}</span>
                  </p>
                  <p className="text-[11px] text-meruem-muted">
                    Validator: <span className="font-mono text-meruem-accent">{agents.ollama.validator_model || "—"}</span>
                  </p>
                  {agents.ollama.models.length > 0 && (
                    <p className="mt-1 max-h-14 overflow-y-auto text-[10px] text-meruem-muted">
                      {agents.ollama.models.join(" · ")}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-[11px] text-meruem-red">{agents.ollama.error || "Ollama not reachable"}</p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ===== Per-Agent Model Assignment (task #12) ===== */}
      <div className="mb-4 space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">
            Per-Agent Models
          </h4>
          <div className="flex items-center gap-2">
            {agentMsg && <span className="text-[11px] text-meruem-green">{agentMsg}</span>}
            <Button size="sm" variant="success" onClick={saveAgentModels} disabled={agentSaving}>
              {agentSaving ? "Saving…" : "Save All"}
            </Button>
          </div>
        </div>
        <p className="text-[11px] text-meruem-muted">
          Assign the exact LLM each agent uses. Model names are discovered live
          from your configured cloud providers and local Ollama, but you can type
          any model name. Blank means "use the catalog default".
          Fields are <span className="text-meruem-text">locked</span> until you unlock them —
          nothing is saved until you hit <span className="text-meruem-accent">Save All</span>.
        </p>
        {errors._catalog && (
          <p className="rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{errors._catalog}</p>
        )}
        {errors._agent && (
          <p className="rounded border border-meruem-red/40 bg-meruem-red/10 p-2 text-xs text-meruem-red">{errors._agent}</p>
        )}

        <div className="grid gap-2 md:grid-cols-2">
          {catalog.map((a) => {
            const options = new Set<string>();
            if (a.engine === "local") {
              (agentModelSrc.ollama ?? []).forEach((m) => options.add(m));
            } else {
              status.providers
                .filter((p) => p.enabled && p.model)
                .forEach((p) => options.add(p.model));
            }
            if (a.recommended_model) options.add(a.recommended_model);
            options.add(a.default_model);
            return (
              <div key={a.id} className="rounded-md border border-meruem-border/60 p-2">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-meruem-text">{a.name}</span>
                  <Badge tone={a.engine === "local" ? "gold" : "green"}>{a.engine}</Badge>
                  <Badge tone="neutral">{a.refresh}</Badge>
                </div>
                <p className="mt-0.5 text-[10px] text-meruem-muted">{a.role}</p>
                <p className="mt-1 flex items-center gap-1 text-[10px] text-meruem-muted" title={a.tip}>
                  <span className="text-meruem-accent">Tip:</span>
                  <span className="line-clamp-2">{a.tip}</span>
                </p>
                <div className="mt-1.5 flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setAgentLocked((m) => ({ ...m, [a.id]: !m[a.id] }))}
                    className={cn(
                      "shrink-0 rounded border px-1.5 py-1 text-[10px] font-medium transition-colors",
                      agentLocked[a.id]
                        ? "border-meruem-border bg-meruem-bg text-meruem-muted hover:text-meruem-text"
                        : "border-amber-500/50 bg-amber-500/10 text-amber-400",
                    )}
                    title={
                      agentLocked[a.id]
                        ? "Locked — unlock to edit this field"
                        : "Unlocked — click to lock again"
                    }
                  >
                    {agentLocked[a.id] ? "🔒" : "🔓"}
                  </button>
                  <input
                    list="agent-model-options"
                    className={cn(
                      "w-full rounded-md border bg-meruem-bg px-2.5 py-1.5 text-sm text-meruem-text placeholder:text-meruem-muted focus:outline-none",
                      agentLocked[a.id]
                        ? "cursor-not-allowed border-meruem-border/50 text-meruem-muted opacity-70"
                        : "border-meruem-accent/60 focus:border-meruem-accent",
                    )}
                    value={agentDraft[a.id] ?? ""}
                    disabled={agentLocked[a.id]}
                    placeholder={a.engine === "local" ? "local model" : "cloud model (default)"}
                    onChange={(e) =>
                      setAgentDraft((d) => ({ ...d, [a.id]: e.target.value }))
                    }
                  />
                </div>
                <div className="mt-1 flex items-center gap-1.5">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={agentLocked[a.id]}
                    title={
                      agentLocked[a.id]
                        ? "Unlock the field first, then save"
                        : "Save this field immediately"
                    }
                    onClick={() => saveAgentModel(a.id)}
                  >
                    Save
                  </Button>
                  {(agentDraft[a.id] ?? "") !== (agentModels[a.id] ?? "") && (
                    <span className="text-[10px] text-amber-400">unsaved edit</span>
                  )}
                </div>
                <p className="mt-1 text-[10px] text-meruem-muted">
                  active: <span className="font-mono text-meruem-accent">{agentModels[a.id] || a.default_model}</span>
                  {(agentDraft[a.id] ?? "") !== "" && (agentDraft[a.id] ?? "") !== a.recommended_model && a.recommended_model && (
                    <span> · recommended <span className="font-mono">{a.recommended_model}</span></span>
                  )}
                </p>
              </div>
            );
          })}
        </div>
        <datalist id="agent-model-options">
          {Array.from(new Set([
            ...catalog.map((a) => a.recommended_model),
            ...catalog.map((a) => a.default_model),
            ...(agentModelSrc.ollama ?? []),
            ...status.providers.filter((p) => p.enabled && p.model).map((p) => p.model),
          ].filter(Boolean))).map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Universal LLM providers */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">
              API Key Bank (encrypted)
            </h4>
            <div className="flex items-center gap-1.5">
              <Badge tone={status.providers.length ? "blue" : "neutral"}>
                {status.providers.length} provider(s)
              </Badge>
              <Button size="sm" variant="outline" onClick={testAllKeys}>
                Test all keys
              </Button>
            </div>
          </div>
          <p className="text-[11px] text-meruem-muted">
            Keys are stored encrypted in your OS keychain (never in the DB or
            sent to the UI). Press{" "}
            <span className="text-meruem-accent">Test</span> to live-verify the
            key against the provider's API and see a green/red indicator.
            Providers run in order, top to bottom. Local Ollama is the final
            fallback for every agent.
          </p>

          {status.providers.map((p, idx) => (
            <div key={p.id} className="space-y-1.5 rounded-md border border-meruem-border/60 p-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Button size="sm" variant="outline" disabled={idx === 0} onClick={() => moveProvider(idx, -1)}>↑</Button>
                  <Button size="sm" variant="outline" disabled={idx === status.providers.length - 1} onClick={() => moveProvider(idx, 1)}>↓</Button>
                  <span className="text-xs font-medium text-meruem-text">
                    {idx + 1}. {p.name}
                  </span>
                  {/* Live key verification indicator */}
                  {test[p.id]?.running ? (
                    <Badge tone="gold">testing…</Badge>
                  ) : test[p.id]?.ok === true ? (
                    <Badge tone="green">key works</Badge>
                  ) : test[p.id]?.ok === false ? (
                    <Badge tone="red">key rejected</Badge>
                  ) : configured(p.key_ref) ? (
                    <Badge tone="blue">stored</Badge>
                  ) : (
                    <Badge tone="neutral">no key</Badge>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  <Badge tone={p.enabled ? "green" : "red"}>{p.enabled ? "on" : "off"}</Badge>
                  <Button size="sm" variant="danger" onClick={() => removeProvider(p.id)}>×</Button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-1.5">
                <label className="flex items-center gap-1 text-[11px] text-meruem-muted">
                  Base URL
                  <Input value={p.base_url} onChange={(e) => updateProvider(p.id, { base_url: e.target.value })} />
                </label>
                <label className="flex items-center gap-1 text-[11px] text-meruem-muted">
                  Model
                  <Input value={p.model} onChange={(e) => updateProvider(p.id, { model: e.target.value })} />
                </label>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="success" onClick={() => saveKey(p.key_ref, p.id)}>Save API Key</Button>
                <label className="flex flex-1 items-center gap-1 text-[11px] text-meruem-muted">
                  <Input
                    type="password"
                    value={keys[p.id] ?? ""}
                    onChange={(e) => setKeys((k) => ({ ...k, [p.id]: e.target.value }))}
                    placeholder={configured(p.key_ref) ? "New key (enter to replace)" : "Paste API key"}
                  />
                </label>
                <Button size="sm" variant="outline" onClick={() => testApiKey(p.id)}>
                  Test
                </Button>
                <Button size="sm" variant="danger" disabled={!configured(p.key_ref)} onClick={() => deleteKey(p.key_ref, p.id)} title="Delete the stored key from the OS keychain">
                  Delete
                </Button>
                <Button size="sm" variant={p.enabled ? "ghost" : "default"} onClick={() => updateProvider(p.id, { enabled: !p.enabled })}>
                  {p.enabled ? "Disable" : "Enable"}
                </Button>
              </div>
              {(() => {
                const slot = storedSlot(p.key_ref);
                if (slot?.redacted) {
                  return (
                    <p className="text-[11px] text-meruem-muted">
                      Stored in OS keychain (persists across relaunch):{" "}
                      <span className="font-mono text-meruem-accent" title="Masked preview — your key is retained, no need to re-paste">{slot.redacted}</span>
                    </p>
                  );
                }
                return null;
              })()}
              {msgs[p.id] && <p className="text-[11px] text-meruem-green">{msgs[p.id]}</p>}
              {(() => {
                const t = test[p.id];
                if (t?.ok === false && t.error) {
                  return (
                    <p className="text-[11px] text-meruem-red" title={t.error}>
                      {t.error.length > 160 ? `${t.error.slice(0, 160)}…` : t.error}
                    </p>
                  );
                }
                return null;
              })()}
            </div>
          ))}

          <div className="space-y-1.5 rounded-md border border-dashed border-meruem-border p-2">
            <p className="text-[11px] font-medium text-meruem-accent">Add provider</p>
            <div className="grid grid-cols-2 gap-1.5">
              <Input value={draft.id} onChange={(e) => setDraft((d) => ({ ...d, id: e.target.value }))} placeholder="id (e.g. openrouter)" />
              <Input value={draft.name} onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))} placeholder="Name" />
              <Input value={draft.base_url} onChange={(e) => setDraft((d) => ({ ...d, base_url: e.target.value }))} placeholder="Base URL (OpenAI-compatible)" className="col-span-2" />
              <Input value={draft.model} onChange={(e) => setDraft((d) => ({ ...d, model: e.target.value }))} placeholder="Model name" className="col-span-2" />
            </div>
            <Button size="sm" className="w-full" onClick={addProvider} disabled={!draft.id.trim() || !draft.name.trim() || !draft.base_url.trim()}>
              Add Provider
            </Button>
          </div>
        </div>

        {/* Ollama local connector */}
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-meruem-accent">
            Ollama (local) Connector
          </h4>
          <p className="text-[11px] text-meruem-muted">
            Runs the scanner and is the final validator fallback. Runs entirely
            on your hardware.
          </p>
          <div className="flex items-end gap-2">
            <Field label="Ollama endpoint">
              <Input value={ollamaUrl} onChange={(e) => setOllamaUrl(e.target.value)} />
            </Field>
            <Button size="sm" onClick={saveOllamaUrl}>Save</Button>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={pullModels}>
              {ollamaModels.length ? "Refresh models" : "List models"}
            </Button>
            {ollamaMsg && <span className="text-[11px] text-meruem-muted">{ollamaMsg}</span>}
          </div>
          {ollamaError && <p className="text-[11px] text-meruem-red">{ollamaError}</p>}

          <div className="grid grid-cols-1 gap-2">
            <Field label="Scanner model (picks set-ups)">
              <Select
                value={String(settings.ollama_scanner_model ?? "")}
                onChange={(e) => setModel("ollama_scanner_model", e.target.value)}
              >
                <option value="">— select / type —</option>
                {ollamaModels.map((m) => <option key={m} value={m}>{m}</option>)}
              </Select>
            </Field>
            <Field label="Validator model (final fallback)">
              <Select
                value={String(settings.ollama_validator_model ?? "")}
                onChange={(e) => setModel("ollama_validator_model", e.target.value)}
              >
                <option value="">— select / type —</option>
                {ollamaModels.map((m) => <option key={m} value={m}>{m}</option>)}
              </Select>
            </Field>
          </div>

          <div className="rounded-md border border-meruem-border/60 p-2 text-[11px] text-meruem-muted">
            <p className="mb-1 text-meruem-text">How the pipeline prefers models</p>
            <ol className="list-decimal space-y-0.5 pl-4">
              <li><span className="text-meruem-accent">Scanner</span> (Ollama) → builds a trade candidate.</li>
              <li><span className="text-meruem-accent">Validators/Refine/QA</span> → cloud providers in list order, then Ollama (local).</li>
              <li>All fail → system pauses (PAUSED_LLM_LIMIT).</li>
              <li>Cloud providers need a key saved above.</li>
            </ol>
            <p className="mt-1">
              Tip: pick a local model that fits your VRAM/RAM — e.g. qwen2.5:3b
              for modest GPUs, 7b/13b on larger ones.
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block flex-1 space-y-1">
      <span className="text-[11px] text-meruem-muted">{label}</span>
      {children}
    </label>
  );
}