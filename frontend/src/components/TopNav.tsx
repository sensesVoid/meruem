import { useState } from "react";
import { api, SystemStatus } from "@/lib/api";
import { usePolling } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

const STATUS_TONE: Record<string, "green" | "red" | "gold" | "blue" | "neutral"> = {
  READY: "green",
  PAUSED_LLM_LIMIT: "red",
  PAUSED_RISK: "red",
  DISABLED: "gold",
};

export function TopNav() {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  usePolling(() => {
    api.get<SystemStatus>("/status").then(setStatus).catch(() => setStatus(null));
  }, 10000);

  return (
    <header className="flex items-center justify-between border-b border-meruem-border bg-meruem-panel/60 px-4 py-2">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full bg-meruem-gold" />
        <h1 className="text-sm font-bold tracking-widest text-meruem-text">MERUEM</h1>
        <span className="text-[11px] text-meruem-muted">AI Trading Desktop</span>
      </div>
      <div className="flex items-center gap-2">
        {status ? (
          <>
            <Badge tone={status.risk.allowed ? "blue" : "red"}>
              Risk: {status.risk.allowed ? "OK" : "BLOCKED"}
            </Badge>
            <Badge
              tone={STATUS_TONE[status.system_status ?? "READY"] ?? "neutral"}
            >
              {status.system_status ?? "READY"}
            </Badge>
            <Badge tone={status.auto_execution_enabled ? "green" : "gold"}>
              {status.auto_execution_enabled ? "AUTO" : "MANUAL"}
            </Badge>
          </>
        ) : (
          <Badge tone="red">BACKEND OFFLINE — run .\start.ps1</Badge>
        )}
      </div>
    </header>
  );
}