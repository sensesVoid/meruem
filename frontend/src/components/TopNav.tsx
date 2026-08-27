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

interface QiMeraStatus {
  frontend_url: string;
  api_url: string;
  reachable: boolean;
  export_dir: string;
  export_dir_exists: boolean;
  assets: string[];
}

export function TopNav() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [qimera, setQimera] = useState<QiMeraStatus | null>(null);

  usePolling(() => {
    api.get<SystemStatus>("/status").then(setStatus).catch(() => setStatus(null));
  }, 10000);

  usePolling(() => {
    api.get<QiMeraStatus>("/qimera/status").then(setQimera).catch(() => setQimera(null));
  }, 15000);

  return (
    <header className="flex items-center justify-between border-b border-meruem-border bg-meruem-panel/60 px-4 py-2">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full bg-meruem-gold" />
        <h1 className="text-sm font-bold tracking-widest text-meruem-text">MERUEM</h1>
        <span className="text-[11px] text-meruem-muted">AI Trading Desktop</span>
      </div>
      <div className="flex items-center gap-2">
        <a
          href={qimera?.frontend_url ?? "http://127.0.0.1:1421"}
          target="_blank"
          rel="noreferrer"
          title={
            qimera
              ? `QiMera ${qimera.reachable ? "online" : "offline"} — ${qimera.assets.length} asset(s) with artifacts injected`
              : "QiMera extension (start qimera\\start.ps1)"
          }
        >
          <Badge tone={qimera?.reachable ? "green" : "neutral"}>
            QiMera↗
          </Badge>
        </a>
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