import { useState } from "react";
import { TopNav } from "@/components/TopNav";
import { ChartPanel } from "@/components/ChartPanel";
import { WatchlistPanel } from "@/components/WatchlistPanel";
import { TradesPanel } from "@/components/TradesPanel";
import { StrategyLibraryPanel } from "@/components/StrategyLibraryPanel";
import { DashboardPanel } from "@/components/DashboardPanel";
import { RiskPanel } from "@/components/RiskPanel";
import { SignalsPanel } from "@/components/SignalsPanel";
import { VaultPanel } from "@/components/VaultPanel";
import { BrokersPanel } from "@/components/BrokersPanel";
import { NewsPanel } from "@/components/NewsPanel";
import { RegimePanel } from "@/components/RegimePanel";
import { Asset } from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "chart" | "signals" | "trades" | "strategies" | "dashboard" | "risk" | "vault" | "brokers";

const TABS: { id: Tab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "chart", label: "Chart" },
  { id: "signals", label: "Signals" },
  { id: "trades", label: "Trades" },
  { id: "brokers", label: "Brokers" },
  { id: "risk", label: "Risk Settings" },
  { id: "strategies", label: "Strategies" },
  { id: "vault", label: "Vault" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("chart");
  const [active, setActive] = useState<Asset | null>(null);

  return (
    <div className="flex h-screen flex-col">
      <TopNav />
      <nav className="flex flex-wrap gap-1 border-b border-meruem-border bg-meruem-panel/40 px-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "rounded-t-md px-3 py-2 text-xs font-medium transition-colors",
              tab === t.id
                ? "border-b-2 border-meruem-accent text-meruem-text"
                : "text-meruem-muted hover:text-meruem-text",
            )}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main
        className={`grid flex-1 gap-3 overflow-auto p-3 ${
          tab === "chart" ? "lg:grid-cols-[280px_1fr_320px]" : "lg:grid-cols-1"
        }`}
      >
        {tab === "chart" && <WatchlistPanel active={active} onSelect={setActive} />}

        <section className="min-h-0">
          {tab === "chart" && (
            <ChartPanel symbol={active?.symbol ?? "EURUSD"} onSelectAsset={setActive} />
          )}
          {tab === "signals" && <SignalsPanel />}
          {tab === "trades" && <TradesPanel />}
          {tab === "strategies" && <StrategyLibraryPanel />}
          {tab === "dashboard" && <DashboardPanel />}
          {tab === "risk" && <RiskPanel />}
          {tab === "vault" && <VaultPanel />}
          {tab === "brokers" && <BrokersPanel />}
        </section>

        {tab === "chart" && (
          <div className="grid min-h-0 grid-rows-[1fr_auto] gap-3">
            <NewsPanel symbol={active?.symbol} />
            <RegimePanel symbol={active?.symbol} />
          </div>
        )}
      </main>
    </div>
  );
}