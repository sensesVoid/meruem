import { useEffect, useState } from "react";
import { api, Asset } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";

interface WatchlistPanelProps {
  active: Asset | null;
  onSelect: (a: Asset) => void;
}

const classTone = (c: string) =>
  c === "crypto" ? "gold" : c === "forex" ? "blue" : "neutral";

export function WatchlistPanel({ active, onSelect }: WatchlistPanelProps) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [symbol, setSymbol] = useState("");
  const [klass, setKlass] = useState("forex");
  const [tf, setTf] = useState("15m,1h");

  const load = async () => setAssets(await api.get<Asset[]>("/assets"));
  useEffect(() => {
    load();
  }, []);

  const add = async () => {
    if (!symbol.trim()) return;
    await api.post("/assets", {
      id: crypto.randomUUID(),
      symbol: symbol.trim().toUpperCase(),
      asset_class: klass,
      timeframe_watchlist: tf,
    });
    setSymbol("");
    load();
  };

  const remove = async (id: string) => {
    await api.del(`/assets/${id}`);
    load();
  };

  return (
    <Card
      className="h-full"
      title="Asset Watchlist"
      action={<span className="text-[11px] text-meruem-muted">{assets.length} tracked</span>}
    >
      <div className="space-y-1.5">
        {assets.map((a) => (
          <div
            key={a.id}
            onClick={() => onSelect(a)}
            className={`group flex cursor-pointer items-center justify-between rounded-md border px-2 py-1.5 transition-colors ${
              active?.id === a.id
                ? "border-meruem-accent bg-meruem-accent/10"
                : "border-meruem-border bg-meruem-bg hover:bg-meruem-panel"
            }`}
          >
            <div>
              <div className="font-mono text-sm">{a.symbol}</div>
              <div className="text-[11px] text-meruem-muted">{a.timeframe_watchlist}</div>
            </div>
            <div className="flex items-center gap-1.5">
              <Badge tone={classTone(a.asset_class)}>{a.asset_class}</Badge>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  remove(a.id);
                }}
                className="hidden text-meruem-red group-hover:block"
                title="Remove"
              >
                x
              </button>
            </div>
          </div>
        ))}
        {assets.length === 0 && (
          <p className="py-4 text-center text-xs text-meruem-muted">No assets tracked yet</p>
        )}
      </div>

      <div className="mt-3 space-y-1.5 border-t border-meruem-border pt-2">
        <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol e.g. EURUSD" />
        <div className="flex gap-1.5">
          <select
            value={klass}
            onChange={(e) => setKlass(e.target.value)}
            className="flex-1 rounded-md border border-meruem-border bg-meruem-bg px-2 py-1 text-xs"
          >
            <option value="forex">Forex</option>
            <option value="crypto">Crypto</option>
            <option value="commodities">Commodities</option>
          </select>
          <select
            value={tf}
            onChange={(e) => setTf(e.target.value)}
            className="flex-1 rounded-md border border-meruem-border bg-meruem-bg px-2 py-1 text-xs"
          >
            <option value="15m,1h">15m / 1h</option>
            <option value="1h,4h">1h / 4h</option>
            <option value="M1,5m,1h">M1 / 5m / 1h</option>
          </select>
        </div>
        <Button size="sm" className="w-full" onClick={add}>
          + Track Asset
        </Button>
      </div>
    </Card>
  );
}