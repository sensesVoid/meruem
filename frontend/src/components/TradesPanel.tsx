import { useEffect, useState } from "react";
import { api, Trade } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const CLOSE_REASON_LABEL: Record<string, string> = {
  sl_hit: "SL hit",
  tp_hit: "TP hit",
  trail: "Trailing stop",
  broker: "Broker",
  manual: "Manual",
};

export function TradesPanel() {
  const [trades, setTrades] = useState<Trade[]>([]);

  const load = async () => setTrades(await api.get<Trade[]>("/trades?open_only=false"));
  useEffect(() => {
    load();
  }, []);

  const close = async (t: Trade) => {
    await api.put(`/trades/${t.id}/close`, { reason: "manual" });
    load();
  };

  const remove = async (t: Trade) => {
    if (!window.confirm(`Delete this trade record?${t.broker_order_id ? " (broker order #" + t.broker_order_id + ")" : ""}`)) return;
    await api.del(`/trades/${t.id}`);
    load();
  };

  const clearPhantoms = async () => {
    const res = await api.post<{ ok: boolean; removed: number }>("/trades/clear-phantoms", {});
    load();
    if (res.removed > 0) window.alert(`Removed ${res.removed} phantom trade record(s) that never reached the broker.`);
  };

  const why = (t: Trade) => {
    const closeR = t.close_reason ? (CLOSE_REASON_LABEL[t.close_reason] ?? t.close_reason) : "";
    const parts: string[] = [];
    if (closeR) parts.push(closeR);
    if (t.loss_reason) parts.push(`Why lost: ${t.loss_reason}`);
    if (t.broker_error && t.broker_error !== "") parts.push(`Broker error: ${t.broker_error}`);
    return parts.join(" · ");
  };

  return (
    <Card
      className="h-full"
      title="Trades / Position Manager"
      action={
        <div className="flex items-center gap-1.5">
          <Button size="sm" variant="outline" onClick={clearPhantoms} title="Delete records that never confirmed on a broker (no symbol / order id)">
            Clear phantom
          </Button>
          <Button size="sm" variant="outline" onClick={load}>Refresh</Button>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-meruem-border text-[11px] uppercase text-meruem-muted">
              <th className="py-1.5 pr-2">Acct</th>
              <th className="py-1.5 pr-2">Sym</th>
              <th className="py-1.5 pr-2">Dir</th>
              <th className="py-1.5 pr-2 text-right">Entry</th>
              <th className="py-1.5 pr-2 text-right">Exit</th>
              <th className="py-1.5 pr-2 text-right">SL</th>
              <th className="py-1.5 pr-2 text-right">TP1</th>
              <th className="py-1.5 pr-2 text-right">Lot</th>
              <th className="py-1.5 pr-2">Mode</th>
              <th className="py-1.5 pr-2">Why</th>
              <th className="py-1.5 pr-2">Status</th>
              <th className="py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id} className="border-b border-meruem-border/50">
                <td className="py-1.5 pr-2 font-mono text-xs">{t.account_id.slice(-8)}</td>
                <td className="py-1.5 pr-2 font-mono text-xs">{t.symbol || "—"}</td>
                <td className="py-1.5 pr-2">
                  <Badge tone={t.direction === "BUY" ? "green" : "red"}>{t.direction}</Badge>
                </td>
                <td className="py-1.5 pr-2 text-right font-mono">{t.entry_price}</td>
                <td className="py-1.5 pr-2 text-right font-mono">{t.exit_price ?? "-"}</td>
                <td className="py-1.5 pr-2 text-right font-mono text-meruem-red">{t.sl}</td>
                <td className="py-1.5 pr-2 text-right font-mono text-meruem-green">{t.tp1 ?? "-"}</td>
                <td className="py-1.5 pr-2 text-right font-mono">{t.lot_size}</td>
                <td className="py-1.5 pr-2">
                  <Badge tone={t.mode === "auto" ? "blue" : "neutral"}>{String(t.mode ?? "-").slice(0, 9)}</Badge>
                </td>
                <td className="py-1.5 pr-2 text-[10px] text-meruem-muted max-w-[180px]">
                  {why(t) || (t.pnl !== undefined && t.pnl < 0 && !t.loss_reason ? "Pending auto-LTR review" : "—")}
                </td>
                <td className="py-1.5 pr-2">
                  {t.broker_error ? (
                    <span title={t.broker_error}>
                      <Badge tone="red">fail</Badge>
                    </span>
                  ) : t.closed_at ? (
                    <Badge tone={t.pnl !== undefined && t.pnl >= 0 ? "green" : "red"}>
                      {t.pnl?.toFixed(2) ?? "closed"}
                    </Badge>
                  ) : (
                    <Button size="sm" variant="danger" onClick={() => close(t)}>Close</Button>
                  )}
                </td>
                <td className="py-1.5">
                  <button
                    className="text-meruem-muted hover:text-meruem-red"
                    onClick={() => remove(t)}
                    title="Delete this trade record"
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={12} className="py-4 text-center text-xs text-meruem-muted">
                  No trades recorded yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}