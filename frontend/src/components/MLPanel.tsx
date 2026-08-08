import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DatasetsPanel } from "@/components/DatasetsPanel";
import {
  RefreshCw,
  Play,
  Trash2,
  CheckCircle,
  AlertCircle,
  Clock,
  BarChart2,
  Database,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

interface MLStatus {
  status: "not_trained" | "training" | "ready" | "error";
  trained_at: string | null;
  n_samples: number;
  n_features: number;
  feature_names: string[];
  train_accuracy: number;
  test_accuracy: number;
  strategy_ids: string[];
  error: string | null;
  model_exists: boolean;
  model_path: string;
}

const STATUS_TONE: Record<string, "green" | "gold" | "red" | "blue" | "neutral"> = {
  ready: "green",
  training: "gold",
  error: "red",
  not_trained: "neutral",
};

const STATUS_ICON: Record<string, React.ReactNode> = {
  ready: <CheckCircle className="h-4 w-4 text-emerald-400" />,
  training: <RefreshCw className="h-4 w-4 text-amber-400 animate-spin" />,
  error: <AlertCircle className="h-4 w-4 text-rose-400" />,
  not_trained: <Clock className="h-4 w-4 text-meruem-muted" />,
};

export function MLPanel() {
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  // Training data (Parquet datasets) section — datasets feed ML training features.
  const [showDatasets, setShowDatasets] = useState(false);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<MLStatus>("/ml/status");
      setMlStatus(res);
    } catch (err: any) {
      setError(err?.message || "Failed to load ML status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setInfo(null);
    try {
      const res = await api.post<MLStatus & { ok: boolean }>("/ml/train", {});
      setInfo(
        `Training complete! ${res.n_samples} samples, ` +
          `Train acc: ${(res.train_accuracy * 100).toFixed(1)}%, ` +
          `Test acc: ${(res.test_accuracy * 100).toFixed(1)}%.`
      );
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  const handleReset = async () => {
    if (!confirm("Delete the trained XGBoost model? The system will fall back to Ollama scanner until retrained.")) return;
    setError(null);
    setInfo(null);
    try {
      await api.del("/ml/model");
      setInfo("Model deleted. System will use Ollama fallback scanner until retrained.");
      await fetchStatus();
    } catch (err: any) {
      setError(err?.message || "Delete failed");
    }
  };

  const statusStr = mlStatus?.status || "not_trained";

  return (
    <Card
      className="h-full"
      title="XGBoost ML Scanner"
      action={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={fetchStatus} disabled={loading}>
            <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
          </Button>
          {mlStatus?.model_exists && (
            <Button size="sm" variant="outline" onClick={handleReset} className="text-rose-400 border-rose-400/30 hover:bg-rose-400/10">
              <Trash2 className="h-3 w-3 mr-1" />
              Reset Model
            </Button>
          )}
          <Button
            size="sm"
            onClick={handleTrain}
            disabled={training}
            className="flex items-center gap-1.5"
          >
            {training ? (
              <RefreshCw className="h-3 w-3 animate-spin" />
            ) : (
              <Play className="h-3 w-3" />
            )}
            {training ? "Training…" : "Train Model"}
          </Button>
        </div>
      }
    >
      {/* Intro */}
      <div className="mb-3 rounded border border-meruem-border/40 bg-meruem-bg/50 p-2.5 text-[11px] text-meruem-muted leading-relaxed">
        <p>
          The <strong className="text-meruem-text">ML Scanner</strong> is a{" "}
          <strong className="text-meruem-text">gradient-boosting classifier</strong> that evaluates
          market conditions across your watchlist and outputs a directional signal —{" "}
          <span className="text-emerald-400 font-semibold">BUY</span>,{" "}
          <span className="text-rose-400 font-semibold">SELL</span>, or{" "}
          <span className="text-meruem-muted font-semibold">NO SIGNAL</span> — based on{" "}
          14 OHLCV-derived indicators plus live regime context (GARCH volatility state, Markov
          turbulence). It only learns from strategies that have cleared the full synthesis and
          backtest gate, so its signals are strictly grounded in your validated playbook.
          The model refreshes automatically every week and can be retrained on demand after
          new strategies are approved. When no model is present, scanning falls back to the
          Ollama LLM scanner seamlessly.
        </p>
      </div>

      {/* Status Card */}
      <div className="mb-3 rounded-md border border-meruem-border bg-meruem-bg p-3 flex items-start gap-3">
        <div className="mt-0.5">{STATUS_ICON[statusStr]}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold">Model Status</span>
            <Badge tone={STATUS_TONE[statusStr]}>{statusStr.replace("_", " ")}</Badge>
          </div>
          {mlStatus?.trained_at && (
            <p className="text-[11px] text-meruem-muted">
              Last trained: {mlStatus.trained_at.slice(0, 19).replace("T", " ")} UTC
            </p>
          )}
          {mlStatus?.error && (
            <p className="mt-1 text-[11px] text-rose-400">{mlStatus.error}</p>
          )}
          {statusStr === "not_trained" && !mlStatus?.error && (
            <p className="text-[11px] text-meruem-muted">
              No model trained yet. Click <strong>Train Model</strong> above after approving at least one green-gate strategy.
            </p>
          )}
        </div>
      </div>

      {/* Accuracy metrics */}
      {mlStatus?.status === "ready" && (
        <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            { label: "Train Accuracy", value: `${(mlStatus.train_accuracy * 100).toFixed(1)}%`, good: mlStatus.train_accuracy >= 0.6 },
            { label: "Test (OOS) Accuracy", value: `${(mlStatus.test_accuracy * 100).toFixed(1)}%`, good: mlStatus.test_accuracy >= 0.55 },
            { label: "Training Samples", value: mlStatus.n_samples.toLocaleString(), good: mlStatus.n_samples >= 50 },
            { label: "Strategies Used", value: mlStatus.strategy_ids.length.toString(), good: mlStatus.strategy_ids.length >= 1 },
          ].map((m) => (
            <div key={m.label} className="rounded border border-meruem-border/40 bg-meruem-bg p-2">
              <span className="block text-[10px] text-meruem-muted mb-0.5">{m.label}</span>
              <span className={`font-mono text-sm font-semibold ${m.good ? "text-emerald-400" : "text-amber-400"}`}>
                {m.value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Feature importance table */}
      {mlStatus?.status === "ready" && mlStatus.feature_names.length > 0 && (
        <div className="mb-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <BarChart2 className="h-4 w-4 text-meruem-accent" />
            <span className="text-xs font-semibold">Feature Set ({mlStatus.n_features} features)</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {mlStatus.feature_names.map((f) => (
              <span key={f} className="rounded border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-[10px] text-indigo-300 font-mono">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* How it works */}
      <div className="rounded border border-meruem-border/40 bg-meruem-bg/40 p-2.5 text-[10px] text-meruem-muted space-y-1">
        <p className="font-semibold text-meruem-text text-[11px] mb-1">How the ML Scanner works</p>
        <p>
          <span className="text-meruem-accent">① Feature Engineering</span> — 14 indicators extracted from each OHLCV window: RSI, ATR%, ADX, EMA cross, momentum, vol-ratio, candle geometry, trend direction, GARCH & Markov regime signals.
        </p>
        <p>
          <span className="text-meruem-accent">② Walk-Forward Training</span> — Multiple time windows extracted from accepted BacktestRun symbols, weighted by recency (60-day exponential decay). 70% train / 30% OOS split.
        </p>
        <p>
          <span className="text-meruem-accent">③ Labels</span> — BUY / SELL / NO_SIGNAL derived from the direction embedded in each accepted strategy's BacktestRun spec.
        </p>
        <p>
          <span className="text-meruem-accent">④ Fallback</span> — When no model is trained or inference fails, Orchestrator transparently falls back to the Ollama LLM scanner. Zero downtime.
        </p>
        <p>
          <span className="text-meruem-accent">⑤ Retraining</span> — Runs automatically every 7 days. Trigger manually with <em>Train Model</em> after adding new approved strategies.
        </p>
      </div>

      {info && (
        <div className="mt-2 rounded border border-emerald-500/30 bg-emerald-500/10 p-2 text-xs text-emerald-300">
          {info}
        </div>
      )}
      {error && (
        <div className="mt-2 rounded border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Training data — extracted Parquet datasets power the walk-forward features */}
      <div className="mt-3 rounded-md border border-meruem-border/60">
        <button
          onClick={() => setShowDatasets((v) => !v)}
          className="flex w-full items-center gap-2 rounded-md bg-meruem-bg/60 px-3 py-2 text-left transition-colors hover:bg-meruem-accent/5"
        >
          {showDatasets ? (
            <ChevronDown className="h-4 w-4 text-meruem-accent" />
          ) : (
            <ChevronRight className="h-4 w-4 text-meruem-accent" />
          )}
          <Database className="h-4 w-4 text-meruem-accent" />
          <span className="text-xs font-semibold">Training Data — Parquet Datasets</span>
          <span className="ml-auto text-[10px] text-meruem-muted">
            extract &amp; manage the historical data used by training features
          </span>
        </button>
        {showDatasets && (
          <div className="border-t border-meruem-border/60 p-2">
            <DatasetsPanel />
          </div>
        )}
      </div>
    </Card>
  );
}
