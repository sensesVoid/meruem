import { useCallback, useEffect, useState } from "react";
import {
  api,
  Asset,
  deleteMLArtifact,
  getMLArtifact,
  listMLArtifacts,
  MLArtifact,
  uploadMLArtifact,
} from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select, Textarea } from "@/components/ui/input";

const ACCEPTED = [".txt", ".md", ".json", ".csv"];

export function MLArtifactUpload() {
  const [artifacts, setArtifacts] = useState<MLArtifact[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [filename, setFilename] = useState("");
  const [symbol, setSymbol] = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState("");
  const [error, setError] = useState("");
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setArtifacts(await listMLArtifacts());
    } catch {
      setError("Could not load ML context files (backend offline?).");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    api.get<Asset[]>("/assets").then((list) => setAssets(list)).catch(() => {});
  }, []);

  const upload = async () => {
    setError("");
    setInfo("");
    if (!filename.trim()) {
      setError("Enter a file name (e.g. model-notes.md).");
      return;
    }
    const ext = "." + filename.split(".").pop()?.toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      setError(`Supported types: ${ACCEPTED.join(", ")}.`);
      return;
    }
    if (!content.trim()) {
      setError("Paste the file content first.");
      return;
    }
    setBusy(true);
    try {
      await uploadMLArtifact({
        filename: filename.trim(),
        symbol: symbol || null,
        content,
      });
      setFilename("");
      setContent("");
      setSymbol("");
      setInfo("ML context file saved — the Scanner reads it as context at scan time.");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    setError("");
    try {
      await deleteMLArtifact(id);
      if (previewId === id) {
        setPreviewId(null);
        setPreview(null);
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const previewFile = async (id: string) => {
    if (previewId === id) {
      setPreviewId(null);
      setPreview(null);
      return;
    }
    setError("");
    try {
      const art = await getMLArtifact(id);
      setPreviewId(id);
      setPreview(art.preview ?? "(no preview)");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Card title="ML context files" className="mt-2">
      <div className="space-y-1.5">
        <p className="text-[11px] text-meruem-muted">
          Upload model / ML context notes (research, model output, indicator studies) as{" "}
          <span className="text-meruem-accent">.txt / .md / .json / .csv</span>. The Scanner
          reads them as context-only at scan time — they sharpen the read but never override the
          approved strategy rules. Optionally scope a file to one symbol.
        </p>
        <div className="flex flex-wrap items-end gap-1.5">
          <Input
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="file-name.md"
            className="w-40"
          />
          <Select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="w-28" title="Optional: scope the file to one symbol">
            <option value="">All symbols</option>
            {assets.map((a) => (
              <option key={a.id} value={a.symbol}>
                {a.symbol}
              </option>
            ))}
          </Select>
        </div>
        <Textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Paste the ML / model context content here…"
          rows={4}
          className="resize-y min-h-[72px]"
        />
        <Button size="sm" onClick={upload} disabled={busy}>
          {busy ? "Saving…" : "Upload"}
        </Button>
        {info && (
          <p className="rounded border border-meruem-green/40 bg-meruem-green/10 p-1.5 text-[11px] text-meruem-green">
            {info}
          </p>
        )}
        {error && (
          <p className="rounded border border-meruem-red/40 bg-meruem-red/10 p-1.5 text-[11px] text-meruem-red">
            {error}
          </p>
        )}
      </div>

      {artifacts.length > 0 && (
        <div className="mt-2 max-h-56 space-y-1 overflow-y-auto">
          {artifacts.map((a) => (
            <div key={a.id} className="rounded border border-meruem-border bg-meruem-bg p-1.5">
              <div className="flex items-center justify-between gap-1.5">
                <span className="min-w-0 truncate text-[11px] text-meruem-text" title={a.filename}>
                  {a.filename}
                </span>
                <div className="flex shrink-0 items-center gap-1">
                  {a.symbol && <Badge tone="blue">{a.symbol}</Badge>}
                  <Button size="sm" variant="outline" onClick={() => previewFile(a.id)}>
                    {previewId === a.id ? "Close" : "Preview"}
                  </Button>
                  <button className="text-meruem-red" onClick={() => remove(a.id)}>x</button>
                </div>
              </div>
              {previewId === a.id && (
                <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded border border-meruem-border bg-meruem-bg p-1.5 text-[10px] text-meruem-text">
                  {preview}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
