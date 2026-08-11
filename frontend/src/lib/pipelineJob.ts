import { api, StrategyPipelineJob } from "@/lib/api";

/**
 * Module-scoped strategy pipeline job poller.
 *
 * The ``StrategyLibraryPanel`` unmounts whenever the user switches tabs, which
 * destroyed the old component-local ``setInterval`` and made in-flight
 * synthesis look "canceled". The backend independently continues the job on a
 * background thread, so this store keeps polling at module scope — independent
 * of any component lifecycle — and replays the latest snapshot to whichever
 * panel subscribes (e.g. the panel re-mounted on returning to the tab).
 */

type Listener = (job: StrategyPipelineJob | null) => void;

const POLL_MS = 1500;

let jobId: string | null = null;
let snapshot: StrategyPipelineJob | null = null;
let timer: number | null = null;
const listeners = new Set<Listener>();
let finalizeHandler: ((job: StrategyPipelineJob) => void) | null = null;

function emit() {
  for (const fn of listeners) fn(snapshot);
}

function clearTimer() {
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
}

function isFinal(s: StrategyPipelineJob): boolean {
  return s.status === "done" || s.status === "error";
}

async function poll() {
  if (!jobId) return;
  try {
    const s = await api.get<StrategyPipelineJob>(`/strategies/submit/${jobId}`);
    snapshot = s;
    emit();
    if (isFinal(s)) {
      // Keep the final snapshot so a returning tab can render the result, but
      // stop hammering the backend.
      clearTimer();
      if (finalizeHandler) finalizeHandler(s);
    }
  } catch {
    // Transient network blip — keep polling so an in-flight job is not lost.
  }
}

/** Begin polling a job. Any previous job is abandoned (its snapshot reset). */
export function startPoll(newJobId: string) {
  clearTimer();
  jobId = newJobId;
  snapshot = null;
  emit();
  timer = window.setInterval(poll, POLL_MS);
}

/** Abandon the active job (Stop button). */
export function stopPoll() {
  clearTimer();
  jobId = null;
  snapshot = null;
  emit();
}

/** Latest known job snapshot (persists after the poll stops finalizing). */
export function getActiveJob(): StrategyPipelineJob | null {
  return snapshot;
}

/** Subscribe to job snapshots; fires immediately with the current state. */
export function subscribeJob(fn: Listener): () => void {
  listeners.add(fn);
  fn(snapshot);
  return () => {
    listeners.delete(fn);
  };
}

/** Register a live-finalization callback (fires on the poll tick that ends a job). */
export function setFinalizeHandler(fn: ((job: StrategyPipelineJob) => void) | null) {
  finalizeHandler = fn;
}