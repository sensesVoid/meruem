import { useEffect } from "react";

export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function fmt(n: number, digits = 2): string {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
}

/**
 * Visibility-aware polling: runs `fn` immediately, then on `intervalMs`
 * — but only while the document tab is VISIBLE. Polls pause in the
 * background (no wasted requests / CPU), and resume + refresh instantly
 * when the user returns to the tab.
 */
export function usePolling(fn: () => void, intervalMs: number) {
  useEffect(() => {
    fn();
    const id = window.setInterval(() => {
      if (document.hidden) return; // skip background ticks
      fn();
    }, intervalMs);
    const onVisible = () => {
      if (!document.hidden) fn(); // catch up when the user comes back
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fn, intervalMs]);
}