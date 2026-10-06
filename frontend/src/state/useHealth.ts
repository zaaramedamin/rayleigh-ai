import { useEffect, useState } from "react";
import type { Api, Health } from "../api/types";

export type Link =
  | { state: "checking" }
  | { state: "online"; health: Health; latencyMs: number }
  | { state: "offline" };

/** Polls /health so the interface always shows whether the backend is really there. */
const HISTORY_LENGTH = 30;

export function useHealth(
  api: Api,
  intervalMs = 10_000,
): { link: Link; recheck: () => void; history: Array<number | null> } {
  const [link, setLink] = useState<Link>({ state: "checking" });
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState<Array<number | null>>([]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const started = performance.now();
      try {
        const health = await api.health();
        const latencyMs = Math.round(performance.now() - started);
        if (cancelled) return;
        setLink({ state: "online", health, latencyMs });
        setHistory((h) => [...h, latencyMs].slice(-HISTORY_LENGTH));
      } catch {
        if (cancelled) return;
        setLink({ state: "offline" });
        setHistory((h) => [...h, null].slice(-HISTORY_LENGTH));
      }
    };
    setLink({ state: "checking" });
    void check();
    const timer = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [api, intervalMs, tick]);

  return { link, recheck: () => setTick((t) => t + 1), history };
}
