import { useEffect, useState } from "react";
import type { Api, Health } from "../api/types";

export type Link =
  | { state: "checking" }
  | { state: "online"; health: Health; latencyMs: number }
  | { state: "offline" };

/** Polls /health so the interface always shows whether the backend is really there. */
export function useHealth(api: Api, intervalMs = 10_000): { link: Link; recheck: () => void } {
  const [link, setLink] = useState<Link>({ state: "checking" });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const started = performance.now();
      try {
        const health = await api.health();
        if (!cancelled) setLink({ state: "online", health, latencyMs: Math.round(performance.now() - started) });
      } catch {
        if (!cancelled) setLink({ state: "offline" });
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

  return { link, recheck: () => setTick((t) => t + 1) };
}
