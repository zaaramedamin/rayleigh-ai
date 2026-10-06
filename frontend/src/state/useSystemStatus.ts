import { useEffect, useState } from "react";
import type { Api, SystemStatus } from "../api/types";

export type StatusState =
  | { state: "loading" }
  | { state: "ready"; status: SystemStatus }
  | { state: "unavailable" };

/** Loads /system/status while the backend is reachable. It checks the model server, so poll slowly. */
export function useSystemStatus(
  api: Api,
  online: boolean,
  /** Change this number to load the status again now (after the library changed). */
  refreshKey = 0,
  intervalMs = 30_000,
): StatusState {
  const [result, setResult] = useState<StatusState>({ state: "loading" });

  useEffect(() => {
    if (!online) {
      setResult({ state: "unavailable" });
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const status = await api.systemStatus();
        if (!cancelled) setResult({ state: "ready", status });
      } catch {
        if (!cancelled) setResult({ state: "unavailable" });
      }
    };
    void load();
    const timer = setInterval(load, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [api, online, refreshKey, intervalMs]);

  return result;
}
