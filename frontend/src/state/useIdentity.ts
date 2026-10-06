import { useCallback, useEffect, useState } from "react";
import type { Api, AssistantIdentity } from "../api/types";

const DEFAULT: AssistantIdentity = {
  name: "Reyleight",
  address: "sir",
  role: "",
  use_profile: true,
  use_memory: true,
};

/**
 * Who the assistant is (its name, how it addresses you), read once the application is open and
 * again whenever `reload` is called. Until it is known, the defaults are used.
 */
export function useIdentity(api: Api, enabled: boolean) {
  const [identity, setIdentity] = useState<AssistantIdentity>(DEFAULT);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    api.assistant().then(
      (next) => !cancelled && setIdentity(next),
      () => undefined, // an older backend has no such page: keep the defaults
    );
    return () => {
      cancelled = true;
    };
  }, [api, enabled, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { identity, reload };
}
