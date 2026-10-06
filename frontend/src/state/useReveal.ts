import { useEffect, useState } from "react";

/** Counts up from 0 to `total`, one step every `stepMs`, once `active` turns true. */
export function useReveal(total: number, active: boolean, stepMs = 260): number {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!active || count >= total) return;
    const timer = setTimeout(() => setCount(count + 1), stepMs);
    return () => clearTimeout(timer);
  }, [active, count, total, stepMs]);

  return count;
}
