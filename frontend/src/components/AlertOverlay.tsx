import { useEffect, useRef, useState } from "react";
import { useEvents } from "../state/events";
import type { AppEvent } from "../state/events";
import { Icon } from "./Icon";

const SHOW_MS = 4200;

/**
 * A system-level alert (the backend link was lost, for example) takes over the screen for a
 * moment: the edges flash red, the whole interface shudders once, and a hazard banner names
 * the problem. Smaller problems only get a toast.
 */
export function AlertOverlay() {
  const all = useEvents();
  const latest = all.find((e) => e.level === "alert");
  const seen = useRef(latest?.id ?? 0);
  const [active, setActive] = useState<AppEvent | null>(null);

  useEffect(() => {
    if (!latest || latest.id <= seen.current) return;
    seen.current = latest.id;
    setActive(latest);
    document.documentElement.dataset.alert = "on";
    const timer = setTimeout(() => {
      setActive(null);
      delete document.documentElement.dataset.alert;
    }, SHOW_MS);
    return () => {
      clearTimeout(timer);
      delete document.documentElement.dataset.alert;
    };
  }, [latest]);

  if (!active) return null;
  return (
    <div className="alert-overlay" role="alert" aria-live="assertive">
      <div className="alert-band">
        <Icon name="alert" size={26} />
        <span className="alert-word">ALERT</span>
        <span className="alert-title">{active.title}</span>
      </div>
    </div>
  );
}
