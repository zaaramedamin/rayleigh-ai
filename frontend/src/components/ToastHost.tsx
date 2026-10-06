import { useEffect, useRef, useState } from "react";
import { useEvents } from "../state/events";
import type { AppEvent, Level } from "../state/events";
import { Icon } from "./Icon";
import type { IconName } from "./Icon";

const LIFETIME_MS: Record<Level, number> = {
  info: 4500,
  success: 4500,
  warning: 7000,
  error: 9000,
  alert: 12000,
};
const ICON: Record<Level, IconName> = {
  info: "info",
  success: "check",
  warning: "alert",
  error: "alert",
  alert: "alert",
};
const MAX_VISIBLE = 4;

/** Pop-up notices in the top-right corner. Click one to dismiss it. */
export function ToastHost() {
  const all = useEvents();
  const [visible, setVisible] = useState<AppEvent[]>([]);
  const seen = useRef(all[0]?.id ?? 0);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  const dismiss = (id: number) => {
    const timer = timers.current.get(id);
    if (timer) clearTimeout(timer);
    timers.current.delete(id);
    setVisible((v) => v.filter((e) => e.id !== id));
  };

  useEffect(() => {
    const fresh = all.filter((e) => e.id > seen.current && e.toast);
    if (all.length) seen.current = all[0].id;
    if (!fresh.length) return;
    setVisible((v) => [...fresh, ...v].slice(0, MAX_VISIBLE));
    for (const event of fresh) {
      timers.current.set(
        event.id,
        setTimeout(() => dismiss(event.id), LIFETIME_MS[event.level]),
      );
    }
    // dismiss only touches refs and state setters
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all]);

  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach((timer) => clearTimeout(timer));
  }, []);

  return (
    <div className="toasts" aria-live="assertive">
      {visible.map((event) => (
        <button
          key={event.id}
          className={`toast toast--${event.level}`}
          role={event.level === "alert" || event.level === "error" ? "alert" : "status"}
          onClick={() => dismiss(event.id)}
          style={{ ["--life" as string]: `${LIFETIME_MS[event.level]}ms` }}
        >
          <Icon name={ICON[event.level]} size={20} />
          <span className="toast-text">
            <b>{event.title}</b>
            {event.detail && <small>{event.detail}</small>}
          </span>
          <i className="toast-timer" aria-hidden />
        </button>
      ))}
    </div>
  );
}
