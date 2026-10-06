import { useEffect, useState } from "react";

/** The local time and date, for the top bar. */
export function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  return (
    <div className="clock" aria-label="Local time">
      <b>{now.toLocaleTimeString([], { hour12: false })}</b>
      <small>{now.toLocaleDateString([], { weekday: "short", day: "2-digit", month: "short" })}</small>
    </div>
  );
}
