import { useEffect, useState } from "react";
import type { Link } from "../state/useHealth";
import { ArcReactor } from "./ArcReactor";

interface Props {
  link: Link;
  onEnter: () => void;
  onDemo: () => void;
  onRetry: () => void;
  demo: boolean;
}

const LINES = [
  "REYLEIGHT CORE // LOCAL-FIRST KNOWLEDGE ASSISTANT",
  "PRIVACY POLICY ........ LOOPBACK ONLY, NO CLOUD CALLS",
  "ALLOW-LIST ............ ONLY FOLDERS YOU CHOSE",
  "RETRIEVED TEXT ........ TREATED AS DATA, NEVER ORDERS",
];

/** The start-up gate. It reports the real backend state and never pretends a link exists. */
export function BootSequence({ link, onEnter, onDemo, onRetry, demo }: Props) {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    if (shown >= LINES.length) return;
    const timer = setTimeout(() => setShown((n) => n + 1), 320);
    return () => clearTimeout(timer);
  }, [shown]);

  const ready = shown >= LINES.length && link.state !== "checking";
  useEffect(() => {
    if (ready && (link.state === "online" || demo)) {
      const timer = setTimeout(onEnter, 700);
      return () => clearTimeout(timer);
    }
  }, [ready, link.state, demo, onEnter]);

  const backendLine =
    link.state === "checking"
      ? "BACKEND LINK ........ SCANNING..."
      : link.state === "online"
        ? `BACKEND LINK ........ ONLINE (${link.latencyMs} ms, v${link.health.version}${demo ? ", DEMO DATA" : ""})`
        : "BACKEND LINK ........ NOT FOUND";

  return (
    <div className="boot" role="status">
      <ArcReactor size={180} state={link.state === "offline" ? "offline" : "thinking"} />
      <h1 className="boot-title glitch" data-text="REYLEIGHT">
        REYLEIGHT
      </h1>
      <div className="boot-log">
        {LINES.slice(0, shown).map((line) => (
          <p key={line}>
            <span className="ok">&gt;</span> {line}
          </p>
        ))}
        {shown >= LINES.length && (
          <p className={link.state === "offline" ? "bad" : undefined}>
            <span className="ok">&gt;</span> {backendLine}
          </p>
        )}
      </div>

      {ready && link.state === "offline" && !demo && (
        <div className="boot-actions">
          <p className="muted">
            Start the backend with <code>python -m uvicorn app.main:app</code> in <code>backend/</code>, or look
            around with simulated data.
          </p>
          <button className="btn" onClick={onRetry}>
            RETRY LINK
          </button>
          <button className="btn btn--alt" onClick={onDemo}>
            ENTER DEMO MODE
          </button>
        </div>
      )}
      <button className="boot-skip" onClick={onEnter}>
        skip
      </button>
    </div>
  );
}
