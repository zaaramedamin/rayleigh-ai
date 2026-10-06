import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { sound, useSoundStatus } from "../audio/sound";
import type { Link } from "../state/useHealth";
import { ArcReactor } from "./ArcReactor";

interface Props {
  link: Link;
  demo: boolean;
  /** True once the app is opening: the reactor flies to the system panel and the text fades. */
  leaving: boolean;
  onEnter: () => void;
  onDemo: () => void;
  onRetry: () => void;
  /** The reactor has reached its place in the system panel. */
  onLanded: () => void;
}

const LINES = [
  "REYLEIGHT CORE // LOCAL-FIRST KNOWLEDGE ASSISTANT",
  "PRIVACY POLICY ........ LOOPBACK ONLY, NO CLOUD CALLS",
  "ALLOW-LIST ............ ONLY FOLDERS YOU CHOSE",
  "RETRIEVED TEXT ........ TREATED AS DATA, NEVER ORDERS",
];

const LINE_DELAY_MS = 800;
const FLIGHT_MS = 1300;
const FLIGHT_EASING = "cubic-bezier(0.65, 0, 0.2, 1)";

/**
 * The start-up gate. It reports the real backend state and never pretends a link exists.
 * The reactor spins at a steady pace, the notes fade in one by one, and when the app opens the
 * very same reactor travels to the system panel on the right.
 */
export function BootSequence({ link, demo, leaving, onEnter, onDemo, onRetry, onLanded }: Props) {
  const [shown, setShown] = useState(0);
  const [settled, setSettled] = useState(false);
  const reactorRef = useRef<HTMLDivElement>(null);
  const soundStatus = useSoundStatus();
  const humPlayed = useRef(false);

  // The start-up hum plays as soon as the browser allows sound, once.
  useEffect(() => {
    if (leaving || humPlayed.current || soundStatus !== "ready") return;
    humPlayed.current = sound.play("boot");
  }, [soundStatus, leaving]);

  useEffect(() => {
    if (shown > 0 && !leaving) sound.play("scan");
  }, [shown, leaving]);

  useEffect(() => {
    if (leaving || shown >= LINES.length) return;
    const timer = setTimeout(() => setShown((n) => n + 1), LINE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [shown, leaving]);

  const ready = shown >= LINES.length && link.state !== "checking";
  useEffect(() => {
    if (leaving || !ready || !(link.state === "online" || demo)) return;
    const timer = setTimeout(onEnter, 900);
    return () => clearTimeout(timer);
  }, [ready, link.state, demo, leaving, onEnter]);

  // Fly the reactor from the middle of the screen to its slot in the system panel.
  useLayoutEffect(() => {
    if (!leaving) return;
    const reactor = reactorRef.current;
    const slot = document.querySelector<HTMLElement>("[data-reactor-target]");
    const to = slot?.getBoundingClientRect();
    const finish = () => {
      setSettled(true);
      onLanded();
    };
    if (!reactor) {
      onLanded();
      return;
    }
    if (!to || to.width === 0) {
      // Narrow screens have no system panel: let the reactor fade away instead.
      const fade = reactor.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 600, fill: "forwards" });
      fade.finished.then(finish, () => undefined);
      return () => fade.cancel();
    }
    const from = reactor.getBoundingClientRect();
    const flight = reactor.animate(
      [
        { transformOrigin: "0 0", transform: "none" },
        {
          transformOrigin: "0 0",
          transform: `translate(${to.left - from.left}px, ${to.top - from.top}px) scale(${to.width / from.width})`,
        },
      ],
      { duration: FLIGHT_MS, easing: FLIGHT_EASING, fill: "forwards" },
    );
    flight.finished.then(finish, () => undefined);
    return () => flight.cancel();
    // onLanded is stable (useCallback in App); the flight must run exactly once per departure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [leaving]);

  const backendLine =
    link.state === "checking"
      ? "BACKEND LINK ........ SCANNING..."
      : link.state === "online"
        ? `BACKEND LINK ........ ONLINE (${link.latencyMs} ms, v${link.health.version}${demo ? ", DEMO DATA" : ""})`
        : "BACKEND LINK ........ NOT FOUND";

  return (
    <div className={`boot${leaving ? " boot--leaving" : ""}`} role="status">
      <div ref={reactorRef} className={`boot-reactor${settled ? " boot-reactor--settled" : ""}`}>
        <ArcReactor size={180} state={link.state === "offline" ? "offline" : "boot"} />
      </div>
      <div className="boot-fade">
        <h1 className="boot-title glitch" data-text="REYLEIGHT">
          REYLEIGHT
        </h1>
        <div className="boot-log">
          {LINES.slice(0, shown).map((line) => (
            <p key={line} className="boot-line">
              <span className="ok">&gt;</span> {line}
            </p>
          ))}
          {shown >= LINES.length && (
            <p className={`boot-line${link.state === "offline" ? " bad" : ""}`}>
              <span className="ok">&gt;</span> {backendLine}
            </p>
          )}
        </div>

        {ready && link.state === "offline" && !demo && (
          <div className="boot-actions">
            <p className="muted">
              Start the backend with <code>python -m app serve</code>, or look around with simulated data.
            </p>
            <button className="btn" onClick={onRetry}>
              RETRY LINK
            </button>
            <button className="btn btn--alt" onClick={onDemo}>
              ENTER DEMO MODE
            </button>
          </div>
        )}
      </div>
      {!leaving && soundStatus === "locked" && <p className="boot-sound">Click anywhere to enable sound</p>}
      {!leaving && (
        <button className="boot-skip" onClick={onEnter}>
          skip
        </button>
      )}
    </div>
  );
}
