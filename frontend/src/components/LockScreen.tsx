import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { ApiError } from "../api/client";
import type { Api } from "../api/types";
import { sound } from "../audio/sound";
import { events } from "../state/events";
import { ArcReactor } from "./ArcReactor";

interface Props {
  api: Api;
  /** Signed in. `token` is null when the server does not ask for a password at all. */
  onUnlocked: (token: string | null) => void;
  onDemo: () => void;
}

type Mode = "checking" | "login" | "setup" | "offline";

const MIN_LENGTH = 8;

/**
 * The door. Everything else in the application sits behind it. The button you press here is
 * also the click browsers need before they allow sound, so the start-up sequence that follows
 * can play with sound.
 */
export function LockScreen({ api, onUnlocked, onDemo }: Props) {
  const [mode, setMode] = useState<Mode>("checking");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [shake, setShake] = useState(false);

  const check = useCallback(async () => {
    setMode("checking");
    try {
      const status = await api.authStatus();
      if (!status.required) onUnlocked(null);
      else setMode(status.configured ? "login" : "setup");
    } catch {
      setMode("offline");
    }
  }, [api, onUnlocked]);

  useEffect(() => {
    void check();
  }, [check]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    // This runs inside the click: the one moment the browser lets sound switch on.
    sound.unlock();
    setError("");
    if (mode === "setup") {
      if (password.length < MIN_LENGTH) return setError(`Use at least ${MIN_LENGTH} characters.`);
      if (password !== confirm) return setError("The two passwords do not match.");
    }
    setBusy(true);
    try {
      const token = mode === "setup" ? await api.setupPassword(password) : await api.login(password);
      setPassword("");
      setConfirm("");
      onUnlocked(token);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not sign in.";
      setError(message);
      if (err instanceof ApiError && err.kind === "offline") {
        setMode("offline");
        events.notify("alert", "BACKEND NOT REACHABLE", { detail: message, key: "link" });
      } else if (err instanceof ApiError && err.kind === "throttled") {
        events.notify("warning", "TOO MANY ATTEMPTS", { detail: message });
      } else {
        events.notify("error", "ACCESS DENIED", { detail: message, sound: "denied" });
      }
      setShake(true);
      setTimeout(() => setShake(false), 650);
    } finally {
      setBusy(false);
    }
  };

  const setup = mode === "setup";
  return (
    <div className="lock" role="main">
      <div className={`lock-card${shake ? " lock-card--denied" : ""}`}>
        <ArcReactor size={110} state={mode === "offline" ? "offline" : "boot"} />
        <h1 className="lock-title">REYLEIGHT</h1>

        {mode === "checking" && <p className="muted">Checking the backend...</p>}

        {mode === "offline" && (
          <>
            <p className="bad">The backend is not reachable.</p>
            <p className="muted">
              Start it with <code>python -m app serve</code>, or look around with simulated data.
            </p>
            <div className="lock-actions">
              <button className="btn" onClick={() => void check()}>
                RETRY
              </button>
              <button className="btn btn--alt" onClick={onDemo}>
                DEMO MODE
              </button>
            </div>
          </>
        )}

        {(mode === "login" || mode === "setup") && (
          <form className="lock-form" onSubmit={submit}>
            <p className="lock-heading">{setup ? "CHOOSE YOUR ACCESS PASSWORD" : "ACCESS REQUIRED"}</p>
            {setup && (
              <p className="muted lock-note">
                This password opens Reyleight on this computer. It is stored only as a salted hash. If you forget it,
                delete <code>data/access.json</code> to choose a new one; your notes are not affected.
              </p>
            )}
            <label>
              <span>{setup ? "New password" : "Password"}</span>
              <input
                type="password"
                autoFocus
                autoComplete={setup ? "new-password" : "current-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            {setup && (
              <label>
                <span>Repeat password</span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                />
              </label>
            )}
            {error && (
              <p className="bad lock-error" role="alert">
                {error}
              </p>
            )}
            <button className="btn lock-submit" disabled={busy || !password}>
              {busy ? "CHECKING..." : setup ? "CREATE AND ENTER" : "UNLOCK"}
            </button>
          </form>
        )}

        {mode !== "offline" && (
          <button className="link lock-demo" onClick={onDemo}>
            explore demo mode instead
          </button>
        )}
      </div>
    </div>
  );
}
