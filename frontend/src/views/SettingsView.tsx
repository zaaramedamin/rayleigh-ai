import { useState } from "react";
import type { FormEvent } from "react";
import type { Api } from "../api/types";
import { speech, useVoices } from "../audio/speech";
import { AgentPanel } from "../components/AgentPanel";
import { HudFrame } from "../components/HudFrame";
import { events } from "../state/events";
import type { Level } from "../state/events";
import { MAX_VOICE_RATE, MIN_VOICE_RATE, useSettings } from "../state/settings";
import type { ThemeName } from "../state/settings";

const THEMES: Array<{ id: ThemeName; name: string; note: string }> = [
  { id: "reactor", name: "ARC REACTOR", note: "Cyan on deep blue" },
  { id: "mark3", name: "MARK III", note: "Gold and red" },
];

interface Props {
  api: Api;
  /** A new sign-in token after the password was changed (the old sessions are ended). */
  onPasswordChanged: (token: string) => void;
}

function PasswordForm({ api, onPasswordChanged }: Props) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setMessage("");
    if (next.length < 8) return setError("Use at least 8 characters.");
    if (next !== again) return setError("The two new passwords do not match.");
    setBusy(true);
    try {
      onPasswordChanged(await api.changePassword(current, next));
      setCurrent("");
      setNext("");
      setAgain("");
      setMessage("Password changed. Other sessions were signed out.");
      events.notify("success", "PASSWORD CHANGED", { detail: "Other sessions were signed out." });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not change the password.";
      setError(message);
      events.notify("error", "PASSWORD NOT CHANGED", { detail: message, sound: "denied" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="profile-form" onSubmit={submit}>
      <label className="profile-field">
        <span>Current password</span>
        <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
      </label>
      <label className="profile-field">
        <span>New password</span>
        <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
      </label>
      <label className="profile-field">
        <span>Repeat new password</span>
        <input type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} />
      </label>
      {error && (
        <p className="bad" role="alert">
          {error}
        </p>
      )}
      {message && <p className="ok">{message}</p>}
      <div className="profile-actions">
        <button className="btn" disabled={busy || !current || !next}>
          CHANGE PASSWORD
        </button>
      </div>
    </form>
  );
}

const PREVIEWS: Array<{ label: string; level: Level; title: string; detail: string; sound?: "denied" }> = [
  { label: "INFO", level: "info", title: "TEST NOTICE", detail: "A quiet message. No sound." },
  { label: "SUCCESS", level: "success", title: "TEST SUCCESS", detail: "Something worked." },
  { label: "WARNING", level: "warning", title: "TEST WARNING", detail: "Something needs attention." },
  { label: "ERROR", level: "error", title: "TEST ERROR", detail: "Something failed." },
  { label: "DENIED", level: "error", title: "TEST ACCESS DENIED", detail: "Wrong password.", sound: "denied" },
  { label: "ALERT", level: "alert", title: "TEST ALERT", detail: "A system-level problem." },
];

export function SettingsView({ api, onPasswordChanged }: Props) {
  const { settings, update } = useSettings();
  const voices = useVoices();
  return (
    <div className="stack">
      <HudFrame title="THEME" tag="LOOK">
        <div className="grid">
          {THEMES.map((t) => (
            <button
              key={t.id}
              className={`module module--pick${settings.theme === t.id ? " module--on" : ""}`}
              onClick={() => update({ theme: t.id })}
              aria-pressed={settings.theme === t.id}
            >
              <header>
                <h3>{t.name}</h3>
                <span className={`swatch swatch--${t.id}`} aria-hidden />
              </header>
              <p>{t.note}</p>
            </button>
          ))}
        </div>
      </HudFrame>
      <HudFrame title="BEHAVIOUR" tag="PREFERENCES">
        <label className="toggle">
          <input
            type="checkbox"
            checked={settings.reducedMotion}
            onChange={(e) => update({ reducedMotion: e.target.checked })}
          />
          <span>Reduce motion (stops scanlines, glitch and spinning rings)</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.sound} onChange={(e) => update({ sound: e.target.checked })} />
          <span>Interface sounds (made on this device; they start after you unlock)</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.demo} onChange={(e) => update({ demo: e.target.checked })} />
          <span>Demo mode (simulated notes instead of the backend)</span>
        </label>
      </HudFrame>
      <HudFrame title="VOICE" tag="SPEAK AND LISTEN">
        <label className="toggle">
          <input type="checkbox" checked={settings.voice} onChange={(e) => update({ voice: e.target.checked })} />
          <span>Read the assistant's replies aloud</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.greeting} onChange={(e) => update({ greeting: e.target.checked })} />
          <span>Greet me aloud when the application opens</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.handsFree} onChange={(e) => update({ handsFree: e.target.checked })} />
          <span>Hands-free: listen again after each spoken reply (Esc stops it)</span>
        </label>
        {voices.length === 0 ? (
          <p className="warn">
            {speech.supported()
              ? "No voice is installed on this computer, so the assistant cannot speak. Add one in Windows Settings > Time & language > Speech."
              : "This browser cannot speak."}
          </p>
        ) : (
          <div className="voice-row">
            <select
              className="voice-select"
              aria-label="Voice"
              value={settings.voiceName}
              onChange={(e) => update({ voiceName: e.target.value })}
            >
              <option value="">Automatic ({speech.currentVoice() ?? "none"})</option>
              {voices.map((v) => (
                <option key={v.name} value={v.name}>
                  {v.name} ({v.lang})
                </option>
              ))}
            </select>
            <label className="toggle toggle--small">
              <span>Speed</span>
              <input
                type="range"
                min={MIN_VOICE_RATE}
                max={MAX_VOICE_RATE}
                step={0.05}
                value={settings.voiceRate}
                onChange={(e) => update({ voiceRate: Number(e.target.value) })}
              />
            </label>
            <button
              className="btn btn--ghost"
              onClick={() => void speech.speak("Good evening. All systems are online. How can I serve you?", true)}
            >
              TEST VOICE
            </button>
          </div>
        )}
        <p className="muted">
          Only voices installed on this computer are used. Voices that work through an online service are never
          chosen, because they would send what the assistant says to that service. Your own voice is turned into text
          by a speech model on this computer (download it once with <code>python -m app download-voice-model</code>).
        </p>
      </HudFrame>
      <HudFrame title="AGENT" tag="ACTS ON YOUR COMPUTER">
        <AgentPanel api={api} />
      </HudFrame>
      <HudFrame title="ALERTS AND SOUNDS" tag="PREVIEW">
        <p className="muted">Try each kind of notice. Sound needs interface sounds switched on above.</p>
        <div className="chips">
          {PREVIEWS.map((p) => (
            <button
              key={p.label}
              className={`chip chip--${p.level}`}
              onClick={() => events.notify(p.level, p.title, { detail: p.detail, sound: p.sound })}
            >
              {p.label}
            </button>
          ))}
        </div>
      </HudFrame>
      {!settings.demo && (
        <HudFrame title="ACCESS PASSWORD" tag="SECURITY">
          <PasswordForm api={api} onPasswordChanged={onPasswordChanged} />
        </HudFrame>
      )}
    </div>
  );
}
