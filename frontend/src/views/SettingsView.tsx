import { HudFrame } from "../components/HudFrame";
import { useSettings } from "../state/settings";
import type { ThemeName } from "../state/settings";

const THEMES: Array<{ id: ThemeName; name: string; note: string }> = [
  { id: "reactor", name: "ARC REACTOR", note: "Cyan on deep blue" },
  { id: "mark3", name: "MARK III", note: "Gold and red" },
];

export function SettingsView() {
  const { settings, update } = useSettings();
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
          <input type="checkbox" checked={settings.demo} onChange={(e) => update({ demo: e.target.checked })} />
          <span>Demo mode (simulated notes instead of the backend)</span>
        </label>
      </HudFrame>
    </div>
  );
}
