import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type ThemeName = "reactor" | "mark3";

export interface Settings {
  theme: ThemeName;
  /** Turns off the decorative animation (scanlines, glitch, rotating rings). */
  reducedMotion: boolean;
  /** Use simulated data instead of the backend. Always shown in the interface. */
  demo: boolean;
}

const KEY = "reyleight.settings";

function systemPrefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function load(): Settings {
  const defaults: Settings = { theme: "reactor", reducedMotion: systemPrefersReducedMotion(), demo: false };
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaults;
    const saved = JSON.parse(raw) as Partial<Settings>;
    return {
      theme: saved.theme === "mark3" ? "mark3" : "reactor",
      reducedMotion: typeof saved.reducedMotion === "boolean" ? saved.reducedMotion : defaults.reducedMotion,
      demo: saved.demo === true,
    };
  } catch {
    return defaults;
  }
}

interface SettingsContextValue {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
    document.documentElement.dataset.motion = settings.reducedMotion ? "reduced" : "full";
    try {
      localStorage.setItem(KEY, JSON.stringify(settings));
    } catch {
      // storage can be blocked; the settings then last for this session only
    }
  }, [settings]);

  const update = useCallback((patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch })), []);
  const value = useMemo(() => ({ settings, update }), [settings, update]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}
