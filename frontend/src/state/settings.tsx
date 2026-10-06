import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { ChatMode } from "../api/types";
import { sound } from "../audio/sound";
import { speech } from "../audio/speech";

export type ThemeName = "reactor" | "mark3";

export interface Settings {
  theme: ThemeName;
  /** Turns off the decorative animation (scanlines, glitch, rotating rings). */
  reducedMotion: boolean;
  /** Use simulated data instead of the backend. Always shown in the interface. */
  demo: boolean;
  /** Interface sounds. Synthesised in the browser; browsers stay silent until the first click. */
  sound: boolean;
  /** What the chat does with a message: the model answers alone, or from the notes with sources. */
  chatMode: ChatMode;
  /** The assistant reads its replies aloud, with a voice installed on this computer. */
  voice: boolean;
  /** The installed voice to use. Empty: the application picks one. */
  voiceName: string;
  /** Speaking speed, 1 is normal. */
  voiceRate: number;
  /** The assistant greets you aloud when the application opens. */
  greeting: boolean;
  /** After each spoken reply the microphone opens again, so a conversation needs no clicks. */
  handsFree: boolean;
  /** Keep conversations on this computer, so they can be reopened later. */
  saveChats: boolean;
}

const KEY = "reyleight.settings";
export const MIN_VOICE_RATE = 0.7;
export const MAX_VOICE_RATE = 1.4;

function systemPrefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function load(): Settings {
  const defaults: Settings = {
    theme: "reactor",
    reducedMotion: systemPrefersReducedMotion(),
    demo: false,
    sound: true,
    chatMode: "general",
    voice: true,
    voiceName: "",
    voiceRate: 1,
    greeting: true,
    handsFree: false,
    saveChats: true,
  };
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaults;
    const saved = JSON.parse(raw) as Partial<Settings>;
    const rate = typeof saved.voiceRate === "number" && Number.isFinite(saved.voiceRate) ? saved.voiceRate : 1;
    return {
      theme: saved.theme === "mark3" ? "mark3" : "reactor",
      reducedMotion: typeof saved.reducedMotion === "boolean" ? saved.reducedMotion : defaults.reducedMotion,
      demo: saved.demo === true,
      sound: saved.sound !== false,
      chatMode: saved.chatMode === "notes" ? "notes" : "general",
      voice: saved.voice !== false,
      voiceName: typeof saved.voiceName === "string" ? saved.voiceName : "",
      voiceRate: Math.min(MAX_VOICE_RATE, Math.max(MIN_VOICE_RATE, rate)),
      greeting: saved.greeting !== false,
      handsFree: saved.handsFree === true,
      saveChats: saved.saveChats !== false,
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
    sound.setEnabled(settings.sound);
    speech.configure({ enabled: settings.voice, voiceName: settings.voiceName, rate: settings.voiceRate });
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
