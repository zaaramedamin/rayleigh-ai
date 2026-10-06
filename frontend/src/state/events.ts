import { useSyncExternalStore } from "react";
import { sound } from "../audio/sound";
import type { SoundName } from "../audio/synth";

/**
 * Everything noteworthy that happens while the app runs: successes, warnings, errors and
 * system alerts. Each one plays its sound, shows a toast and lands in the activity feed.
 * Events hold short messages only, never note text or questions.
 */
export type Level = "info" | "success" | "warning" | "error" | "alert";

export interface AppEvent {
  id: number;
  level: Level;
  title: string;
  detail?: string;
  at: number;
  /** Show a toast (default). Quiet events still reach the activity feed. */
  toast: boolean;
}

interface NotifyOptions {
  detail?: string;
  /** Override the sound for this level; null for silence. */
  sound?: SoundName | null;
  toast?: boolean;
  /** Identical keys within a few seconds collapse into one event (no spam). */
  key?: string;
}

const SOUND_FOR_LEVEL: Record<Level, SoundName | null> = {
  info: null,
  success: "success",
  warning: "warning",
  error: "error",
  alert: "alert",
};

const MAX_EVENTS = 50;
const DEDUPE_MS = 8000;

let list: AppEvent[] = [];
let nextId = 1;
const listeners = new Set<() => void>();
const recent = new Map<string, number>();

export const events = {
  notify(level: Level, title: string, options: NotifyOptions = {}): AppEvent | null {
    const now = Date.now();
    if (options.key) {
      const last = recent.get(options.key);
      if (last !== undefined && now - last < DEDUPE_MS) return null;
      recent.set(options.key, now);
    }
    const event: AppEvent = {
      id: nextId++,
      level,
      title,
      detail: options.detail,
      at: now,
      toast: options.toast ?? true,
    };
    list = [event, ...list].slice(0, MAX_EVENTS);
    const name = options.sound === undefined ? SOUND_FOR_LEVEL[level] : options.sound;
    if (name) sound.play(name);
    listeners.forEach((listener) => listener());
    return event;
  },

  /** Forget a dedupe key, so the next event with it is shown even if it is very recent. */
  forget(key: string): void {
    recent.delete(key);
  },

  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  snapshot(): AppEvent[] {
    return list;
  },
};

export function useEvents(): AppEvent[] {
  return useSyncExternalStore(events.subscribe, events.snapshot, events.snapshot);
}
