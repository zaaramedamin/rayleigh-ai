import { useSyncExternalStore } from "react";
import { schedule } from "./synth";
import type { SoundName } from "./synth";

// Browsers refuse to make sound until the person has clicked or pressed a key on the page, so
// the audio engine starts "locked" and `unlock()` is called from the first such gesture.

const MASTER_GAIN = 0.35;

let ctx: AudioContext | null = null;
let master: GainNode | null = null;
let enabled = true;
const listeners = new Set<() => void>();

function emit(): void {
  listeners.forEach((listener) => listener());
}

export type SoundStatus = "off" | "locked" | "ready";

function status(): SoundStatus {
  if (!enabled) return "off";
  return ctx?.state === "running" ? "ready" : "locked";
}

export const sound = {
  setEnabled(value: boolean): void {
    enabled = value;
    emit();
  },

  /** Call from a click or key press. Creates the audio engine the first time. */
  unlock(): void {
    try {
      if (!ctx) {
        const Engine = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!Engine) return;
        ctx = new Engine();
        master = ctx.createGain();
        master.gain.value = MASTER_GAIN;
        const limiter = ctx.createDynamicsCompressor();
        master.connect(limiter).connect(ctx.destination);
        ctx.onstatechange = emit;
      }
      if (ctx.state !== "running") void ctx.resume().then(emit, () => undefined);
    } catch {
      // no audio device or blocked: the interface simply stays silent
    }
    emit();
  },

  /** Plays a sound if sound is on and unlocked. Returns whether it was played. */
  play(name: SoundName): boolean {
    if (!enabled || !ctx || !master || ctx.state !== "running") return false;
    try {
      schedule(name, ctx, master, ctx.currentTime + 0.01);
      return true;
    } catch {
      return false;
    }
  },

  status,

  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useSoundStatus(): SoundStatus {
  return useSyncExternalStore(sound.subscribe, sound.status, () => "locked" as const);
}
