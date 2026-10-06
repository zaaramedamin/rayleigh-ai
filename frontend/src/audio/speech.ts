import { useSyncExternalStore } from "react";

// The assistant's voice. It uses the browser's speech synthesis, and only voices installed on
// this computer: a voice that works through an online service (Edge's "Online (Natural)" voices,
// for example) would send every reply to that service, so those are never used.

/** The parts of a browser voice this module looks at. */
export interface VoiceLike {
  name: string;
  lang: string;
  localService: boolean;
}

const MAX_CHUNK_CHARS = 220;
/** Voices that sound right for a calm butler, when the choice is left to the application. */
const PREFERRED_NAMES = /george|ryan|daniel|david|mark|james|guy/i;

/** What is read aloud: no citation markers, links, markup or symbols that a voice would spell. */
export function cleanForSpeech(text: string): string {
  return text
    .replace(/\[\d+(?:\s*,\s*\d+)*\]/g, "")
    .replace(/https?:\/\/\S+/g, "a link")
    .replace(/[*_#`>|~]+/g, " ")
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

/** Short pieces, cut at sentence ends: long single utterances are cut off by some browsers. */
export function splitForSpeech(text: string, maxChars = MAX_CHUNK_CHARS): string[] {
  const chunks: string[] = [];
  let current = "";
  const push = () => {
    if (current.trim()) chunks.push(current.trim());
    current = "";
  };
  for (const sentence of text.match(/[^.!?。؟]+[.!?。؟]*\s*/g) ?? []) {
    if (current && current.length + sentence.length > maxChars) push();
    current += sentence;
    while (current.length > maxChars) {
      const cut = current.lastIndexOf(" ", maxChars);
      const at = cut > 0 ? cut : maxChars;
      chunks.push(current.slice(0, at).trim());
      current = current.slice(at);
    }
  }
  push();
  return chunks;
}

/**
 * The voice to speak with: the one asked for if this computer has it, else the best installed
 * English voice, else any installed voice. Null when no voice is installed locally.
 */
export function pickVoice<T extends VoiceLike>(voices: readonly T[], preferred: string): T | null {
  const local = voices.filter((v) => v.localService);
  const asked = local.find((v) => v.name === preferred);
  if (asked) return asked;
  const score = (v: T) => {
    const lang = v.lang.toLowerCase().replace("_", "-");
    return (lang.startsWith("en-gb") ? 4 : lang.startsWith("en") ? 2 : 0) + (PREFERRED_NAMES.test(v.name) ? 1 : 0);
  };
  return [...local].sort((a, b) => score(b) - score(a))[0] ?? null;
}

let enabled = true;
let voiceName = "";
let rate = 1;
let stops = 0; // goes up on every stop(), so anything queued before it is dropped
let pending = 0; // sentences being said or waiting to be said
let tail: Promise<unknown> = Promise.resolve();
let abortCurrent: (() => void) | null = null;
const listeners = new Set<() => void>();

function synth(): SpeechSynthesis | null {
  return typeof window !== "undefined" && "speechSynthesis" in window ? window.speechSynthesis : null;
}

function emit(): void {
  listeners.forEach((listener) => listener());
}

function localVoices(): SpeechSynthesisVoice[] {
  return (synth()?.getVoices() ?? []).filter((v) => v.localService);
}

function canSpeak(force: boolean): boolean {
  return synth() !== null && (enabled || force) && pickVoice(localVoices(), voiceName) !== null;
}

/** Say `text` now. Resolves true when it was said to the end. */
function utter(text: string, force: boolean, epoch: number): Promise<boolean> {
  const engine = synth();
  const chunks = splitForSpeech(cleanForSpeech(text));
  const voice = pickVoice(localVoices(), voiceName);
  if (!engine || !voice || chunks.length === 0 || epoch !== stops || !(enabled || force)) {
    return Promise.resolve(false);
  }
  return new Promise<boolean>((resolve) => {
    abortCurrent = () => resolve(false);
    const next = (index: number) => {
      if (epoch !== stops) return resolve(false);
      if (index >= chunks.length) return resolve(true);
      const utterance = new SpeechSynthesisUtterance(chunks[index]);
      utterance.voice = voice;
      utterance.lang = voice.lang;
      utterance.rate = rate;
      utterance.onend = () => next(index + 1);
      utterance.onerror = () => resolve(false);
      engine.speak(utterance);
    };
    next(0);
  });
}

// The list of voices arrives a moment after the page loads.
synth()?.addEventListener?.("voiceschanged", emit);

export const speech = {
  supported: (): boolean => synth() !== null,

  /** The voices installed on this computer. */
  voices: (): VoiceLike[] => localVoices().map((v) => ({ name: v.name, lang: v.lang, localService: true })),

  /** The name of the voice that would speak now, or null if this computer has none. */
  currentVoice: (): string | null => pickVoice(localVoices(), voiceName)?.name ?? null,

  configure(options: { enabled: boolean; voiceName: string; rate: number }): void {
    const wasEnabled = enabled;
    enabled = options.enabled;
    voiceName = options.voiceName;
    rate = Math.min(1.5, Math.max(0.6, options.rate));
    if (wasEnabled && !enabled) speech.stop();
    emit();
  },

  /**
   * Read `text` aloud after anything already being said. Resolves true when it was said to
   * the end, false if it was stopped or could not be said. `force` speaks even when spoken
   * replies are switched off (the test button).
   */
  say(text: string, force = false): Promise<boolean> {
    if (!canSpeak(force)) return Promise.resolve(false);
    const epoch = stops;
    pending++;
    emit();
    const result = tail
      .then(() => utter(text, force, epoch))
      .finally(() => {
        pending--;
        emit();
      });
    tail = result.catch(() => false);
    return result;
  },

  /** Read `text` aloud now, instead of anything being said. */
  speak(text: string, force = false): Promise<boolean> {
    speech.stop();
    return speech.say(text, force);
  },

  /** Stop talking, and drop everything that was waiting to be said. */
  stop(): void {
    stops++;
    synth()?.cancel();
    abortCurrent?.();
    abortCurrent = null;
  },

  speaking: (): boolean => pending > 0,

  /** Resolves once nothing is being said or waiting to be said. */
  async whenQuiet(): Promise<void> {
    let seen: Promise<unknown>;
    do {
      seen = tail;
      await seen;
    } while (seen !== tail);
  },

  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useSpeaking(): boolean {
  return useSyncExternalStore(speech.subscribe, speech.speaking, () => false);
}

/** The installed voices, refreshed when the browser finishes loading them. */
export function useVoices(): VoiceLike[] {
  // The snapshot must be stable between changes, so it is compared by its names.
  return useSyncExternalStore(speech.subscribe, voicesSnapshot, () => NO_VOICES);
}

const NO_VOICES: VoiceLike[] = [];
let lastVoices: VoiceLike[] = NO_VOICES;

function voicesSnapshot(): VoiceLike[] {
  const now = speech.voices();
  const same = now.length === lastVoices.length && now.every((v, i) => v.name === lastVoices[i].name);
  if (!same) lastVoices = now;
  return lastVoices;
}
