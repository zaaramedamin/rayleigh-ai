import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { ApiError } from "../api/client";
import type { Api, VoiceStatus } from "../api/types";
import { MicrophoneError, listen, microphoneSupported } from "../audio/recorder";
import { sound } from "../audio/sound";
import { speech } from "../audio/speech";
import { events } from "./events";
import { useSettings } from "./settings";
import type { AskOptions } from "./useAssistant";

// idle: the microphone is off. listening: recording one order. transcribing: the backend is
// turning the recording into text.
export type VoiceState = "idle" | "listening" | "transcribing";

// The microphone loudness changes many times a second, so it lives outside React state and
// only the meter that shows it re-renders.
let level = 0;
const levelListeners = new Set<() => void>();

function setLevel(value: number): void {
  level = value;
  levelListeners.forEach((listener) => listener());
}

function subscribeLevel(listener: () => void): () => void {
  levelListeners.add(listener);
  return () => levelListeners.delete(listener);
}

/** The microphone loudness right now, 0..1. */
export function useMicLevel(): number {
  return useSyncExternalStore(
    subscribeLevel,
    () => level,
    () => 0,
  );
}

export interface Voice {
  state: VoiceState;
  /** This browser can record from a microphone. */
  supported: boolean;
  /** What the backend says about its speech model; null until it answered. */
  status: VoiceStatus | null;
  /** Start listening for an order. Call it from a click. */
  start: () => void;
  /** Stop listening, and stop talking. */
  stop: () => void;
}

/**
 * Voice orders: listen, turn the recording into text on this machine, and hand the text to the
 * assistant. With hands-free on, the microphone opens again after each spoken reply, until
 * nothing is said or `stop` is called.
 */
export function useVoice(
  api: Api,
  ask: (text: string, options?: AskOptions) => Promise<void>,
  enabled: boolean,
): Voice {
  const [state, setState] = useState<VoiceState>("idle");
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const { settings } = useSettings();
  const running = useRef(false);
  const controller = useRef<AbortController | null>(null);

  // The loop below outlives many renders, so it reads the latest of these through refs.
  const askRef = useRef(ask);
  askRef.current = ask;
  const handsFree = useRef(settings.handsFree);
  handsFree.current = settings.handsFree;

  // Ask the backend to load its speech model as soon as the application is open, so the first
  // order does not wait for it.
  useEffect(() => {
    if (!enabled) {
      setStatus(null);
      return;
    }
    let cancelled = false;
    const prepare = microphoneSupported() ? api.prepareVoice() : api.voiceStatus();
    prepare.then(
      (next) => !cancelled && setStatus(next),
      () => !cancelled && setStatus(null),
    );
    return () => {
      cancelled = true;
    };
  }, [api, enabled]);

  const stop = useCallback(() => {
    running.current = false;
    controller.current?.abort();
    speech.stop();
  }, []);

  // Leaving the application (lock, demo switch) closes the microphone.
  useEffect(() => {
    if (!enabled) stop();
  }, [enabled, stop]);
  useEffect(() => stop, [stop]);

  const start = useCallback(() => {
    if (running.current) return;
    running.current = true;

    const run = async () => {
      // Check the speech model first: without it there is no point opening the microphone.
      let current: VoiceStatus;
      try {
        current = await api.prepareVoice();
        setStatus(current);
      } catch (error) {
        const outdated = error instanceof ApiError && error.kind === "not_found";
        events.notify("warning", outdated ? "BACKEND OUT OF DATE" : "VOICE ORDERS UNAVAILABLE", {
          detail: outdated ? "Restart the backend to use voice orders." : messageOf(error),
        });
        return;
      }
      if (current.state === "not_downloaded") {
        events.notify("warning", "SPEECH MODEL NOT DOWNLOADED", { detail: current.hint ?? undefined });
        return;
      }

      while (running.current) {
        speech.stop();
        const abort = new AbortController();
        controller.current = abort;
        setState("listening");
        sound.play("blip");
        const recording = await listen({ signal: abort.signal, onLevel: setLevel });
        setLevel(0);
        if (!recording || !running.current) return; // nothing was said, or it was stopped

        setState("transcribing");
        const { text } = await api.transcribe(recording.wav);
        if (!running.current) return;
        if (!text.trim()) {
          events.notify("info", "I DID NOT CATCH THAT", { detail: "Try again, a little closer to the microphone." });
          return;
        }
        setState("idle");
        await askRef.current(text, { spoken: true });
        if (!handsFree.current) return;
        await speech.whenQuiet();
      }
    };

    run()
      .catch((error: unknown) => {
        if (error instanceof MicrophoneError) {
          events.notify("warning", "MICROPHONE NOT AVAILABLE", { detail: error.message });
        } else {
          events.notify("error", "COULD NOT UNDERSTAND THE RECORDING", { detail: messageOf(error) });
        }
      })
      .finally(() => {
        running.current = false;
        controller.current = null;
        setLevel(0);
        setState("idle");
      });
  }, [api]);

  return { state, supported: microphoneSupported(), status, start, stop };
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}
