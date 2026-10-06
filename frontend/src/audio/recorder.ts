// Listening: records one spoken order from the microphone and hands it over as a small WAV
// file for the backend's local speech model. The recording stays in memory, is sent only to
// this machine's own backend, and the microphone is released as soon as the order ends.

export const TARGET_SAMPLE_RATE = 16000;

/** 16-bit mono PCM WAV, the format the backend reads. Samples are -1..1. */
export function encodeWav(samples: Float32Array, sampleRate = TARGET_SAMPLE_RATE): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // size of the format block
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per sample
  view.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true);
  }
  return buffer;
}

/** Loudness of a block of samples, 0..1 (root mean square). */
export function loudness(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

export interface DetectorOptions {
  /** Give up if nobody has started speaking after this long. */
  waitForSpeechMs: number;
  /** A pause this long after speech means the order is finished. */
  silenceMs: number;
  /** Stop here whatever happens. */
  maxMs: number;
  /** Speech must last this long to count, so a click or a cough does not start it. */
  minSpeechMs: number;
}

export const DEFAULT_DETECTOR: DetectorOptions = {
  waitForSpeechMs: 7000,
  silenceMs: 1100,
  maxMs: 25000,
  minSpeechMs: 120,
};

const MIN_THRESHOLD = 0.012;
const NOISE_FACTOR = 3;
const CALIBRATION_MS = 250;

export type DetectorState = "waiting" | "speaking" | "done" | "nothing";

/**
 * Decides when an order starts and ends from the loudness of the microphone. The first moments
 * measure the room's background noise; speech is anything clearly louder than that.
 */
export class SpeechDetector {
  private elapsed = 0;
  private noise = 0;
  private noiseSamples = 0;
  private loudFor = 0;
  private quietFor = 0;
  private state: DetectorState = "waiting";

  constructor(private readonly options: DetectorOptions = DEFAULT_DETECTOR) {}

  get threshold(): number {
    return Math.max(MIN_THRESHOLD, this.noise * NOISE_FACTOR);
  }

  /** Feed the loudness measured over the last `ms` milliseconds. Returns the new state. */
  push(level: number, ms: number): DetectorState {
    if (this.state === "done" || this.state === "nothing") return this.state;
    this.elapsed += ms;

    if (this.state === "waiting" && this.elapsed <= CALIBRATION_MS) {
      this.noise = (this.noise * this.noiseSamples + level) / ++this.noiseSamples;
      return this.state;
    }

    const loud = level >= this.threshold;
    if (this.state === "waiting") {
      this.loudFor = loud ? this.loudFor + ms : 0;
      if (this.loudFor >= this.options.minSpeechMs) this.state = "speaking";
      else if (this.elapsed >= this.options.waitForSpeechMs) this.state = "nothing";
    } else {
      this.quietFor = loud ? 0 : this.quietFor + ms;
      if (this.quietFor >= this.options.silenceMs || this.elapsed >= this.options.maxMs) this.state = "done";
    }
    return this.state;
  }
}

export interface Recording {
  wav: Blob;
  seconds: number;
}

export interface ListenOptions {
  /** Called many times a second with the microphone loudness, 0..1, for the meter. */
  onLevel?: (level: number) => void;
  /** Called once when speech is first heard. */
  onSpeech?: () => void;
  /** Abort to stop listening and discard what was recorded. */
  signal?: AbortSignal;
  detector?: DetectorOptions;
}

export class MicrophoneError extends Error {
  constructor(
    readonly reason: "unsupported" | "denied" | "missing" | "failed",
    message: string,
  ) {
    super(message);
    this.name = "MicrophoneError";
  }
}

export function microphoneSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined" &&
    typeof AudioContext !== "undefined"
  );
}

function microphoneError(error: unknown): MicrophoneError {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return new MicrophoneError("denied", "The microphone is blocked. Allow it for this page in the browser's address bar.");
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return new MicrophoneError("missing", "No microphone was found on this computer.");
  }
  return new MicrophoneError("failed", "The microphone could not be started.");
}

const POLL_MS = 50;

/**
 * Record one spoken order. Resolves with the recording once the speaker pauses, or with null
 * when nothing was said or listening was aborted. Throws MicrophoneError if the microphone
 * cannot be used.
 */
export async function listen(options: ListenOptions = {}): Promise<Recording | null> {
  if (!microphoneSupported()) {
    throw new MicrophoneError("unsupported", "This browser cannot record from the microphone.");
  }
  if (options.signal?.aborted) return null;

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    throw microphoneError(error);
  }

  const context = new AudioContext();
  const release = () => {
    stream.getTracks().forEach((track) => track.stop());
    void context.close().catch(() => undefined);
  };

  try {
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    context.createMediaStreamSource(stream).connect(analyser);
    const block = new Float32Array(analyser.fftSize);

    const recorder = new MediaRecorder(stream);
    const parts: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) parts.push(event.data);
    };
    const stopped = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });
    recorder.start();

    const detector = new SpeechDetector(options.detector);
    let heardSpeech = false;
    const outcome = await new Promise<DetectorState | "aborted">((resolve) => {
      const timer = setInterval(() => {
        analyser.getFloatTimeDomainData(block);
        const level = loudness(block);
        options.onLevel?.(Math.min(1, level * 6));
        const state = detector.push(level, POLL_MS);
        if (state === "speaking" && !heardSpeech) {
          heardSpeech = true;
          options.onSpeech?.();
        }
        if (state === "done" || state === "nothing") finish(state);
      }, POLL_MS);
      const finish = (result: DetectorState | "aborted") => {
        clearInterval(timer);
        options.signal?.removeEventListener("abort", onAbort);
        resolve(result);
      };
      const onAbort = () => finish("aborted");
      options.signal?.addEventListener("abort", onAbort);
    });

    recorder.stop();
    await stopped;
    options.onLevel?.(0);
    if (outcome !== "done") return null;

    // Decode what the browser recorded, then convert it to the rate the speech model expects.
    const encoded = await new Blob(parts, { type: recorder.mimeType }).arrayBuffer();
    const decoded = await context.decodeAudioData(encoded);
    const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, frames, TARGET_SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const samples = (await offline.startRendering()).getChannelData(0);
    return {
      wav: new Blob([encodeWav(samples)], { type: "audio/wav" }),
      seconds: samples.length / TARGET_SAMPLE_RATE,
    };
  } catch (error) {
    if (error instanceof MicrophoneError) throw error;
    throw new MicrophoneError("failed", "The recording could not be read.");
  } finally {
    release();
  }
}
