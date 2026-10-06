// All sounds are synthesised here with the Web Audio API: no audio files, no downloads, nothing
// that could reach the network. Each function schedules a sound starting at time `at` and
// returns how long it lasts, so it can be rendered offline in a test.

export type SoundName =
  | "boot"
  | "scan"
  | "whoosh"
  | "landed"
  | "blip"
  | "click"
  | "send"
  | "receive"
  | "error"
  | "alert"
  | "denied"
  | "warning"
  | "success"
  | "remove"
  | "lock";

interface Tone {
  type: OscillatorType;
  from: number;
  to?: number;
  at: number;
  length: number;
  gain: number;
  attack?: number;
}

interface Noise {
  at: number;
  length: number;
  gain: number;
  from: number;
  to: number;
  q?: number;
}

const SILENT = 0.0001;

function tone(ctx: BaseAudioContext, out: AudioNode, s: Tone): void {
  const osc = ctx.createOscillator();
  const env = ctx.createGain();
  osc.type = s.type;
  osc.frequency.setValueAtTime(s.from, s.at);
  if (s.to) osc.frequency.exponentialRampToValueAtTime(s.to, s.at + s.length);
  env.gain.setValueAtTime(SILENT, s.at);
  env.gain.exponentialRampToValueAtTime(s.gain, s.at + (s.attack ?? 0.005));
  env.gain.exponentialRampToValueAtTime(SILENT, s.at + s.length);
  osc.connect(env).connect(out);
  osc.start(s.at);
  osc.stop(s.at + s.length + 0.02);
}

/** Band-passed noise whose centre frequency sweeps from `from` to `to`: wind, scans, whooshes. */
function noise(ctx: BaseAudioContext, out: AudioNode, s: Noise): void {
  const frames = Math.ceil(ctx.sampleRate * s.length);
  const buffer = ctx.createBuffer(1, frames, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) data[i] = Math.random() * 2 - 1;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const filter = ctx.createBiquadFilter();
  filter.type = "bandpass";
  filter.Q.value = s.q ?? 1;
  filter.frequency.setValueAtTime(s.from, s.at);
  filter.frequency.exponentialRampToValueAtTime(s.to, s.at + s.length);
  const env = ctx.createGain();
  env.gain.setValueAtTime(SILENT, s.at);
  env.gain.exponentialRampToValueAtTime(s.gain, s.at + s.length * 0.3);
  env.gain.exponentialRampToValueAtTime(SILENT, s.at + s.length);
  source.connect(filter).connect(env).connect(out);
  source.start(s.at);
}

type Recipe = (ctx: BaseAudioContext, out: AudioNode, t: number) => number;

const RECIPES: Record<SoundName, Recipe> = {
  // A low reactor hum that rises, with a few distant data chirps and a wash of air.
  boot: (ctx, out, t) => {
    tone(ctx, out, { type: "sine", from: 55, to: 110, at: t, length: 2.8, gain: 0.5, attack: 0.9 });
    tone(ctx, out, { type: "triangle", from: 110, to: 220, at: t, length: 2.8, gain: 0.16, attack: 1.1 });
    noise(ctx, out, { at: t, length: 2.6, gain: 0.05, from: 300, to: 3200, q: 0.8 });
    [
      [0.6, 1760],
      [1.1, 2093],
      [1.7, 1568],
      [2.2, 2349],
    ].forEach(([dt, f]) => tone(ctx, out, { type: "sine", from: f, at: t + dt, length: 0.14, gain: 0.07 }));
    return 2.9;
  },
  // A short two-note data tick for each line of the start-up log.
  scan: (ctx, out, t) => {
    tone(ctx, out, { type: "square", from: 1320, at: t, length: 0.05, gain: 0.16 });
    tone(ctx, out, { type: "square", from: 1760, at: t + 0.055, length: 0.05, gain: 0.12 });
    return 0.12;
  },
  // The interface sweeping open.
  whoosh: (ctx, out, t) => {
    noise(ctx, out, { at: t, length: 1, gain: 0.8, from: 200, to: 5200, q: 1.2 });
    tone(ctx, out, { type: "sine", from: 90, to: 420, at: t, length: 1, gain: 0.4, attack: 0.3 });
    return 1.05;
  },
  // The reactor locking into place: a thump, a chord and a glint.
  landed: (ctx, out, t) => {
    tone(ctx, out, { type: "sine", from: 110, to: 50, at: t, length: 0.55, gain: 0.6 });
    [440, 660, 880].forEach((f) => tone(ctx, out, { type: "sine", from: f, at: t, length: 0.9, gain: 0.1, attack: 0.03 }));
    tone(ctx, out, { type: "sine", from: 1760, at: t + 0.06, length: 0.45, gain: 0.05 });
    return 0.95;
  },
  blip: (ctx, out, t) => {
    tone(ctx, out, { type: "triangle", from: 1500, to: 1900, at: t, length: 0.07, gain: 0.3 });
    return 0.1;
  },
  click: (ctx, out, t) => {
    tone(ctx, out, { type: "sine", from: 720, to: 480, at: t, length: 0.05, gain: 0.26 });
    return 0.08;
  },
  send: (ctx, out, t) => {
    tone(ctx, out, { type: "triangle", from: 520, to: 780, at: t, length: 0.11, gain: 0.32 });
    return 0.14;
  },
  receive: (ctx, out, t) => {
    tone(ctx, out, { type: "triangle", from: 780, to: 520, at: t, length: 0.12, gain: 0.3 });
    tone(ctx, out, { type: "triangle", from: 1040, at: t + 0.11, length: 0.12, gain: 0.24 });
    return 0.26;
  },
  // System-level failure: a rising and falling two-tone siren with a hit of static at the start.
  alert: (ctx, out, t) => {
    noise(ctx, out, { at: t, length: 0.25, gain: 0.5, from: 3000, to: 600, q: 0.7 });
    tone(ctx, out, { type: "sine", from: 70, to: 45, at: t, length: 1.5, gain: 0.55, attack: 0.02 });
    [0, 0.32, 0.64, 0.96].forEach((dt, i) =>
      tone(ctx, out, {
        type: "sawtooth",
        from: i % 2 === 0 ? 760 : 560,
        to: i % 2 === 0 ? 560 : 760,
        at: t + dt,
        length: 0.3,
        gain: 0.2,
        attack: 0.02,
      }),
    );
    return 1.35;
  },
  // Access refused: two low, flat buzzes.
  denied: (ctx, out, t) => {
    tone(ctx, out, { type: "square", from: 170, to: 120, at: t, length: 0.16, gain: 0.3, attack: 0.01 });
    tone(ctx, out, { type: "square", from: 170, to: 105, at: t + 0.22, length: 0.22, gain: 0.3, attack: 0.01 });
    return 0.5;
  },
  // Something needs attention: two even mid beeps.
  warning: (ctx, out, t) => {
    tone(ctx, out, { type: "sine", from: 660, at: t, length: 0.13, gain: 0.38, attack: 0.01 });
    tone(ctx, out, { type: "sine", from: 660, at: t + 0.19, length: 0.16, gain: 0.38, attack: 0.01 });
    tone(ctx, out, { type: "triangle", from: 1320, at: t, length: 0.1, gain: 0.08 });
    return 0.4;
  },
  // It worked: a quick rising arpeggio.
  success: (ctx, out, t) => {
    [523.25, 659.25, 783.99, 1046.5].forEach((f, i) =>
      tone(ctx, out, { type: "triangle", from: f, at: t + i * 0.075, length: 0.18, gain: 0.42, attack: 0.008 }),
    );
    return 0.5;
  },
  // Something was deleted: a short falling zap.
  remove: (ctx, out, t) => {
    tone(ctx, out, { type: "sawtooth", from: 720, to: 110, at: t, length: 0.26, gain: 0.34, attack: 0.005 });
    noise(ctx, out, { at: t, length: 0.12, gain: 0.25, from: 4000, to: 800, q: 1 });
    return 0.3;
  },
  // Locking the application: a closing sweep.
  lock: (ctx, out, t) => {
    tone(ctx, out, { type: "sine", from: 780, to: 140, at: t, length: 0.4, gain: 0.34, attack: 0.01 });
    tone(ctx, out, { type: "triangle", from: 390, to: 70, at: t, length: 0.4, gain: 0.2, attack: 0.01 });
    return 0.45;
  },
  error: (ctx, out, t) => {
    tone(ctx, out, { type: "sawtooth", from: 150, to: 80, at: t, length: 0.32, gain: 0.32 });
    return 0.36;
  },
};

/** Schedule `name` on `ctx` starting at time `t`. Returns its length in seconds. */
export function schedule(name: SoundName, ctx: BaseAudioContext, out: AudioNode, t: number): number {
  return RECIPES[name](ctx, out, t);
}

export const SOUND_NAMES = Object.keys(RECIPES) as SoundName[];
