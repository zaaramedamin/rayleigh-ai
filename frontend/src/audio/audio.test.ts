import { describe, expect, it } from "vitest";
import { SpeechDetector, encodeWav, loudness } from "./recorder";
import { cleanForSpeech, pickVoice, splitForSpeech } from "./speech";

describe("encodeWav", () => {
  it("writes a 16 kHz, 16-bit mono WAV the backend can read", () => {
    const wav = new DataView(encodeWav(new Float32Array([0, 0.5, -0.5, 2, -2])));
    const text = (at: number, n: number) => String.fromCharCode(...Array.from({ length: n }, (_, i) => wav.getUint8(at + i)));
    expect(text(0, 4)).toBe("RIFF");
    expect(text(8, 4)).toBe("WAVE");
    expect(wav.getUint16(22, true)).toBe(1); // mono
    expect(wav.getUint32(24, true)).toBe(16000);
    expect(wav.getUint16(34, true)).toBe(16);
    expect(wav.getUint32(40, true)).toBe(10); // five samples of two bytes
    expect(wav.byteLength).toBe(54);
    expect(wav.getInt16(46, true)).toBe(16384);
    // Louder than full scale is clipped, not wrapped around.
    expect(wav.getInt16(50, true)).toBe(32767);
    expect(wav.getInt16(52, true)).toBe(-32767);
  });
});

describe("loudness", () => {
  it("measures the level of a block", () => {
    expect(loudness(new Float32Array(0))).toBe(0);
    expect(loudness(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(0.5);
  });
});

describe("SpeechDetector", () => {
  const options = { waitForSpeechMs: 2000, silenceMs: 500, maxMs: 6000, minSpeechMs: 100 };
  const feed = (detector: SpeechDetector, level: number, ms: number) => {
    let state = "waiting";
    for (let t = 0; t < ms; t += 50) state = detector.push(level, 50);
    return state;
  };

  it("finishes after speech followed by a pause", () => {
    const detector = new SpeechDetector(options);
    expect(feed(detector, 0.002, 300)).toBe("waiting"); // room noise
    expect(feed(detector, 0.2, 400)).toBe("speaking");
    expect(feed(detector, 0.002, 600)).toBe("done");
  });

  it("gives up when nobody speaks", () => {
    expect(feed(new SpeechDetector(options), 0.002, 2500)).toBe("nothing");
  });

  it("ignores a click that is too short to be speech", () => {
    const detector = new SpeechDetector(options);
    feed(detector, 0.002, 300);
    detector.push(0.3, 50);
    expect(feed(detector, 0.002, 400)).toBe("waiting");
  });

  it("is not stopped by a short pause inside a sentence", () => {
    const detector = new SpeechDetector(options);
    feed(detector, 0.002, 300);
    feed(detector, 0.2, 400);
    expect(feed(detector, 0.002, 300)).toBe("speaking");
    expect(feed(detector, 0.2, 200)).toBe("speaking");
  });

  it("stops at the longest allowed time", () => {
    const detector = new SpeechDetector(options);
    feed(detector, 0.002, 300);
    expect(feed(detector, 0.2, 7000)).toBe("done");
  });

  it("treats a hum in the room as background, not speech", () => {
    const detector = new SpeechDetector(options);
    expect(feed(detector, 0.05, 300)).toBe("waiting");
    expect(feed(detector, 0.05, 1000)).toBe("waiting");
    expect(detector.threshold).toBeGreaterThan(0.05);
  });
});

describe("speech text", () => {
  it("removes what a voice would spell out", () => {
    expect(cleanForSpeech("It leaves at 07:45 [1][2]. See https://x.com/a **now** #1")).toBe(
      "It leaves at 07:45. See a link now 1",
    );
  });

  it("splits long text at sentence ends, and long sentences at spaces", () => {
    const chunks = splitForSpeech("One two three. Four five six. Seven eight nine.", 20);
    expect(chunks).toEqual(["One two three.", "Four five six.", "Seven eight nine."]);
    const long = splitForSpeech("word ".repeat(100), 50);
    expect(long.every((c) => c.length <= 50)).toBe(true);
    expect(long.join(" ").split(" ").filter(Boolean)).toHaveLength(100);
    expect(splitForSpeech("")).toEqual([]);
  });
});

describe("pickVoice", () => {
  const voice = (name: string, lang: string, localService: boolean) => ({ name, lang, localService });
  const voices = [
    voice("Microsoft Aria Online (Natural)", "en-US", false),
    voice("Microsoft Zira Desktop", "en-US", true),
    voice("Microsoft David Desktop", "en-US", true),
    voice("Microsoft Hortense", "fr-FR", true),
  ];

  it("never picks a voice that works through an online service", () => {
    expect(pickVoice(voices, "Microsoft Aria Online (Natural)")?.localService).toBe(true);
    expect(pickVoice([voices[0]], "")).toBeNull();
  });

  it("honours the chosen voice if it is installed, else the best English one", () => {
    expect(pickVoice(voices, "Microsoft Hortense")?.name).toBe("Microsoft Hortense");
    expect(pickVoice(voices, "gone")?.name).toBe("Microsoft David Desktop");
  });

  it("falls back to any installed voice", () => {
    expect(pickVoice([voice("Hortense", "fr-FR", true)], "")?.name).toBe("Hortense");
    expect(pickVoice([], "")).toBeNull();
  });
});
