import { useEffect, useState } from "react";
import { useSettings } from "../state/settings";

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*+<>/";

interface Props {
  text: string;
  /** How long the text takes to settle, in milliseconds. */
  duration?: number;
  className?: string;
}

/** Text that deciphers itself: random characters resolve into the real ones, left to right. */
export function ScrambleText({ text, duration = 650, className }: Props) {
  const { settings } = useSettings();
  const [shown, setShown] = useState(settings.reducedMotion ? text : "");

  useEffect(() => {
    if (settings.reducedMotion) {
      setShown(text);
      return;
    }
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const settled = Math.floor(progress * text.length);
      let out = "";
      for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        out += i < settled || ch === " " ? ch : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
      }
      setShown(out);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [text, duration, settings.reducedMotion]);

  return (
    <span className={className} aria-label={text}>
      {shown}
    </span>
  );
}
