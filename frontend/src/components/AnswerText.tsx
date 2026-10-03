import { useEffect, useState } from "react";
import type { AskSource } from "../api/types";

interface Props {
  text: string;
  sources: AskSource[];
  animate: boolean;
  activeMarker: number | null;
  onMarker: (marker: number) => void;
}

/** Splits "text [1] more [2]" into plain parts and the citation numbers the app can verify. */
export function splitCitations(text: string, valid: Set<number>): Array<string | number> {
  const parts: Array<string | number> = [];
  let last = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    const marker = Number(match[1]);
    if (!valid.has(marker)) continue;
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(marker);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

/** The answer, revealed like a terminal; [n] markers become buttons that open the source. */
export function AnswerText({ text, sources, animate, activeMarker, onMarker }: Props) {
  const [count, setCount] = useState(animate ? 0 : text.length);

  useEffect(() => {
    if (!animate || count >= text.length) return;
    const timer = setTimeout(() => setCount((c) => Math.min(text.length, c + 4)), 14);
    return () => clearTimeout(timer);
  }, [animate, count, text]);

  const valid = new Set(sources.map((s) => s.marker));
  const parts = splitCitations(text.slice(0, count), valid);
  return (
    <p className="answer-text">
      {parts.map((part, i) =>
        typeof part === "number" ? (
          <button
            key={i}
            className={`cite${activeMarker === part ? " cite--active" : ""}`}
            onClick={() => onMarker(part)}
            aria-label={`Show source ${part}`}
          >
            {part}
          </button>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
      {count < text.length && <span className="caret" aria-hidden />}
    </p>
  );
}
