import { useEffect, useRef, useState } from "react";
import type { Api, Summary } from "../api/types";

/** The sentence under a summary of a document that was longer than one summary reads, or null. */
export function summaryNote(summary: Pick<Summary, "truncated" | "parts" | "covered_parts">): string | null {
  if (!summary.truncated) return null;
  return `This document is long: only the first ${summary.covered_parts} of its ${summary.parts} parts were summarized.`;
}

/** Why a document cannot be summarized right now, or null when it can. */
export function summaryBlocked(document: { status: "active" | "missing"; chunks: number }): string | null {
  if (document.status === "missing") return "This file was not found in the last two updates, so it cannot be summarized.";
  if (document.chunks === 0) return "This document has not been cut into passages yet: press UPDATE LIBRARY first.";
  return null;
}

/** A short summary of one document, written by the local model on request. */
export function DocumentSummary({ api, id, blocked }: { api: Api; id: number; blocked: string | null }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const shown = useRef(id);

  // Another document was opened: forget the last one's summary, and ignore a reply still on its way.
  useEffect(() => {
    shown.current = id;
    setSummary(null);
    setError("");
    setWorking(false);
  }, [id]);

  async function summarize() {
    const asked = id;
    setWorking(true);
    setError("");
    setSummary(null);
    try {
      const result = await api.summarizeDocument(asked);
      if (shown.current === asked) setSummary(result);
    } catch (err) {
      if (shown.current === asked) setError(err instanceof Error ? err.message : "Could not summarize this document.");
    } finally {
      if (shown.current === asked) setWorking(false);
    }
  }

  const note = summary ? summaryNote(summary) : null;
  return (
    <div className="inspector-summary">
      <div className="inspector-summary-bar">
        <button className="btn btn--ghost" disabled={working || blocked !== null} onClick={() => void summarize()}>
          {working ? "READING..." : summary ? "SUMMARIZE AGAIN" : "SUMMARIZE"}
        </button>
        <span className="muted">
          {blocked ??
            (working
              ? "The local model is reading the document. A long one can take a minute or two."
              : "Written by the local model from this document alone.")}
        </span>
      </div>
      {error && (
        <p className="bad" role="alert">
          {error}
        </p>
      )}
      {summary && (
        <>
          <p className="inspector-summary-text">{summary.text}</p>
          {note && <p className="warn">{note}</p>}
          <p className="muted">A summary has no sources and the model can make mistakes: check it against the passages below.</p>
        </>
      )}
    </div>
  );
}
