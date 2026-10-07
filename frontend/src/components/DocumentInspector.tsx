import { useCallback, useEffect, useState } from "react";
import type { Api, ChunkInfo, DocumentDetail } from "../api/types";
import { locationPath, locationState } from "../lib/documents";
import { DocumentSummary, summaryBlocked } from "./DocumentSummary";
import { locationLabel } from "./SourceCard";

const PAGE_SIZE = 20;

function when(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function kilobytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

/** What the library holds for one document: where it was found, its older versions, and its passages. */
export function DocumentInspector({ api, id, onClose }: { api: Api; id: number; onClose: () => void }) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    let current = true;
    setDetail(null);
    setChunks([]);
    setError("");
    Promise.all([api.documentDetail(id), api.documentChunks(id, 0, PAGE_SIZE)]).then(
      ([info, page]) => {
        if (!current) return;
        setDetail(info);
        setChunks(page.chunks);
        setTotal(page.total);
      },
      (err: unknown) => {
        if (current) setError(err instanceof Error ? err.message : "Could not read this document.");
      },
    );
    return () => {
      current = false;
    };
  }, [api, id]);

  const more = useCallback(async () => {
    setLoadingMore(true);
    try {
      const page = await api.documentChunks(id, chunks.length, PAGE_SIZE);
      setChunks((shown) => [...shown, ...page.chunks]);
      setTotal(page.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read more passages.");
    } finally {
      setLoadingMore(false);
    }
  }, [api, id, chunks.length]);

  return (
    <section className="inspector" aria-label="Document details">
      <div className="inspector-head">
        <b>WHAT THE LIBRARY HOLDS</b>
        <button className="link" onClick={onClose}>
          close
        </button>
      </div>
      {error && (
        <p className="bad" role="alert">
          {error}
        </p>
      )}
      {!detail && !error && <p className="muted">Loading...</p>}
      {detail && (
        <>
          <p className="muted">
            {detail.indexed_chunks} of {detail.chunks} passages are searchable.
            {detail.supersedes_id !== null && " This file was edited; this is its newest version."}
          </p>

          <h4 className="inspector-title">SUMMARY</h4>
          <DocumentSummary api={api} id={id} blocked={summaryBlocked(detail)} />

          <h4 className="inspector-title">WHERE IT WAS FOUND</h4>
          {detail.locations.length === 0 ? (
            <p className="muted">
              {detail.is_profile ? "Written in the Profile page." : "No place recorded yet: press UPDATE LIBRARY."}
            </p>
          ) : (
            <ul className="inspector-list">
              {detail.locations.map((place) => (
                <li key={`${place.folder}|${place.path}`}>
                  <span className="mono ellipsis" title={locationPath(place)}>
                    {locationPath(place)}
                  </span>
                  <span className={place.status === "present" ? "ok" : "warn"}>{locationState(place)}</span>
                  <span className="muted">last seen {when(place.last_seen_at)}</span>
                </li>
              ))}
            </ul>
          )}

          {detail.older_versions.length > 0 && (
            <>
              <h4 className="inspector-title">EARLIER VERSIONS</h4>
              <p className="muted">Kept as history. They are never searched or quoted.</p>
              <ul className="inspector-list">
                {detail.older_versions.map((version) => (
                  <li key={version.id}>
                    <span>{when(version.created_at)}</span>
                    <span className="muted">
                      {kilobytes(version.size_bytes)}, {version.chunks} passages
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}

          <h4 className="inspector-title">PASSAGES ({total})</h4>
          {chunks.length === 0 && <p className="muted">This document has not been cut into passages yet.</p>}
          <div className="inspector-chunks">
            {chunks.map((chunk) => (
              <details key={chunk.index} className="inspector-chunk">
                <summary>
                  <span className="mono">#{chunk.index + 1}</span>
                  <span className="ellipsis">{chunk.heading_path || "(no heading)"}</span>
                  <span className="muted">{locationLabel(chunk)}</span>
                  <span className={chunk.searchable ? "ok" : "warn"}>{chunk.searchable ? "SEARCHABLE" : "NOT INDEXED"}</span>
                </summary>
                <pre>{chunk.text}</pre>
              </details>
            ))}
          </div>
          {chunks.length < total && (
            <button className="btn btn--ghost" disabled={loadingMore} onClick={() => void more()}>
              {loadingMore ? "LOADING..." : `SHOW MORE (${total - chunks.length} left)`}
            </button>
          )}
        </>
      )}
    </section>
  );
}
