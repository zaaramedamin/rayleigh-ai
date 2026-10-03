import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { Api, FileTypeInfo, SearchResult } from "../api/types";
import { HudFrame } from "../components/HudFrame";
import { SourceCard } from "../components/SourceCard";

type Phase = "idle" | "loading" | "done" | "error";

/** Raw semantic search: what the retriever finds, before any model writes an answer. */
export function KnowledgeView({ api }: { api: Api }) {
  const [query, setQuery] = useState("");
  const [types, setTypes] = useState<FileTypeInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .fileTypes()
      .then(setTypes)
      .catch(() => setTypes([]));
  }, [api]);

  const toggle = (ext: string) =>
    setSelected((s) => (s.includes(ext) ? s.filter((e) => e !== ext) : [...s, ext]));

  const run = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setPhase("loading");
    try {
      setResults(await api.search(query.trim(), selected.length ? { file_types: selected } : {}));
      setPhase("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed.");
      setPhase("error");
    }
  };

  return (
    <div className="stack">
      <HudFrame title="KNOWLEDGE SEARCH" tag="RETRIEVAL ONLY // NO MODEL">
        <form className="command" onSubmit={run}>
          <span className="prompt" aria-hidden>
            ?
          </span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by meaning, not just keywords..."
            aria-label="Search your notes"
            maxLength={2000}
          />
          <button className="btn" disabled={!query.trim() || phase === "loading"}>
            SCAN
          </button>
        </form>
        {types.length > 0 && (
          <div className="chips" aria-label="File types">
            {types.flatMap((t) =>
              t.extensions.map((ext) => (
                <button
                  key={ext}
                  className={`chip${selected.includes(ext) ? " chip--on" : ""}`}
                  onClick={() => toggle(ext)}
                  title={t.description}
                >
                  {ext}
                </button>
              )),
            )}
          </div>
        )}
      </HudFrame>

      <HudFrame title="RESULTS" tag={phase === "done" ? `${results.length} FOUND` : phase.toUpperCase()}>
        {phase === "error" && <p className="bad" role="alert">{error}</p>}
        {phase === "done" && results.length === 0 && <p className="muted">Nothing indexed matches that.</p>}
        {phase === "idle" && <p className="muted">Results show the exact passages, with file, heading and line range.</p>}
        <div className="source-list source-list--grid">
          {results.map((r) => (
            <SourceCard key={r.citation_id} item={r} />
          ))}
        </div>
      </HudFrame>
    </div>
  );
}
