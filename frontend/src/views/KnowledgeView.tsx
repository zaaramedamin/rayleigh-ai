import { Fragment, useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import type {
  Api,
  FileTypeInfo,
  JobInfo,
  LibraryDocument,
  LibraryFolder,
  SearchMode,
  SearchResult,
  SyncStatus,
} from "../api/types";
import { DocumentInspector } from "../components/DocumentInspector";
import { HudFrame } from "../components/HudFrame";
import { jobSummary } from "../lib/documents";
import { events } from "../state/events";
import { SourceCard } from "../components/SourceCard";

type Tab = "library" | "search";

export function KnowledgeView({ api, onChanged }: { api: Api; onChanged: () => void }) {
  const [tab, setTab] = useState<Tab>("library");
  return (
    <div className="stack">
      <div className="tabs" role="tablist">
        {(["library", "search"] as const).map((id) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={`tab${tab === id ? " tab--on" : ""}`}
            onClick={() => setTab(id)}
          >
            {id === "library" ? "LIBRARY" : "SEARCH"}
          </button>
        ))}
      </div>
      {tab === "library" ? <LibraryTab api={api} onChanged={onChanged} /> : <SearchTab api={api} />}
    </div>
  );
}

// --- library -----------------------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const HISTORY_SHOWN = 5;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function SyncSummary({ sync }: { sync: SyncStatus }) {
  if (sync.state === "idle") return null;
  if (sync.state === "running") {
    const indexing = sync.phase === "indexing";
    const byChunks = indexing && sync.chunks_total > 0;
    const share = byChunks
      ? sync.chunks_done / sync.chunks_total
      : indexing && sync.indexing_total > 0
        ? sync.indexed / sync.indexing_total
        : 0;
    return (
      <div className="sync" role="status">
        <p>
          {indexing
            ? byChunks
              ? `Making notes searchable: ${sync.chunks_done.toLocaleString()} of ${sync.chunks_total.toLocaleString()} parts`
              : `Making notes searchable: ${sync.indexed} of ${sync.indexing_total}`
            : "Reading your folders..."}
        </p>
        <div className="meter">
          <span style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
      </div>
    );
  }
  return (
    <div className={`sync${sync.state === "failed" ? " sync--bad" : ""}`} role="status">
      <p>
        {sync.state === "failed" ? "The update stopped." : "Update finished."}{" "}
        {sync.state === "done" &&
          `Added ${sync.added}, unchanged ${sync.unchanged}, made searchable ${sync.indexed}` +
            (sync.skipped_excluded ? `, kept out ${sync.skipped_excluded}` : "") +
            (sync.failed_files ? `, could not read ${sync.failed_files}` : "") +
            (sync.replaced ? `, replaced by an edit ${sync.replaced}` : "") +
            (sync.missing ? `, file not found ${sync.missing}` : "") +
            "."}
      </p>
      {sync.message && <p className="hint">{sync.message}</p>}
    </div>
  );
}

function LibraryTab({ api, onChanged }: { api: Api; onChanged: () => void }) {
  const [documents, setDocuments] = useState<LibraryDocument[]>([]);
  const [removedCount, setRemovedCount] = useState(0);
  const [folders, setFolders] = useState<LibraryFolder[]>([]);
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [filter, setFilter] = useState("");
  const [newPath, setNewPath] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [confirmDoc, setConfirmDoc] = useState<number | null>(null);
  const [inspecting, setInspecting] = useState<number | null>(null);
  const [confirmFolder, setConfirmFolder] = useState<string | null>(null);
  const [alsoRemove, setAlsoRemove] = useState(false);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [docs, dirs] = await Promise.all([api.documents(), api.folders()]);
      setDocuments(docs.documents);
      setRemovedCount(docs.removed_count);
      setFolders(dirs);
      setError("");
      // The history is a convenience: a failure to read it must not hide the library.
      api.jobs().then((list) => setJobs(list.slice(0, HISTORY_SHOWN)), () => undefined);
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    void reload();
    api.syncStatus().then(setSync, () => undefined);
  }, [api, reload]);

  // While an update runs, follow its progress; when it ends, show the new library.
  const running = sync?.state === "running";
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(async () => {
      try {
        const next = await api.syncStatus();
        setSync(next);
        if (next.state !== "running") {
          void reload();
          onChanged();
          if (next.state === "failed") {
            events.notify("error", "UPDATE FAILED", { detail: next.message ?? undefined });
          } else if (next.message) {
            events.notify("warning", "LIBRARY UPDATED, WITH A NOTE", { detail: next.message });
          } else {
            events.notify("success", "LIBRARY UPDATED", {
              detail: `${next.added} added, ${next.indexed} made searchable`,
            });
          }
        }
      } catch {
        // the next tick tries again
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [running, api, reload, onChanged]);

  const act = async (work: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await work();
      onChanged();
    } catch (err) {
      setError(messageOf(err));
      events.notify("error", "THAT DID NOT WORK", { detail: messageOf(err) });
    } finally {
      setBusy(false);
    }
  };

  const addFolder = (event: FormEvent) => {
    event.preventDefault();
    if (!newPath.trim()) return;
    void act(async () => {
      setFolders(await api.addFolder(newPath.trim()));
      setNewPath("");
      events.notify("success", "FOLDER ADDED", { detail: "Press UPDATE LIBRARY to read it." });
    });
  };

  const removeFolder = (path: string) =>
    act(async () => {
      const result = await api.removeFolder(path, alsoRemove);
      events.notify("info", "FOLDER REMOVED", {
        detail: result.documents_removed ? `${result.documents_removed} documents removed too` : undefined,
        sound: "remove",
      });
      setConfirmFolder(null);
      setAlsoRemove(false);
      await reload();
    });

  const removeDocument = (id: number) =>
    act(async () => {
      await api.deleteDocument(id);
      events.notify("info", "DOCUMENT REMOVED", { detail: "Your original file was not touched.", sound: "remove" });
      setConfirmDoc(null);
      setInspecting((open) => (open === id ? null : open));
      await reload();
    });

  const update = () =>
    act(async () => {
      setSync(await api.startSync());
      events.notify("info", "UPDATE STARTED", { toast: false, sound: null });
    });

  const needle = filter.trim().toLowerCase();
  const shown = documents.filter(
    (d) => !needle || d.name.toLowerCase().includes(needle) || (d.source ?? "").toLowerCase().includes(needle),
  );
  const missingFiles = documents.filter((d) => d.status === "missing").length;
  const pending = documents.filter((d) => !d.searchable && d.status !== "missing").length;

  return (
    <>
      <HudFrame title="FOLDERS" tag="WHAT REYLEIGHT MAY READ">
        {error && (
          <p className="bad" role="alert">
            {error}
          </p>
        )}
        {loading && <p className="muted">Loading...</p>}
        {!loading && folders.length === 0 && (
          <p className="muted">No folders yet. Add one below, then press UPDATE LIBRARY.</p>
        )}
        <ul className="rows">
          {folders.map((folder) => (
            <li key={folder.path} className="row-item">
              <div className="row-main">
                <span className="mono ellipsis" title={folder.path}>
                  {folder.path}
                </span>
                <span className="row-meta">
                  <span className="badge">{folder.origin === "env" ? "FROM .ENV" : "ADDED HERE"}</span>
                  {!folder.exists && <span className="badge badge--bad">NOT FOUND</span>}
                  <span>{folder.documents} documents</span>
                </span>
              </div>
              {folder.removable &&
                (confirmFolder === folder.path ? (
                  <div className="confirm">
                    <label className="toggle toggle--small">
                      <input type="checkbox" checked={alsoRemove} onChange={(e) => setAlsoRemove(e.target.checked)} />
                      <span>Also remove its {folder.documents} documents</span>
                    </label>
                    <button className="btn btn--danger" disabled={busy} onClick={() => void removeFolder(folder.path)}>
                      REMOVE
                    </button>
                    <button className="link" onClick={() => setConfirmFolder(null)}>
                      cancel
                    </button>
                  </div>
                ) : (
                  <button className="btn btn--ghost" onClick={() => setConfirmFolder(folder.path)}>
                    REMOVE FOLDER
                  </button>
                ))}
            </li>
          ))}
        </ul>

        <form className="command" onSubmit={addFolder}>
          <span className="prompt" aria-hidden>
            +
          </span>
          <input
            value={newPath}
            onChange={(e) => setNewPath(e.target.value)}
            placeholder="Full path of a folder, e.g. C:\Users\you\Documents\notes"
            aria-label="Folder path"
            maxLength={1000}
          />
          <button className="btn" disabled={busy || !newPath.trim()}>
            ADD FOLDER
          </button>
        </form>
        <p className="command-hint">
          Only folders you add here (or list in .env) are ever read. A native folder picker comes with the desktop app.
        </p>

        <div className="sync-bar">
          <button className="btn" disabled={busy || running || folders.length === 0} onClick={() => void update()}>
            {running ? "UPDATING..." : "UPDATE LIBRARY"}
          </button>
          <span className="muted">Reads the folders, then makes anything new searchable.</span>
        </div>
        {sync && <SyncSummary sync={sync} />}
      </HudFrame>

      <HudFrame title="DOCUMENTS" tag={`${documents.length} IN LIBRARY`}>
        <div className="doc-tools">
          <input
            className="field"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by name or source..."
            aria-label="Filter documents"
          />
          {pending > 0 && <span className="hint">{pending} not searchable yet: press UPDATE LIBRARY</span>}
          {missingFiles > 0 && (
            <span className="hint">
              {missingFiles} file(s) not found in your folders, so not searched. They come back if the file
              returns, or run python -m app prune to remove them.
            </span>
          )}
          {removedCount > 0 && (
            <span className="muted">
              {removedCount} removed and kept out of updates.{" "}
              <button
                className="link"
                onClick={() =>
                  void act(async () => {
                    await api.restoreRemoved();
                    await reload();
                  })
                }
              >
                restore
              </button>
            </span>
          )}
        </div>

        {!loading && shown.length === 0 && (
          <p className="muted">{documents.length === 0 ? "The library is empty." : "No document matches that filter."}</p>
        )}
        <ul className="rows docs">
          {shown.map((doc) => (
            <Fragment key={doc.id}>
            <li className="row-item">
              <div className="row-main">
                <span className="doc-name">
                  <b>{doc.name}</b>
                  {doc.is_profile && <span className="badge">YOUR PROFILE</span>}
                  {doc.status === "missing" && <span className="badge">FILE NOT FOUND</span>}
                </span>
                <span className="mono ellipsis muted" title={doc.source ?? undefined}>
                  {doc.source ?? (doc.is_profile ? "written in the Profile page" : "source not recorded: press UPDATE LIBRARY")}
                </span>
                <span className="row-meta">
                  <span>{formatSize(doc.size_bytes)}</span>
                  <span>{doc.chunks} chunks</span>
                  <span className={doc.searchable ? "ok" : "warn"}>
                    {doc.searchable ? "SEARCHABLE" : doc.status === "missing" ? "NOT SEARCHED" : "NOT INDEXED"}
                  </span>
                </span>
              </div>
              {confirmDoc === doc.id ? (
                <div className="confirm">
                  <span className="muted">Remove from the library? Your original file is not touched.</span>
                  <button className="btn btn--danger" disabled={busy} onClick={() => void removeDocument(doc.id)}>
                    REMOVE
                  </button>
                  <button className="link" onClick={() => setConfirmDoc(null)}>
                    cancel
                  </button>
                </div>
              ) : (
                <div className="row-actions">
                  <button
                    className="btn btn--ghost"
                    aria-expanded={inspecting === doc.id}
                    onClick={() => setInspecting(inspecting === doc.id ? null : doc.id)}
                    aria-label={`Inspect ${doc.name}`}
                  >
                    {inspecting === doc.id ? "HIDE" : "INSPECT"}
                  </button>
                  <button className="btn btn--ghost" onClick={() => setConfirmDoc(doc.id)} aria-label={`Remove ${doc.name}`}>
                    REMOVE
                  </button>
                </div>
              )}
            </li>
            {inspecting === doc.id && (
              <li className="row-detail">
                <DocumentInspector api={api} id={doc.id} onClose={() => setInspecting(null)} />
              </li>
            )}
            </Fragment>
          ))}
        </ul>
      </HudFrame>

      {jobs.length > 0 && (
        <HudFrame title="RECENT UPDATES" tag="WHAT EACH ONE DID">
          <ul className="rows">
            {jobs.map((job) => (
              <li key={job.id} className="row-item">
                <div className="row-main">
                  <span className="doc-name">
                    <b>{new Date(job.started_at).toLocaleString()}</b>
                    <span className={`badge${job.state === "failed" ? " badge--bad" : ""}`}>{job.state.toUpperCase()}</span>
                  </span>
                  <span className="muted">{jobSummary(job)}</span>
                </div>
              </li>
            ))}
          </ul>
        </HudFrame>
      )}
    </>
  );
}

// --- search ------------------------------------------------------------------------------------

type Phase = "idle" | "loading" | "done" | "error";

const SEARCH_MODES: Array<{ id: SearchMode; label: string; hint: string }> = [
  { id: "hybrid", label: "BOTH", hint: "By meaning and by the words in the query, merged" },
  { id: "vector", label: "MEANING", hint: "By meaning: good for a question in your own words" },
  { id: "keyword", label: "WORDS", hint: "By the words in the query: good for exact codes, names and numbers" },
];

/** Raw search: what the retriever finds, before any model writes an answer. */
function SearchTab({ api }: { api: Api }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
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
      setResults(await api.search(query.trim(), { mode, ...(selected.length ? { file_types: selected } : {}) }));
      setPhase("done");
    } catch (err) {
      setError(messageOf(err));
      setPhase("error");
    }
  };

  return (
    <>
      <HudFrame title="KNOWLEDGE SEARCH" tag="RETRIEVAL ONLY // NO MODEL">
        <form className="command" onSubmit={run}>
          <span className="prompt" aria-hidden>
            ?
          </span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by meaning or by exact words..."
            aria-label="Search your notes"
            maxLength={2000}
          />
          <button className="btn" disabled={!query.trim() || phase === "loading"}>
            SCAN
          </button>
        </form>
        <div className="chips" aria-label="Find by">
          {SEARCH_MODES.map((m) => (
            <button
              type="button"
              key={m.id}
              className={`chip${mode === m.id ? " chip--on" : ""}`}
              aria-pressed={mode === m.id}
              onClick={() => setMode(m.id)}
              title={m.hint}
            >
              {m.label}
            </button>
          ))}
        </div>
        {types.length > 0 && (
          <div className="chips" aria-label="File types">
            {types.flatMap((t) =>
              t.extensions.map((ext) => (
                <button
                  key={ext}
                  type="button"
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
        {phase === "error" && (
          <p className="bad" role="alert">
            {error}
          </p>
        )}
        {phase === "done" && results.length === 0 && <p className="muted">Nothing indexed matches that.</p>}
        {phase === "idle" && <p className="muted">Results show the exact passages, with file, heading and line range.</p>}
        <div className="source-list source-list--grid">
          {results.map((r) => (
            <SourceCard key={r.citation_id} item={r} />
          ))}
        </div>
      </HudFrame>
    </>
  );
}
