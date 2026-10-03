# Product architecture and roadmap

How Reyleight grows from a command-line RAG backend into a private assistant with a web app, a Windows `.exe` and a phone app, without giving up its privacy rules. This extends [roadmap.md](roadmap.md) (the MVP, now done) and obeys [security.md](security.md).

Status: the web interface prototype exists in [frontend/](../frontend/) (see [frontend.md](frontend.md)). Everything after it is a plan, not built.

## 1. What the two reference projects teach

Sources: the READMEs of [PersonalJarvis](https://github.com/PersonalJarvis/PersonalJarvis) and [isair/jarvis](https://github.com/isair/jarvis), read as summaries. I did not read their source code, so treat the points below as ideas to verify, not as proven designs.

| Idea | Seen in | Reyleight today | Take it? |
| --- | --- | --- | --- |
| Local server + web UI + desktop shell | PersonalJarvis | Backend only | Yes, section 2 |
| Every tool call goes through an approval policy (safe / monitor / ask / block) | PersonalJarvis | No tools exist | Yes, before any tool ships |
| One entry point that routes to answer / tool / agent | PersonalJarvis | Always retrieves and answers | Later (phase 6) |
| Activity history: every step saved and visible | PersonalJarvis | None | Yes, phase 1 |
| Bounded parallel workers, no unlimited agents | PersonalJarvis | None | Yes, with agents |
| Setup wizard, settings without editing JSON | isair/jarvis | `.env` file and CLI | Yes, phase 1 |
| Memory viewer, knowledge shown to the owner | isair/jarvis | None | Yes, with memory (phase 4) |
| Redact sensitive data before it reaches the model or storage | isair/jarvis | Not done | Yes, with memory |
| Keyword fallback when embeddings are missing | isair/jarvis | None | Yes, as part of hybrid search |
| Tool routing: send the model only the relevant tools | isair/jarvis | None | Yes, with tools |
| Voice: local Whisper, local TTS, wake word, dictation hotkey | both | None | Phase 5 |
| MCP servers as the plug-in format for tools | both | None | Phase 6 |
| `evals/` directory next to the code | isair/jarvis | Already ahead: `python -m app eval` | Keep, and grow it |

Where Reyleight can be better than both: every answer is cited from application-controlled metadata, weak evidence produces an explicit refusal, retrieved text is never an instruction, and the whole pipeline can be proven offline. The interface should make those properties visible (cited sources panel, "insufficient data" state, privacy page), which the prototype already does.

## 2. Architecture

```
                 ┌─────────────────────── one UI codebase (React + TypeScript) ─────────────────────┐
                 │                                                                                   │
        Web (browser)             Windows .exe (Tauri 2)                 Phone (PWA first)
        served by backend         WebView2 + Python sidecar              talks to YOUR computer
                 │                          │                                     │
                 └──────────────┬───────────┴─────────────────────────────────────┘
                                │  HTTP + SSE, bearer token, /api/v1 (OpenAPI = the contract)
                       ┌────────▼─────────┐
                       │  FastAPI backend │  jobs, conversations, memory, tools, approvals, audit log
                       └───┬─────┬────┬───┘
                           │     │    └── Ollama (LLM, loopback only)
              SQLite (meta)│     └─────── Qdrant local (vectors) + SQLite FTS5 (keywords)
                           └───────────── allow-listed folders (originals read-only)
```

### Decisions

1. **One UI, three shells.** The React app in `frontend/` is the only UI. Web serves it as static files from the backend (same origin, so no CORS rules to loosen). Desktop wraps it. Phone reuses it. No second codebase to keep in sync.
2. **Desktop shell: Tauri 2, not Electron.** Installer around 10 MB instead of 150 MB+, uses the system WebView2, and has a capability system that denies the UI filesystem and network access by default, which matches "least privilege". Cost: a little Rust at the edges (start/stop the sidecar, tray icon, native folder picker). Electron is the fallback if Tauri blocks a needed feature.
3. **Backend stays Python, shipped as a sidecar.** Tauri starts the backend (built with PyInstaller or Nuitka) on `127.0.0.1` with a random port and a per-launch token, and stops it on exit. Rewriting the RAG stack in Rust is not worth it.
4. **Phone is a remote control for your own computer, not a second brain.** A phone cannot run Ollama plus Qdrant plus a 4B model comfortably, and copying the notes onto a phone widens the attack surface. So the phone app talks to the desktop backend over a private network you own (Tailscale or WireGuard), after a QR-code pairing. No public relay, no cloud account. Fully on-device phone inference is an experiment for later, not the plan.
5. **The API schema is the contract.** FastAPI already emits OpenAPI. Generate the TypeScript types and client from it (`openapi-typescript`) instead of hand-writing them as the prototype does today, so a backend change breaks the build instead of the user.
6. **Streaming and jobs over SSE.** Server-sent events for token streaming and for ingestion progress. Simpler than WebSockets, works through the Tauri webview and mobile browsers.
7. **No external requests from the UI, ever.** Fonts and scripts are bundled, and the page has a Content-Security-Policy (`connect-src 'self'`). The prototype already does this; keep it as a CI check.

### Security work the new clients force

These change `security.md`, so each needs an explicit decision from you:

- **Authentication.** Today the API trusts anything on loopback. A malicious web page can try to reach `127.0.0.1` (DNS rebinding). Add a per-launch bearer token, a `Host` header check and an `Origin` allow-list before a UI or phone is allowed to connect.
- **Remote access for the phone.** `security.md` says never bind `0.0.0.0`. Phone access needs a deliberate exception: bind only to the private-network interface, require the pairing token, and make it an opt-in switch that is off by default and visible in the UI.
- **Approval policy before tools.** Adopt the safe / monitor / ask / block levels. Default for anything that writes, deletes, sends or runs code is `ask`; destructive is `block`.
- **Audit log.** An append-only local record of tool calls and approvals, viewable in the UI, containing no note or question text (matches the logging rule).
- **Windows trust.** Smart App Control already blocks some unsigned files on this machine. An unsigned `.exe` and unsigned bundled `.pyd` files will be blocked for other users too. Plan for code signing (Azure Trusted Signing or a certificate) before any distribution.

## 3. Backend gaps the UI needs

The prototype uses only `GET /health`, `GET /ingestion/file-types`, `POST /search` and `POST /ask`. A real product needs these slices, each small and tested like the MVP:

| Slice | What | Why the UI needs it |
| --- | --- | --- |
| Status | `GET /system/status`: LLM reachable, model names, embedding model, document and chunk counts, index age | Replace guesses with facts in the System panel |
| Library | List, inspect, re-index, remove documents; manage the folder allow-list | Library manager screen |
| Ingestion jobs | Start ingest/index as a job, progress over SSE, cancel | Progress bars instead of CLI |
| Streaming ask | `POST /ask/stream` (SSE): tokens, then sources and the grounded flag | Feels alive; allows cancel |
| Conversations | Tables for conversations and messages, follow-up questions that use history | Chat memory and history sidebar |
| Auth | Token, Host/Origin checks | Section 2 |
| Settings | Read and change safe settings (model, top-k, threshold) with validation | Settings screen, setup wizard |
| Feedback | Thumbs and "save as test case" | Grows the evaluation set from real use |
| Serve UI | FastAPI serves `frontend/dist` | Web and desktop share one origin |

## 4. Making the RAG better

Today: heading/size chunks, one dense embedding (all-MiniLM-L6-v2), Qdrant, top-k, a minimum-score gate, a 4B local model, 37 evaluation questions. Every change below must be accepted or rejected by the evaluation set, not by feel. Order is value for effort.

1. **Grow the evaluation first.** 150+ questions: multi-hop, dates, numbers, near-duplicates, unanswerable, prompt-injection notes. Report recall@k, MRR, refusal precision/recall, citation precision. Run it in CI. Without this, every item below is a guess.
2. **Hybrid search.** Add SQLite FTS5 (BM25) next to the vectors and merge with reciprocal rank fusion. Fixes names, ids, codes and rare words that embeddings blur, and doubles as the no-embeddings fallback.
3. **A stronger embedding model.** Candidates: bge-m3, nomic-embed-text, Qwen3-Embedding. The `indexed_model` column already tracks which model built each document, so a re-index flow is mostly a UI job. Compare on your own evaluation set, not on leaderboards.
4. **Reranking.** Retrieve 30, rerank with a local cross-encoder (bge-reranker), keep 5. Usually the biggest precision gain after hybrid search.
5. **Better chunks.** Prepend document title and heading path to each chunk before embedding; retrieve small chunks but show the parent section (parent-child); row-level chunks for CSV and tables; keep provenance exact so the UI can jump to the line or page.
6. **Query understanding.** Rewrite follow-ups using the conversation ("and the return flight?"); optional multi-query or decomposition for multi-hop questions. Needs conversations (section 3).
7. **More sources.** PDF (page-level provenance), DOCX, images via local OCR, email exports, code. Each type gets a parser, tests with a made-up fixture, and a size cap.
8. **Live sync.** File watcher with hash diffing so edits, moves and deletions update the index; stale citations are the quickest way to lose trust.
9. **Metadata and filters.** Folder, tag, date and type facets in the UI; time-aware questions ("last month").
10. **Verified citations.** After generation, check each cited claim against its note (quote match or a small entailment check) and drop or flag unsupported claims. Strengthens the existing "invented citations are discarded" rule.
11. **Retrieval inspector in the UI.** For any answer, show candidate chunks with dense score, keyword score and rerank score, and why each was kept or cut. This is how you learn what to tune, and it is a feature no reference project has.
12. **Memory as a second, separate collection.** Facts and preferences you approve, each with a source and date, editable and deletable in a memory viewer; redact secrets before storing; always labelled as "memory" versus "note" in citations. Never silently mix.
13. **Agentic retrieval and graph RAG, last.** Iterative retrieve-then-reason loops with a step budget, and a knowledge graph, only if the multi-hop evaluation shows a gap that steps 2-6 cannot close. They cost latency and complexity on a 4B local model.

## 5. Phased roadmap

Same rule as before: one small tested slice at a time, each confirmed before work starts.

| Phase | Outcome | Main slices | Done when |
| --- | --- | --- | --- |
| 0 | Interface prototype (done) | React app: boot, console, sources, search, modules, privacy, themes, demo mode | It runs against the real health and search endpoints and works on a phone-sized screen |
| 1 | The UI is real | Status, library, ingestion jobs, streaming ask, conversations, auth, settings, serve UI, setup wizard, generated API client | You can install nothing but Python and Ollama, add a folder in the UI, and ask a question end to end |
| 2 | Retrieval you can trust | Bigger evaluation, hybrid search, reranker, new embeddings, chunk upgrades, retrieval inspector | Measured improvement on the evaluation set; no regression in refusals |
| 3 | Windows `.exe` | Tauri shell, sidecar packaging, installer, tray, native folder picker, auto-update, code signing | A clean Windows machine installs and runs it with the network off |
| 4 | Memory | Conversation summaries, approved facts, memory viewer, redaction | You can see, edit and delete everything it remembers |
| 5 | Voice | Push-to-talk, local Whisper (faster-whisper), local TTS (Piper); wake word later | Dictate a question and hear a cited answer, fully offline |
| 6 | Tools and agents | Approval policy, audit log, MCP client, a first read-only tool, then bounded agents | Every action is approved or blocked visibly, and the log proves it |
| 7 | Phone | Pairing, private-network access, PWA install, then a Tauri mobile shell if the PWA is not enough | Ask your notes from your phone with no cloud account involved |
| 8 | Connectors | GitHub, calendar, mail, as read-only tools first | Each connector has the narrowest scope and an off switch |

Phone and desktop can swap order if you want the phone sooner: the responsive UI already exists, so only auth and remote access (section 2) stand in the way.

## 6. Design system (for the person extending the UI)

- **Look:** deep blue-black background with a drifting grid, cyan accent, gold secondary, red for danger or offline. Alternate "Mark III" theme swaps cyan for gold and red. Chamfered panels with corner brackets, scanlines, a sweeping scan bar, a glitch title, a spinning arc reactor that speeds up while the model works and turns red with a flicker when the backend is unreachable.
- **Tokens:** all colours, fonts and the chamfer size live in `frontend/src/styles/tokens.css`. A new theme is one block of variables.
- **Motion rules:** decoration only, never required to understand state. Reduced motion (the system setting or the in-app switch) stops every animation.
- **Honest states:** the interface shows only what it knows. Backend offline, demo data, "insufficient data" and "unverified" are all distinct, labelled states. Planned features appear as locked modules, not as buttons that do nothing.
- **Accessibility:** keyboard operable, visible focus, labelled controls, live region for new messages, text contrast checked against the dark background.
- **Layout:** three columns on desktop (navigation, work area, system panel), two on tablet, one column with a bottom tab bar on phones.

## 7. Decisions I need from you

1. Tauri 2 (recommended) or Electron for the `.exe`?
2. Phone via your own private network (recommended) or an eventual on-device model?
3. Are you willing to pay for code signing when you distribute the `.exe`? Without it, Windows will warn or block.
4. Phase order: library and streaming first (recommended), or retrieval upgrades first?
