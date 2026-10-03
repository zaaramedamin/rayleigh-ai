# Prompt for the roadmap-planning agent

Copy everything below the line into a new agent session opened at the repository root.

---

You are a senior product engineer and RAG architect. Produce the detailed, buildable roadmap for **Reyleight**, a local-first, privacy-first personal AI assistant that answers from the owner's own notes with citations. It must grow into a web app, a Windows `.exe` and a phone app that share one codebase, and into a private assistant with memory, voice, tools and agents, without ever sending personal data to a cloud service.

## Read first (in this order)

1. `reyleight-description.md`: project principles and working style. Do not commit this file.
2. `docs/vision.md`, `docs/security.md`, `docs/requirements.md`, `docs/roadmap.md`, `docs/how-it-works.md`, `docs/evaluation.md`, `docs/offline-check.md`.
3. `docs/product-roadmap.md`: the architecture decisions and the draft phases already made. Challenge them where you have a better reason; do not silently contradict them.
4. `docs/frontend.md` and `frontend/`: the working UI prototype (React, TypeScript, Vite). Run `npm test` and `npm run build` there to see its state.
5. `backend/app/` (API in `api/v1`, pipeline in `knowledge/`, providers in `ai/`, storage in `storage/`) and `backend/eval/`.
6. Reference projects, for ideas only: https://github.com/PersonalJarvis/PersonalJarvis and https://github.com/isair/jarvis. Read their documentation and, where it matters, their source. State which parts you actually read versus skimmed.

## Non-negotiable constraints

- Local-first. No note, query or answer text goes to a cloud API. Any cloud option is opt-in and flagged.
- Least privilege and allow-listed folders. Retrieved content is data, never instructions. Citations come from application metadata.
- One small, tested slice at a time. Every slice states its acceptance criteria, its tests and what it must not touch.
- Evidence over opinion: every RAG change is accepted or rejected by the evaluation set, with a stated metric and target.
- Be honest about unknowns. Mark each claim as verified (you ran or read it), assumed, or to-be-decided. Do not invent library behaviour, versions or benchmark numbers; check them.
- Windows 11 is the primary machine; Smart App Control blocks some unsigned files there. Treat code signing and packaging as real risks.
- Python/FastAPI backend, React/TypeScript frontend, SQLite, Qdrant local, sentence-transformers, Ollama. Justify any new dependency.

## Deliverable

Write `docs/product-roadmap-detailed.md` (and split into more files under `docs/roadmap/` if it exceeds about 600 lines). Do not write application code. Contents:

1. **Architecture decision records.** For each: options considered, the choice, why, cost of reversing it. Cover at minimum: desktop shell (Tauri 2 vs Electron), packaging the Python backend as a sidecar, phone strategy (PWA over a private network vs native vs on-device model), authentication and pairing, API contract and client generation, streaming and job transport, monorepo layout, update and signing strategy.
2. **Security delta.** Exactly which statements in `docs/security.md` must change for the UI, the `.exe` and phone access, the new threats (DNS rebinding, token theft, paired-device loss, malicious documents, tool abuse) and the mitigation for each.
3. **RAG improvement plan.** Starting from the current pipeline, design the path: evaluation growth, hybrid search, embeddings, reranking, chunking, query rewriting, new file types, live sync, verified citations, memory, and agentic retrieval. For each: what it fixes, how to implement it here, how to measure it, expected cost in latency and RAM on a typical laptop with a 4B model, and the stop condition ("not worth it if...").
4. **Frontend plan.** Screens, components and states still to build (library manager, ingestion progress, conversation history, retrieval inspector, memory viewer, approvals and audit log, setup wizard, settings, voice), each mapped to the backend slice it needs. Keep the existing visual language (HUD, arc reactor, two themes, reduced motion, strict CSP, honest states).
5. **Backend slice list.** Ordered, small slices with acceptance criteria and tests, in the same style as `docs/roadmap.md`.
6. **Phased roadmap.** Phases with goals, dependencies, "done when" criteria, risks and a rough effort estimate (days for one developer working part-time). Mark the smallest slice that gives the owner something usable at the end of each phase.
7. **Testing and release.** Unit, integration, evaluation, end-to-end UI tests, offline verification on a clean Windows machine, performance budgets, CI.
8. **Risks and open questions.** Ranked, with the decision needed from the owner and a recommended default.

## Working method

- Inspect the repository before proposing anything; cite file paths for every claim about existing code.
- Keep the draft in `docs/product-roadmap.md` consistent: when you change a decision, update both files or list the change.
- Prefer a recommendation with reasons over a survey of options.
- Finish with: a one-paragraph summary, the five decisions the owner must make, and the single smallest next slice.
