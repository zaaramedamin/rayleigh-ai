# Roadmap

Work proceeds one small, tested slice at a time. Before each step, its slice and "done when" criteria are confirmed.

## MVP

### Phase 0: Foundation
1. **Repo skeleton and docs.** Docs, `.gitignore`, `.env.example`, README. Done when the secret/data/model paths are verifiably ignored and docs match the plan.
2. **FastAPI skeleton.** `/health`, Pydantic settings, structured logging, Ruff and pytest. Done when the server starts, `/health` returns 200 and a test passes.

### Phase 1: Storage and ingestion
3. **Database and safe file storage.** SQLite, SQLAlchemy, Alembic, `Document` model, hashed storage paths. Done when a malicious filename cannot escape `data/`.
4. **TXT/Markdown ingestion.** Allow-listed folders, idempotent ingestion. Done when re-ingesting creates no duplicates and outside files are rejected.
5. **Chunking with provenance.** Heading/size chunks with overlap, each carrying source, location and chunk id.

### Phase 2: Retrieval
6. **Local embeddings.** `EmbeddingProvider` with sentence-transformers.
7. **Qdrant vector store.** `VectorStore` interface, local Qdrant.
8. **Retrieval.** Query embedding, similarity search, filters, configurable top-k.

### Phase 3: Generation
9. **Local LLM via Ollama.** `LLMProvider` interface with explicit error handling.
10. **End-to-end RAG.** Grounded answers with citations, an explicit refusal on weak evidence, and injection-resistant prompting.

### Phase 4: Proof
11. **Evaluation set.** Answerable and unanswerable questions with a scoring script. **Done:** 37 questions over 12 notes, `python -m app eval`; see [evaluation.md](evaluation.md).
12. **Offline verification.** Full pipeline works with the network disconnected. **Done:** `python -m app offline-check` and `--offline`, plus the manual steps in [offline-check.md](offline-check.md).

## Future direction (not started; only on explicit request)

The detailed plan for everything below, with slices, acceptance criteria and risks, is in [full-assistant-guide.md](full-assistant-guide.md). The multi-platform architecture (web, Windows `.exe`, phone) and the interface prototype are in [product-roadmap.md](product-roadmap.md) and [frontend.md](frontend.md).

- PDF and DOCX ingestion
- React + TypeScript frontend
- Long-term memory
- Tool/agent framework
- Connectors (GitHub, Google, databases)
- Browser automation and computer use
- Multi-agent orchestration
- Packaging for distribution
