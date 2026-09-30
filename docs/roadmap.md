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
11. **Evaluation set.** Answerable and unanswerable questions with a scoring script.
12. **Offline verification.** Full pipeline works with the network disconnected.

## Future direction (not started; only on explicit request)

- PDF and DOCX ingestion
- React + TypeScript frontend
- Long-term memory
- Tool/agent framework
- Connectors (GitHub, Google, databases)
- Browser automation and computer use
- Multi-agent orchestration
- Packaging for distribution
