# Requirements (MVP)

The MVP is a working, evaluated, fully offline RAG loop over the owner's own notes and documents.

## Functional requirements

1. Index only folders on an explicit allow-list.
2. Ingest TXT and Markdown first. PDF and DOCX are added once that pipeline is proven.
3. Split documents into chunks that keep provenance: source file, page or heading, chunk id.
4. Embed chunks locally behind a swappable `EmbeddingProvider` interface.
5. Store vectors in a local Qdrant instance behind a `VectorStore` interface.
6. Retrieve by similarity with metadata filters and a configurable top-k.
7. Generate answers with a local LLM (Ollama) behind an `LLMProvider` interface.
8. Every answer cites its sources. When evidence is weak, the answer is an explicit "I don't have enough information."

## Quality requirements

- An evaluation set measures retrieval and answer quality, including unanswerable questions.
- The whole pipeline works with the network disconnected, once models are installed.
- Each step ships with tests, lint and type checks.

## Acceptance criteria per step

See [roadmap.md](roadmap.md). Each step states its own "done when" before work starts.

## Out of scope

Long-term memory, agent/tool framework, external integrations, browser or computer use, multi-agent orchestration, a full frontend, packaging.
