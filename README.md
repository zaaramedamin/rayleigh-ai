# Reyleight

A local-first, privacy-first personal AI knowledge assistant. It ingests your own notes and documents, answers questions with source citations, and says so when it doesn't have enough evidence.

Everything runs on your machine. No document or query text is sent to a cloud API.

## Status

Early development. Phases 0–2 of the [roadmap](docs/roadmap.md) are done: API skeleton, database and safe file storage, ingestion of text-based files, chunking with provenance, local embeddings, the vector index, and semantic search with sources. Answering questions with a local LLM (Phase 3) is not built yet.

## Docs

- [Vision](docs/vision.md): what this is and why
- [Requirements](docs/requirements.md): MVP scope and acceptance criteria
- [Security](docs/security.md): privacy and safety principles
- [Roadmap](docs/roadmap.md): step-by-step plan and future direction

## Planned stack

Python, FastAPI, SQLAlchemy/Alembic, SQLite, Qdrant (local), sentence-transformers, Ollama. React + TypeScript later.

## Running locally

Requires Python 3.11+. From the repo root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

alembic upgrade head              # create/update the SQLite database in DATA_DIR
python -m app download-model      # one time: download the embedding model into MODELS_DIR
python -m app ingest              # ingest supported files from ALLOWED_FOLDERS, and chunk them

uvicorn app.main:app --reload     # API on http://127.0.0.1:8000 (docs at /docs)
pytest                            # tests
ruff check . ; ruff format --check .
```

Set `ALLOWED_FOLDERS` (comma-separated) and `MAX_FILE_SIZE_MB` in `.env`. Only those folders are ever read.

### Commands

`python -m app <command>` (add `--help` to any command for details):

| Command | What it does |
|---|---|
| `ingest` | Ingest, chunk and index files from `ALLOWED_FOLDERS` |
| `rechunk` | Rebuild all chunks after changing chunk settings (then re-indexes them) |
| `index` | Embed documents that are not searchable yet; `--rebuild` re-embeds everything |
| `status` | Show documents, chunks, and how many are searchable |
| `search <question>` | Show the most relevant chunks, with scores and sources (`--top-k`, `--type .md`, `--document ID`) |
| `check-llm` | Send a test prompt to the local LLM (Ollama) |
| `types` | List supported file types |
| `download-model` | Download the embedding model (the only command that uses the internet) |

### Search index

Each chunk is embedded (turned into a vector) and stored in [Qdrant](https://qdrant.tech/), which runs embedded inside the app and saves to `DATA_DIR/qdrant`. There is no server or Docker to run.

- `ingest` and `rechunk` index new or changed documents automatically when the model is downloaded; otherwise they say what to run.
- Each embedding model gets its own collection, so vectors from different models never mix. After changing `EMBEDDING_MODEL`, run `download-model`, then `index`.
- The vector store only holds ids and filterable metadata. Chunk text and citations always come from the SQLite database.
- Only one process can open the vector store at a time. If a command says it is in use, stop the API server or wait for the other command.

### Local LLM (Ollama)

Answers are written by a model running in [Ollama](https://ollama.com) on this machine. Set `LLM_MODEL` (default `qwen3.5:4b`) to any model you have pulled with `ollama pull`.

- `python -m app check-llm` sends a test prompt; `python -m app status` shows whether Ollama is running and the model is installed.
- `OLLAMA_URL` must point at this machine (`localhost`, `127.0.0.1` or `::1`). Any other address is rejected at startup, so prompts built from your notes can never be sent to another computer.
- Requests ignore system proxy settings and refuse redirects. The prompt and the model's answer are never logged.
- Reasoning ("thinking") mode is off by default because it is far slower; set `LLM_THINK=true` to turn it on.

### Searching

```powershell
python -m app search how do I cook porridge
python -m app search train times --top-k 3 --type md
```

Or through the API (start `uvicorn app.main:app`, then try it at http://127.0.0.1:8000/docs):

```
POST /api/v1/search
{"query": "how do I cook porridge", "top_k": 3, "file_types": [".md"], "document_ids": [1, 2]}
```

Only `query` is required. `top_k` defaults to `RETRIEVAL_TOP_K` (5), up to 50. Each result has a `score` (cosine similarity, higher is closer), the chunk `text`, and its source: file name, heading path, line range, and citation id `<document>:<chunk>`.

Search always returns the closest chunks, even when none is really relevant. In a small test with the default model, relevant matches scored about 0.45–0.75 and unrelated ones below about 0.35. The answering step will use a score threshold, measured by the evaluation step, to say "I don't have enough information" instead of guessing. The query text is never logged.

### Embedding model

Embeddings run locally with [sentence-transformers](https://www.sbert.net/). The model is set by `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`, about 90 MB, English). `download-model` saves it under `MODELS_DIR`. After that, the model is always loaded from that folder with the Hugging Face libraries forced offline, and code shipped with models is never run.

For notes in French or several languages, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is a multilingual alternative (about 470 MB). Change `EMBEDDING_MODEL`, then run `download-model` again.

### Supported file types

Run `python -m app.knowledge.ingestion --list-types`, or open `GET /api/v1/ingestion/file-types` in the API docs page. The list lives in `backend/app/knowledge/ingestion/file_types.py`.

| Type | Extensions | Notes |
|---|---|---|
| Plain text | `.txt` `.log` | UTF-8 |
| Markdown | `.md` `.markdown` | Headings kept for chunking |
| reStructuredText | `.rst` | Read as text |
| CSV / TSV | `.csv` `.tsv` | Read as text |
| JSON | `.json` | Must be valid JSON |
| YAML | `.yaml` `.yml` | Read as text |
| HTML | `.html` `.htm` | Visible text only; scripts and styles dropped |

Not supported yet: PDF, DOCX and other Office files, images, audio, video. Everything else is skipped and counted by extension in the ingestion summary.

### Chunking

Each stored document is split into chunks, so answers can later cite an exact location. Every chunk records its document, index, heading path (Markdown only, e.g. `Guide > Setup`), and start/end line.

- A chunk's citation id is `<document_id>:<chunk_index>`, generated by the app.
- Markdown is split at headings; chunks never cross a heading and never overlap across one. Code blocks are never split.
- Plain text is split at paragraphs. A paragraph or line that is too long is split further.
- Line numbers refer to the extracted text. They match the original file for text and Markdown, but not for HTML, where tags are removed.
- Overlap is made of whole lines, so it can be smaller than `CHUNK_OVERLAP_CHARS`, or absent when lines are long.
- Size and overlap (characters) are `CHUNK_SIZE_CHARS` and `CHUNK_OVERLAP_CHARS` in `.env`. After changing them, run `--rechunk`.

## Configuration

Copy `.env.example` to `.env` and adjust. `.env`, `data/` and `models/` are gitignored.
