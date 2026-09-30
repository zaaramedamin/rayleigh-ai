# Reyleight

A local-first, privacy-first personal AI knowledge assistant. It ingests your own notes and documents, answers questions with source citations, and says so when it doesn't have enough evidence.

Everything runs on your machine. No document or query text is sent to a cloud API.

## Status

Early development. Currently at Phase 0, Step 1: repo skeleton and docs. There is no runnable code yet.

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

uvicorn app.main:app --reload     # http://127.0.0.1:8000/api/v1/health
pytest                            # tests
ruff check . ; ruff format --check .
alembic upgrade head              # create/update the SQLite database in DATA_DIR
python -m app.knowledge.ingestion # ingest .txt/.md from ALLOWED_FOLDERS
```

Set `ALLOWED_FOLDERS` (comma-separated) and `MAX_FILE_SIZE_MB` in `.env`. Only those folders are ever read.

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

## Configuration

Copy `.env.example` to `.env` and adjust. `.env`, `data/` and `models/` are gitignored.
