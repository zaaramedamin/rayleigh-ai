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
```

## Configuration

Copy `.env.example` to `.env` and adjust. `.env`, `data/` and `models/` are gitignored.
