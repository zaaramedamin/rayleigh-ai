# Reyleight

A local-first, privacy-first personal AI knowledge assistant. It ingests your own notes and documents, answers questions with source citations, and says so when it doesn't have enough evidence.

Everything runs on your machine. No document or query text is sent to a cloud API.

## Status

Early development. Phases 0–3 of the [roadmap](docs/roadmap.md) are done: API skeleton, database and safe file storage, ingestion of text-based files, chunking with provenance, local embeddings, the vector index, semantic search, and cited answers from a local LLM. Phase 4 is done too: a built-in evaluation set measures quality, and an offline check proves nothing leaves your computer. All 12 steps of the MVP roadmap are complete.

## Docs

- [How it works](docs/how-it-works.md): what happens to your notes, step by step
- [Evaluation](docs/evaluation.md): how quality is measured, and the results
- [Offline check](docs/offline-check.md): proving nothing leaves your computer
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
pip install -r requirements.lock  # the exact, tested versions of every dependency
pip install --no-deps -e .         # the project itself

alembic upgrade head              # create/update the SQLite database in DATA_DIR
python -m app download-model      # one time: download the embedding model into MODELS_DIR
python -m app ingest              # ingest supported files from ALLOWED_FOLDERS, and chunk them

uvicorn app.main:app --reload     # API on http://127.0.0.1:8000 (docs at /docs)
pytest                            # tests
ruff check . ; ruff format --check .
mypy                              # strict type checking of app/ (configured in pyproject.toml)
```

Set `ALLOWED_FOLDERS` (comma-separated) and `MAX_FILE_SIZE_MB` in `.env`. Only those folders are ever read.

### Dependencies and the lock file

`backend/pyproject.toml` lists what the project needs; `backend/requirements.lock` pins the exact version of everything that gets installed, so a rebuild on another day gives the same result. It was generated for Python 3.13 on Windows and records the versions the whole test suite was run against.

To change a version: install it in your working environment, run `pytest`, `ruff check .` and `mypy`, and only then record it:

```powershell
pip install pip-tools                      # a tool, not a project dependency
cd backend
pip freeze --exclude-editable > tested.txt # the versions you just tested (delete pip and setuptools lines)
pip-compile pyproject.toml --extra dev --strip-extras --no-emit-index-url --constraint tested.txt --output-file requirements.lock
```

The `--constraint` matters: without it, `pip-compile` picks the newest version of everything, which is not what you tested. Remove the `-c tested.txt` lines `pip-compile` adds, so no local path ends up in the file, and delete `tested.txt`. The lock pins versions but not file hashes: hashing every platform's wheels means downloading gigabytes of PyTorch builds that are never used.

**Windows Smart App Control** can block the compiled files of a package version it has not seen before ("An Application Control policy has blocked this file"). This happened for `scikit-learn` (a placeholder is used, see Troubleshooting), for mypy's native parser (`native_parser = false` in `pyproject.toml`), and for a newer SQLAlchemy than the one that was tested. If a rebuild is blocked, installing the exact locked versions is the fix, and re-running while online sometimes clears it.

### Commands

`python -m app <command>` (add `--help` to any command for details):

| Command | What it does |
|---|---|
| `ingest` | Ingest, chunk and index files from `ALLOWED_FOLDERS` |
| `rechunk` | Rebuild all chunks after changing chunk settings (then re-indexes them) |
| `index` | Embed documents that are not searchable yet; `--rebuild` re-embeds everything |
| `status` | Show documents, chunks, and how many are searchable |
| `search <question>` | Show the most relevant chunks, with scores and sources (`--top-k`, `--type .md`, `--document ID`) |
| `ask <question>` | Answer a question from your notes, with citations (`--top-k`, `--type`, `--document`) |
| `check-llm` | Send a test prompt to the local LLM (Ollama) |
| `eval` | Measure retrieval and answer quality on a built-in test set (`--answers`, `--chunk-size`, `--min-score`, `--output`) |
| `offline-check` | Run the whole pipeline with all non-local network access blocked, and report any attempt |
| `encrypt-library` | Encrypt the library: note text, headings, file names and stored files (makes a safety backup first) |
| `security` | Show whether the library is encrypted and how it unlocks |
| `recovery-check` | Confirm that your recovery passphrase unlocks the library |
| `change-passphrase` | Set a new recovery passphrase |
| `doctor` | Check the whole setup and say how to fix each problem (`--quick`, `--fix`) |
| `backup` | Save the library (database and stored files) to one verified archive |
| `restore` | Restore a backup archive into an empty folder, after verifying every checksum |
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

### Asking questions

```powershell
python -m app ask how long should I simmer oats in milk
```

```
You should simmer rolled oats in milk for five minutes [1].

Sources:
  [1] oats.md > Oats > Cooking, lines 5-7  (id 1:1)
```

Or `POST /api/v1/ask` with `{"question": "..."}`. The response has `answer`, `grounded`, `reason`, and `sources` (file, heading, lines, score and the cited text).

How an answer is made, and why you can trust the sources:

1. **Relevance gate.** Only notes scoring at least `ANSWER_MIN_SCORE` (default 0.30) are used. If none do, you get "I don't have enough information in your notes to answer that." and the model is never called.
2. **Notes are data.** Notes are given to the model as numbered notes inside delimiters that contain a random value, new for every question, so a note cannot fake the end of a note. The model is told to ignore any instructions found inside notes.
3. **The model can decline.** If the notes are related but don't contain the answer, the model replies `INSUFFICIENT` and you get the same refusal.
4. **Citations are checked by the app.** The model cites notes by number, and the app maps numbers to real sources from the database. Numbers the model invents are removed. If an answer has no valid citation, it is not shown.

`reason` is one of `answered`, `no_relevant_notes`, `model_declined`, `no_valid_citation`. `ANSWER_MIN_SCORE` was checked with the evaluation set (see [Measuring quality](#measuring-quality)). A 4B-parameter local model can still make mistakes, so use the citations to check the answer against your note.

### Measuring quality

```powershell
python -m app eval --answers
```

Runs 37 questions with known answers (some answerable, some not, one with a hidden instruction) against 12 made-up notes, in a throwaway library that never touches your data. Measured on 2026-10-03: the right note ranked first for 27 of 27 questions, 24 of 26 answerable questions were answered correctly with a valid citation, 10 of 10 unanswerable ones were refused, and the hidden instruction was ignored. The two wrong answers were mistakes by the small model. Details, how to read the report and how to add your own questions: [docs/evaluation.md](docs/evaluation.md).

To test **your own questions about your real notes**, make a folder with the same layout as `backend/eval/` (a `corpus/` folder of notes and a `questions.json`) and point `eval` at it:

```powershell
python -m app eval --set C:\path\to\my-eval-set --answers
```

Keep that folder **outside the repository**, or in `eval-private/`, which git ignores: it contains your real notes. The evaluation copies the folder into a temporary library, never reads outside it, and never changes your own library.

### Proving it works offline

```powershell
python -m app offline-check             # whole pipeline, with all non-local network blocked
python -m app --offline ask <question>  # any command, same guard, on your own notes
```

Both report how many outbound connection attempts were blocked (it should be 0). See [docs/offline-check.md](docs/offline-check.md), which also gives the stronger check with the network physically switched off.

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

Search always returns the closest chunks, even when none is really relevant. A low score means a weak match, but a high score does not prove the answer is there: in the evaluation, questions on a related topic that the notes cannot answer scored as high as real answers. That is why `ask` also lets the model decline. The query text is never logged.

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

## Encrypting your library

By default your library is stored as plain data in `data/`. To encrypt it:

```powershell
python -m app encrypt-library      # asks for a recovery passphrase, backs up first, then encrypts
python -m app recovery-check       # confirm the passphrase you wrote down really works
python -m app doctor               # everything should say OK, including "encryption"
```

Stop the API server and any other Reyleight program before encrypting, and start the server again afterwards.

**What is encrypted** (AES-256-GCM, through Windows' own cryptography, so there is nothing extra to install): the text of your notes, their headings, their file names, the stored copies of your files, and therefore backups, which contain only those encrypted forms. Every value is authenticated: if anyone changes a single value, reading it fails instead of returning altered text.

**What is not encrypted:** the search vectors in `data/qdrant/` (search needs them readable; they are numbers, but they do reveal what your notes are about), and structure such as sizes, line numbers, counts and the content hash of each file. Turn on **BitLocker** for the drive to protect those too.

**How it unlocks.** One random key encrypts everything. It is stored in `data/security.json` in two protected forms:
- **by your Windows account**: while you are signed in, everything unlocks automatically and you never type anything;
- **by your recovery passphrase** (either one you choose, or a random key the tool generates): used on another computer, or if Windows is reinstalled or the account is lost. If you lose both the Windows account and the passphrase, the data **cannot be recovered**, by anyone.

When Windows cannot unlock the library (for example a restored backup on a new PC), the first command asks for the recovery passphrase, then remembers it for that Windows account. In unattended runs, set the environment variable `REYLEIGHT_PASSPHRASE` instead. `change-passphrase` replaces the passphrase without re-encrypting any data (older backups still need the passphrase they were made with).

**Honest limits.**
- This protects the files on disk (a stolen laptop, a copied folder, a leaked backup). It does not protect against a program running as you while you are signed in, because that program can use the same Windows unlock as Reyleight does.
- Anything written to disk in plaintext *before* you encrypted may survive in unused disk space. `encrypt-library` removes the old plaintext from the database file and replaces the stored files, but the file system may keep old blocks. To overwrite them, run `cipher /w:<your data folder>` (a Windows built-in; it takes a while) or rely on BitLocker. The safety backup made before encrypting is the unencrypted original: delete it once everything works.
- Values are tied to their column (or file name), so a value moved to another place fails to decrypt, but swapping whole *rows* of the same column is not detected.
- Encryption uses Windows' cryptography, so it is available on Windows only.
- A note whose first characters happen to be `reyleight-enc-v1:` would be mistaken for encrypted data in a library that is not encrypted. That is vanishingly unlikely, and it cannot happen in an encrypted one.

## Checking and protecting your library

```powershell
python -m app doctor            # is everything healthy? read-only; says how to fix each problem
python -m app backup            # save the library to one archive (see below for where)
python -m app restore FILE --to C:\restored-library
```

**`doctor`** checks the database version, that every stored file exists and matches its checksum, that the search index agrees with the database, the allowed folders, free disk space, the embedding model, Ollama, and Windows Smart App Control. It exits with code 1 only if something has failed. It changes nothing; `--fix` does one safe thing (marks documents for re-indexing when the search index disagrees with the database) and never deletes anything. `--quick` skips verifying checksums, which is faster for large libraries.

**`backup`** writes one `.zip` containing the database (copied with SQLite's own backup method, so it is consistent even while the library is in use), the stored copies of your files, and a manifest with a SHA-256 checksum of every entry. By default it goes to `%LOCALAPPDATA%\Reyleight\backups` (outside the repository and not synced); choose another place with `--to FILE`. It never overwrites a file and refuses to save inside the data folder it protects. The search vectors are not included, because `python -m app index` rebuilds them. **A backup contains your private notes**, so keep it somewhere safe.

**`restore`** verifies every checksum before writing anything, refuses an archive that has unexpected or unsafe entries, refuses to restore into a folder that is not empty, never touches your current library, and marks everything as "not searchable yet". To use the restored library, set `DATA_DIR` to that folder and run `python -m app index`.

## Troubleshooting

- **"blocked by an Application Control policy" (Windows).** Smart App Control can block newly installed compiled library files. Reyleight works around the one that matters here: if scikit-learn cannot be loaded, it is replaced by a placeholder, because only unused helper functions of sentence-transformers need it (a warning is logged, and embeddings are unaffected). If a different library file is blocked, the command stops with a one-line error; re-running while online sometimes clears it. Reyleight never changes Windows security settings.
- **"ollama is not reachable".** Open the Ollama app or run `ollama serve`, then retry. `python -m app status` shows its state.
- **"search index is in use by another process".** Only one program can open the vector index at a time. Stop the API server or wait for the running command.

## Configuration

Copy `.env.example` to `.env` and adjust. `.env`, `data/` and `models/` are gitignored.
