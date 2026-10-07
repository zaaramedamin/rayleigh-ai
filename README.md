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

python -m app migrate              # create/update the SQLite database in DATA_DIR
python -m app download-model      # one time: download the embedding model into MODELS_DIR
python -m app download-voice-model  # one time, optional: the speech model for voice orders
python -m app ingest              # ingest supported files from ALLOWED_FOLDERS, and chunk them

python -m app serve               # the interface and API on http://127.0.0.1:8000 (docs at /docs); upgrades the database first (after a backup)
uvicorn app.main:app --reload     # the same API, restarting when code changes (for development)
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
| `ingest` (or `sync`) | Read your allowed folders (from `.env` and the interface), chunk and index new files, and notice edited and deleted ones |
| `feedback` | List your marks on answers, or turn the failures into evaluation questions (`feedback export`) |
| `prune` | Remove documents whose files are gone (shows them first, asks before deleting; `--dry-run`, `--yes`, `--superseded`) |
| `rechunk` | Rebuild all chunks after changing chunk settings (then re-indexes them) |
| `index` | Embed documents that are not searchable yet; `--rebuild` re-embeds everything |
| `serve` | Start the interface and API on this computer only (`--port`, `--no-migrate`); upgrades an out-of-date database first, after saving a backup of it |
| `migrate` | Create or upgrade the database (the same upgrade `serve` does) |
| `status` | Show documents, chunks, how many are searchable, and any missing or replaced documents |
| `search <question>` | Show the most relevant chunks, with scores and sources (`--top-k`, `--type .md`, `--document ID`) |
| `ask <question>` | Answer a question from your notes, with citations (`--top-k`, `--type`, `--document`) |
| `summarize <document>` | A short summary of one document, by number or file name |
| `compare <document> <document> ...` | What two to four documents have in common and where they differ, with sources |
| `extract <what you want>` | A table of the facts your notes hold that match a request, each with its source (`--top-k`, `--type`, `--document`, `--mode`) |
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
| `download-model` | Download the embedding model (uses the internet, once) |
| `download-voice-model` | Download the speech model (Whisper) for voice orders (uses the internet, once) |

### Keeping the library in step with your folders

Every `ingest` (or `sync`, the same command) records where each file was found, and from that it notices what changed:

- **An edited file** replaces its old version. The new text is searched; the old text is kept as history, never searched, and its vectors are removed. Changing a file back brings the old version back.
- **A renamed or copied file** is the same document at a second place, not a duplicate.
- **A deleted file** is marked *missing* only after it was absent from **two syncs in a row**, so an offline cloud folder, an unplugged drive or a moved folder does not look like a deletion. A folder that cannot be read is not judged at all, and if many files vanish from one folder at once the folder is not judged either (it is probably unavailable, and the sync says so). A missing document stops being searched, and comes back if the file returns.
- Nothing is deleted by a sync. `python -m app prune` lists the missing documents (add `--superseded` to include older versions of edited files), asks, and then removes the document, its text, its search entries and its stored copy, and compacts the database file so the removed text does not stay inside it. `--dry-run` only lists. Your original files are never touched.
- Removing a document in the interface also removes its older versions, so an edited note's earlier text does not linger.

Where each file was found (the folder and the path inside it) is stored in the database, encrypted with the rest of the library if you encrypted it. Older libraries kept these paths in plain text in `data/library.json`; the first update moves them into the database and empties that list.

### Search index

Each chunk is embedded (turned into a vector) and stored in [Qdrant](https://qdrant.tech/), which runs embedded inside the app and saves to `DATA_DIR/qdrant`. There is no server or Docker to run.

- `ingest` and `rechunk` index new or changed documents automatically when the model is downloaded; otherwise they say what to run.
- Indexing shows its progress on the terminal (chunks done, speed, time left) and in the interface. It saves after every batch of chunks, so an interrupted run (Ctrl+C, a closed window, a crash) continues at the first chunk without a vector instead of starting over. On this computer the model embeds about 36 chunks a second whatever the batch size, so a library of 15,000 chunks takes about 7 minutes the first time; storing the vectors is under a tenth of that.
- Each embedding model gets its own collection, so vectors from different models never mix. After changing `EMBEDDING_MODEL`, run `download-model`, then `index`.
- The vector store only holds ids and filterable metadata. Chunk text and citations always come from the SQLite database.
- Only one process can open the vector store at a time. A command that needs it while the server (or another command) holds it waits up to 20 seconds, says so, and then stops with a clear message: stop the server, or use the interface instead.

### Local LLM (Ollama)

Answers are written by a model running in [Ollama](https://ollama.com) on this machine. Set `LLM_MODEL` (default `qwen3.5:4b`) to any model you have pulled with `ollama pull`.

- `python -m app check-llm` sends a test prompt; `python -m app status` shows whether Ollama is running and the model is installed.
- `OLLAMA_URL` must point at this machine (`localhost`, `127.0.0.1` or `::1`). Any other address is rejected at startup, so prompts built from your notes can never be sent to another computer.
- Requests ignore system proxy settings and refuse redirects. The prompt and the model's answer are never logged.
- Reasoning ("thinking") mode is off by default because it is far slower; set `LLM_THINK=true` to turn it on.
- **The prompt always fits the model's memory.** The model reads about 8,000 tokens at a time, and Ollama silently drops the *start* of a prompt that is too long, which is where the instructions are. So Reyleight keeps every prompt inside that window itself, and decides what gives way first: the instructions and your latest message are never cut (a message too long to share the window with the instructions is refused with a sentence that says so), notes handed to the model have their own limit that always leaves room, and earlier turns of a conversation take what is left, oldest first. A conversation of hundreds of exchanges keeps working and the assistant still knows its name and how to address you. With `LLM_THINK=true` the model reserves more room for its reasoning, so the history gets shorter.

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

### Summaries, comparisons and tables

Three more things the assistant can do with your notes, from the command line or the API (`POST /api/v1/tasks/summarize`, `/compare` and `/extract`, behind the same access password as everything else):

```powershell
python -m app summarize lisbon-trip.md
python -m app compare invoices-2025.md invoices-2026.md
python -m app extract the serial number of each device
```

```
  router   SN-7F3K-9921  [1]
  printer  SN-2B8M-1146  [1]

Sources:
  [1] devices.md > Devices, lines 1-5  (id 7:0)
```

A document is given by its number or by its file name (`python -m app search` shows a document's number in the id of each result: id 7:0 is document 7).

In the interface, **Knowledge > Library > INSPECT** on a document has a **SUMMARIZE** button. The summary is shown with a reminder that it has no sources, and a long document says how much of it was read. Comparisons and tables are not in the interface yet; use the command line or the API.

- **summarize** reads one document from its own text, so it needs no search. A document too long for one reading is cut into parts, each part is summarized, and the partial summaries are combined; if it is longer than the first 8 parts (about 48,000 characters), the result says that only the start was read.
- **compare** reads two to four documents and writes what they have in common and where they differ. The model cites documents by number, and the application checks every citation: a comparison that cites no document is withheld, as an answer without a valid citation is.
- **extract** searches your notes like `ask`, then asks the model for the facts as a table of rows (what, the value, which note). The application reads the table strictly and drops every row that names a note it was not given or is not a short fact; the sources are built from the database. If no note is relevant enough, the model is not called.

All three work on the local model only, treat your notes as data and not as instructions, and are measured by `eval --answers` (see [Measuring quality](#measuring-quality)). Their instructions are kept in one place, `app/knowledge/answering/prompts.py`, each with a version and a fingerprint of its text: a test fails if a prompt changes without its version, and the evaluation report names the versions it ran with, so two reports can be compared knowing which prompts they used.

### General chat

`POST /api/v1/chat` with `{"message": "...", "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` talks to the local model directly. It answers from what the model knows and reads nothing from your library, so it works even when the library is empty or locked. The response has `answer`, `model`, and `truncated` (true when the reply hit the length limit). `history` is optional and holds the earlier turns, oldest first; only the recent ones that fit the model are used.

In the web interface this is how the chat starts. The **MY NOTES** switch next to the message box turns on answers from your notes, with sources, and the choice is remembered.

A general reply is not checked against anything and has no citations, so treat it as unverified: a small local model is often wrong about facts. Only answers from your notes are backed by sources.

### The assistant: identity, memory, actions and voice

General chat is also where the assistant lives. Run `python -m app migrate` once (it adds two small tables), then open **Profile**:

- **Your assistant.** Its name, how it addresses you ("sir" by default) and its role, in your own words. They go into its instructions in every conversation. You can switch off whether it may read the "About you" form and its memories.
- **Memory.** Short facts that last between sessions. Add them yourself, or say or type "remember that ..." and the assistant saves one (the chat shows `REMEMBERED: ...`). Every memory is listed there and can be deleted one by one or all at once. At most 200 are kept.
- **Actions.** The assistant can open pages, change the theme, switch sounds, voice and animations on or off, search your notes, start a library update, read out the system status, clear the conversation, lock the application and remember things. That is a fixed list: the model can only ask, the backend and the interface each check every request against the list, and every action taken is shown under the reply. It cannot delete documents or folders, change the password or reach anything outside the application, and those are not on the list at all.
- **Voice.** Press **VOICE** (or `POST /api/v1/voice/transcribe` with a 16 kHz WAV) and speak; it stops when you pause. Your voice is turned into text by a Whisper model on this computer (`SPEECH_MODEL`, default `openai/whisper-base`, about 290 MB; set `SPEECH_LANGUAGE` such as `en` to skip language detection). Replies are read aloud with a voice installed on this computer, and the assistant greets you when the application opens. All of it is in Settings > Voice. Esc stops it; hands-free mode listens again after each reply.

Spoken orders always go to the assistant, which searches your notes itself when you ask what they say. The small local model sometimes searches your notes for something it already knows about you, or hesitates when two memories disagree; delete the wrong one on the Profile page.

### Measuring quality

```powershell
python -m app eval --answers
```

Runs 37 questions with known answers (some answerable, some not, one with a hidden instruction) against 12 made-up notes, in a throwaway library that never touches your data. With `--answers` it also runs follow-up questions and five summarize, compare and extract jobs, and its report names the version of every prompt used. Measured on 2026-10-03: the right note ranked first for 27 of 27 questions, 24 of 26 answerable questions were answered correctly with a valid citation, 10 of 10 unanswerable ones were refused, and the hidden instruction was ignored. The two wrong answers were mistakes by the small model. Details, how to read the report and how to add your own questions: [docs/evaluation.md](docs/evaluation.md).

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

Or through the API (start `python -m app serve`, then try it at http://127.0.0.1:8000/docs):

```
POST /api/v1/search
{"query": "how do I cook porridge", "top_k": 3, "file_types": [".md"], "document_ids": [1, 2]}
```

Only `query` is required. `top_k` defaults to `RETRIEVAL_TOP_K` (5), up to 50. Each result has a `score` (cosine similarity, higher is closer), the chunk `text`, and its source: file name, heading path, line range (or pages, for a PDF), and citation id `<document>:<chunk>`.

**Three ways to search** (`mode` in the request, `--mode` on the command line, `SEARCH_MODE` as the default, which is `hybrid`):

| Mode | Finds notes | Good for |
|---|---|---|
| `vector` | by meaning | a question put in your own words |
| `keyword` | by the words in the question | exact codes, names and numbers (`INV-2026-0418`, a surname, an extension) |
| `hybrid` | both, merged | everything; the default |

Meaning search is good at topics and weak at exact strings: two invoices that differ only in a year look almost the same to it. The keyword side finds the exact code. In hybrid mode a chunk that holds most of the question's words (including the exact code) counts for much more than one that matches a single common word, which is ignored. Every result still carries its meaning similarity as `score`, and that, never the keyword strength, is what the relevance gate (`ANSWER_MIN_SCORE`) looks at, so a keyword match on a note about something else cannot make the assistant answer from it. Results found by their words also carry a `keyword_score`.

The keyword index is built **in memory** from your notes the first time it is needed (about a second for 15,000 chunks) and again after the library changes. It is never written to disk, because the words of an encrypted library must not sit in a plaintext index. Search by words needs SQLite's FTS5, which Python ships with on Windows; if it is missing, hybrid falls back to meaning search and `keyword` reports a clear error. Question words are matched whole (no stemming, accents ignored); a question's operators and quotes are just words.

`python -m app eval` measures all three modes on the same questions, and reports the exact-code questions as their own group. Measured with your embedding model on 2026-10-05 (34 questions whose answer is in the notes): meaning 97.1% right at rank 1, words alone 82.4%, hybrid 100%; on the four exact-code questions 75%, 100% and 100%.

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
| PDF | `.pdf` | Text of each page, cited by page ("report.pdf, page 12"). Running headers and page numbers are dropped (`PDF_KEEP_HEADERS_FOOTERS=true` keeps them). Scans have no text to read (OCR is not built), and PDFs that ask for a password are skipped |
| Word | `.docx` | Headings (kept for chunking), lists and tables. The old `.doc` format is not read |
| Excel | `.xlsx` | Each sheet is a section and each row is read as "column: value" pairs under the first row's headings (the first 5,000 rows per sheet). Formulas are not run: the value saved in the cell is read, so dates show as their serial numbers |
| PowerPoint | `.pptx` | One section per slide, in presentation order: title, text, tables and speaker notes |

Not supported: the old Office formats (`.doc` `.xls` `.ppt`), images and scans (no OCR), audio, video. Everything else is skipped and counted by extension in the ingestion summary.

**How the complicated formats are read safely.** PDF and Office files are read by a separate, time-limited process (`PARSER_TIMEOUT_SECONDS`, 120 s), so a damaged or hostile file cannot hang or crash the program: the worst result is that file failing with a reason, shown in the summary of `ingest` (`no_text`, `encrypted`, `corrupt`, `timeout`, `unsafe_archive`, ...). Office files are read from memory with the standard library only: an archive that unpacks to far more than it should (a zip bomb), has file names that could escape a folder, or contains XML entity declarations is refused; macros are never read or run, and no link inside a file is followed. The worker has no network access and does not get this program's environment variables. A memory limit is not applied: the file-size limit bounds the input, and the time limit stops the rest.

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

## The interface and the API

Everything the commands do is also available in the interface (**Knowledge**) and through the API, so a terminal is never required:

| In the interface | API |
|---|---|
| Documents, with the file name, size, passages and whether they are searchable (with paging) | `GET /api/v1/library/documents?offset=&limit=&state=` |
| **INSPECT** a document: every place it was found, earlier versions of an edited file, and its passages with their headings and lines or pages | `GET /api/v1/library/documents/{id}` and `/chunks?offset=&limit=` |
| Remove a document (its older versions too) | `DELETE /api/v1/library/documents/{id}` |
| UPDATE LIBRARY with progress in parts (not only documents), and a history of recent updates that survives a restart | `POST/GET /api/v1/library/sync`, `GET /api/v1/library/jobs` |
| Counts, model and Ollama state | `GET /api/v1/system/status` |
| **HELPFUL / NOT HELPFUL / WRONG SOURCE / MISSING INFO** under each answer | `GET/POST /api/v1/feedback`, `PUT/DELETE /api/v1/feedback/{id}` |
| **HISTORY**: saved conversations, reopen, rename, delete | `GET/POST /api/v1/conversations`, `GET/PUT/DELETE /api/v1/conversations/{id}`, `POST .../messages` |
| A cited answer that appears as it is written | `POST /api/v1/ask/stream` (server-sent events) |
| What the program is set to do (read-only) | `GET /api/v1/system/settings` |

One update runs at a time; asking for a second while one runs is refused. The history holds only counts and short sentences, never file names or paths. The TypeScript types in `frontend/src/api/types.ts` are checked against the API's own description by a test (`tests/unit/test_frontend_contract.py`), so a field added on one side only fails the build instead of showing up as a blank in the interface.

The API only accepts requests that name this computer, from this computer, and limits their size and number; see `docs/security.md`.

### Conversations, follow-ups and streaming

- **Saved conversations.** What you say and what comes back is kept on this computer, so a conversation can be reopened, renamed or deleted from **HISTORY** in the chat. The **SAVE** switch turns saving off. Titles, messages and the sources shown beside an answer are stored in encrypted columns when the library is encrypted. Deleting a conversation removes it and every message in it for good (the database overwrites deleted rows). If a note is removed from the library, the answers that quoted it are replaced by a short notice, so a removed note does not live on inside a conversation. `CONVERSATION_RETENTION_DAYS` in `.env` deletes conversations that were not touched for that many days; the default, 0, keeps them until you delete them.
- **Follow-up questions.** When you ask the notes something like "and how much was it?", the model first rewrites it into a question that stands alone, using the last few turns, and the interface shows what was searched for. The first message of a conversation is never rewritten. The answer is still built from the notes and checked exactly as before; a rewrite can never make an unrelated question answerable. `python -m app eval --answers` measures this on a group of follow-up questions (6 of 7 with the local model on 2026-10-05; the miss is a year hidden inside an invoice number).
- **Marking answers.** Under each answer are HELPFUL and NOT HELPFUL. After NOT HELPFUL on an answer from your notes two more appear: WRONG SOURCE (it used the wrong note) and MISSING INFO (your notes should have held the answer). Pressing the chosen mark again takes it back, and choosing another changes the same mark. A mark keeps the question and the answer on this computer, in encrypted columns when the library is encrypted, and it is kept only when you press a button (whatever the SAVE switch says about conversations). The sources are kept by name, never by their text, and if a note is removed from the library the answers that quoted it are removed from your marks too.
  `python -m app feedback` lists how many marks of each kind you have and your latest failures. `python -m app feedback export` turns every failure on an answer from your notes into an evaluation question waiting for review, in `eval-private/feedback-candidates.json` (a folder git ignores, because it holds your own questions): it names the files the assistant cited, says whether it refused, and lists what you still have to fill in, namely the note that should answer it (`expected_sources`) and what a good answer contains (`answer_contains`). Until you do, the entry cannot be loaded as part of an evaluation set, so it can never end up there by accident. Copy the reviewed entries into the `questions.json` of your private set and run `python -m app eval --set <folder>`. Running it again adds only new marks and leaves entries you edited alone. Known limit: a conversation you reopen later does not remember which answers you marked.
- **Streaming.** A cited answer appears as the model writes it (about 0.1 s to the first words instead of waiting for the whole answer) and a STOP button ends it. The streamed text is marked NOT CHECKED YET: citations are only verified when the answer is complete, and the checked answer then replaces what was streamed. An answer that cites a note that does not exist is refused at the end, exactly as without streaming. Streaming applies to answers from your notes; general chat answers in one piece.

## Checking and protecting your library

```powershell
python -m app doctor            # is everything healthy? read-only; says how to fix each problem
python -m app backup            # save the library to one archive (see below for where)
python -m app restore FILE --to C:\restored-library
```

**`doctor`** checks the database version, that every stored file exists and matches its checksum, that the search index agrees with the database, the allowed folders, free disk space, the embedding model, Ollama, and what Windows says: **Smart App Control**, whether **BitLocker** protects the drive that holds the library, which files of this program Windows **blocked** in the last 14 days (read from its own Code Integrity log, no administrator rights needed, listed by their place inside the package), and whether the scikit-learn placeholder is needed, and whether the library that turns text into vectors (PyTorch and sentence-transformers) can be loaded at all. That last check is a failure when it cannot, because nothing can then be searched, indexed or answered from your notes; it says what Windows blocked and that chat with the model alone (MY NOTES off) still works. It exits with code 1 only if something has failed (a BitLocker or blocked-file finding is a warning). If Windows cannot be asked, the check says so and how to look yourself. It changes nothing; `--fix` does one safe thing (marks documents for re-indexing when the search index disagrees with the database) and never deletes anything. `--quick` skips verifying checksums, which is faster for large libraries.

**`backup`** writes one `.zip` containing the database (copied with SQLite's own backup method, so it is consistent even while the library is in use), the stored copies of your files, and a manifest with a SHA-256 checksum of every entry. By default it goes to `%LOCALAPPDATA%\Reyleight\backups` (outside the repository and not synced); choose another place with `--to FILE`. It never overwrites a file and refuses to save inside the data folder it protects. The search vectors are not included, because `python -m app index` rebuilds them. **A backup contains your private notes**, so keep it somewhere safe.

**`restore`** verifies every checksum before writing anything, refuses an archive that has unexpected or unsafe entries, refuses to restore into a folder that is not empty, never touches your current library, and marks everything as "not searchable yet". To use the restored library, set `DATA_DIR` to that folder and run `python -m app index`.

## Troubleshooting

- **"blocked by an Application Control policy" (Windows).** Smart App Control can block newly installed compiled library files. Reyleight works around the one that matters here: if scikit-learn cannot be loaded, it is replaced by a placeholder, because only unused helper functions of sentence-transformers need it (a warning is logged, and embeddings are unaffected). If a different library file is blocked, the command stops with a one-line error; re-running while online sometimes clears it. Reyleight never changes Windows security settings.
- **"ollama is not reachable".** Open the Ollama app or run `ollama serve`, then retry. `python -m app status` shows its state.
- **"search index is in use by another process".** Only one program can open the vector index at a time. Commands wait about 20 seconds for it; if it is still busy, stop the API server or wait for the running command.
- **"the database is out of date".** Run `python -m app migrate` (or start the server with `python -m app serve`, which does it for you after saving a backup to `%LOCALAPPDATA%\Reyleight\backups`). `alembic.exe` can be blocked by Smart App Control; `python -m alembic upgrade head` does the same thing.
- **"This address is not allowed" (400) or "Requests from other websites are not allowed" (403).** The API only answers requests that name this computer. If you reach it through another name or from the development server, list it in `ALLOWED_HOSTS` or `CORS_ORIGINS` in `.env`.

## Configuration

Copy `.env.example` to `.env` and adjust. `.env`, `data/` and `models/` are gitignored.
