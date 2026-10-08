# Security and Privacy

## Principles

1. **Local-first.** No personal document, note or query text is sent to a cloud API. Embeddings and generation run locally. A cloud option, if ever added, is opt-in and clearly flagged.
2. **No blanket filesystem access.** Only allow-listed folders are indexed. The app never silently expands that scope or reads outside it.
3. **Retrieved content is untrusted data.** Documents (and any future tool or web output) are never treated as instructions and cannot grant permissions or trigger actions.
4. **Least privilege.** Any future tool or integration gets the narrowest scope it needs. Sensitive or irreversible actions need explicit confirmation. Destructive actions are denied by default.
5. **Grounded answers.** Citations come from application-controlled metadata. The model never invents a source id or path.
6. **Safe file handling.** Uploaded filenames are metadata only, never paths. Stored files use hashed paths.
7. **No secrets in git.** `.env`, `data/`, `models/` and database files are gitignored from the start.

## Network access

- The only command that uses the internet is `python -m app download-model`, run explicitly by the owner. It downloads the embedding model into `MODELS_DIR`.
- At runtime the model is loaded from that folder only. The Hugging Face libraries are switched to offline mode, so an accidental download attempt fails instead of reaching the network.
- `trust_remote_code` is always off: code shipped inside a model repository is never executed.
- The API server listens on `127.0.0.1` (the uvicorn default). `python -m app serve` refuses any other address unless the loudly named `--unsafe-expose-to-network` is given (it then prints a warning; not recommended, the access password would be the only protection). Do not start uvicorn with `--host 0.0.0.0`.

## Web pages cannot drive the API

A browser on this computer will send a request to `127.0.0.1` when any website tells it to. Three rules, applied to every request before the application sees it (`app/api/guard.py`), stop a page you happen to visit from using your assistant:

- **The Host header must name this computer** (`127.0.0.1`, `localhost`, `::1`, or a name in `ALLOWED_HOSTS`). This is what stops DNS rebinding, where a hostile site's name is made to point at `127.0.0.1` so the browser treats the local API as that site's own.
- **An Origin header, when the browser sends one, must be this computer too** (or listed in `CORS_ORIGINS`). A page on another site always sends its own Origin on a cross-site request, so it is refused with a 403 before anything else runs. A Host header that does not name this computer gets a 400. Only the origins listed in `CORS_ORIGINS` may read answers across sites; the default is none, because the interface is served from the same address.
- **Size and rate limits.** A request larger than `MAX_REQUEST_MB` (checked on the declared length and again while reading, so a missing or false length does not get past it) gets a 413. More than `RATE_LIMIT_PER_MINUTE` requests in a minute get a 429 with a `Retry-After`. Nothing in a request is logged.

Refusals are short JSON messages; an unexpected error answers a fixed sentence and never a stack trace. `/health` is the only route that skips the rate limit. Together with the access password this means a website cannot read your notes or start an update, and a script that does reach the API still needs the password.

Read-only views of the setup (`GET /system/settings`) hold no paths, passwords, tokens or keys, and cannot be changed through the API: settings stay in `.env`.

## Access password

- Every API route except `/health` and `/auth` needs a sign-in token. The interface shows a password screen first; the token lives in the page's memory only, so reloading or pressing LOCK asks again.
- The password is at least 8 characters and is stored only as a salted scrypt hash in `DATA_DIR/access.json`. There is no email recovery: if it is forgotten, delete that file and choose a new one. Notes are not affected.
- Tokens are random 256-bit values; only their hashes are kept, in memory, so a server restart signs everyone out. A session ends after 8 idle hours. After 4 wrong passwords each further attempt must wait longer, up to a minute.
- Known limit: until a password is chosen, anything that can reach the API on this machine could choose it. Choose it on first launch. `ACCESS_REQUIRED=false` turns the check off for development only.
- This protects use of the application. It does not encrypt files on disk; that is a separate feature.

## Folders and removal from the interface

- A folder added in the interface must be an existing, absolute folder. Refused: a whole drive, the user's home folder itself, system folders, `AppData`, folders that hold keys (`.ssh`, `.aws`, `.gnupg`, ...), and any folder that contains or sits inside Reyleight's own data or model folders (their files would otherwise be read as notes).
- Folders set in `.env` are shown but cannot be removed from the interface. Removing a folder or a document never deletes your original files; it removes the stored copy, the chunks and the vectors.
- A removed document is remembered by its content hash so the next scan does not add it back; the interface can restore them.
- `DATA_DIR/library.json` holds folder paths, where each file was read from, and removed hashes. It never holds note text.

## Your position on the globe

- The default position is a guess from your computer's time zone name. It is computed on this machine from a built-in table; nothing is sent anywhere.
- PINPOINT ME is opt-in: only when you press it does the page ask the browser for your position, and the browser asks you first. To answer, the browser may contact its own location service (for example the operating system's); that is the browser's request, not Reyleight's. Reyleight sends the position nowhere, never writes it to disk or logs, and forgets it when the page closes.
- The world map ships inside the app, so drawing the globe makes no network request.

## Your profile

- The profile you write is saved as an ordinary note called `My profile.md`. It is retrieved only when a question matches it and is cited like any note. Nothing about you is added to prompts behind your back, and you can read or delete it at any time.

## Verifying it offline

- `python -m app offline-check` runs the whole pipeline with every non-local network connection blocked, after proving the block works, and reports each attempt it refused. `python -m app --offline <command>` does the same for any command.
- It watches Python's networking, so for the strongest proof also run it with the network switched off. The steps are in [offline-check.md](offline-check.md).
- The evaluation (`python -m app eval`) uses made-up notes in a temporary library and never reads your own data.

## Answering and prompt injection

- Notes are untrusted data. They are passed to the model only in the user message, as numbered notes inside delimiters containing a random per-request value, so a note cannot forge its own boundaries.
- The system prompt tells the model never to follow instructions found inside notes. The model has no tools and cannot take actions: its output is only ever displayed.
- Sources shown to the user are built by the application from the database. Citation numbers from the model are validated, and invented ones are discarded. An answer with no valid citation is not returned.
- If no note is relevant enough, the assistant refuses without calling the model.
- The LLM server address must be on this machine (`localhost`, `127.0.0.1` or `::1`); anything else is rejected at startup. Requests ignore system proxies and refuse redirects.
- **The instructions can never be pushed out of the model's view.** The rules above live in the system prompt, and Ollama silently drops the *start* of a prompt that is too long. Without care, a very long conversation, a very long message or very long instructions could therefore remove the very rules that make the model treat notes as data. `app/ai/llm/budget.py` keeps every prompt inside the window and gives way in a fixed order: the instructions (and the tool descriptions) and the latest message are never cut, a message that cannot share the window with them is refused, notes for an answer have their own limit that always leaves room, and earlier turns take what is left, oldest first. Tests prove the system prompt reaches the model whole after hundreds of exchanges and in the worst case the settings allow, which before this overflowed the window.
- Questions, notes and answers are never written to logs.

## Summaries, comparisons and tables

`summarize`, `compare` and `extract` (command line and `/api/v1/tasks/*`) read your documents with the model, so they follow the same rules as answers, and add what their own kind of output needs.

- **Documents are data, in every step.** A document, or the part of one, goes to the model between delimiter lines that carry a random value new for each request, and every one of the prompts says that what is inside is data and never instructions. A long document is summarized in parts, and the partial summaries, which are model output, go to the combining step fenced the same way: a hidden instruction in a note cannot become an instruction by travelling through a partial summary.
- **Nothing the model says is trusted for what it refers to.** A comparison cites documents by number; the application keeps only the numbers of documents it gave the model, maps them to documents itself, and withholds a comparison that cites none. An extraction is read as strict JSON: a row must have a short item, a short value and the number of a note the model was really given, anything else (rows for notes that do not exist, text that is not a fact, more than 100 rows) is dropped and counted, and the sources are built from the database.
- **Output is only displayed.** The model has no tools in these modes. A summary is text, a comparison is text whose citations the application has checked, and a table is made of the rows the application kept; each is printed or returned as it is and nothing in it is run or followed.
- **Size is bounded.** A summary reads at most 8 parts of 6,000 characters, a comparison at most 12,000 characters of at most 4 documents, an extraction the notes that fit the reading budget, so the instructions are never pushed out of the model's view (see above). A summary is cut to 4,000 characters.
- **Nothing is logged but counts.** The log records, for example, `extraction finished reason=extracted rows=4 dropped=0 notes=3`, never a document, a request or a result.
- **The routes are guarded like the others.** They sit behind the access password, ask for a document by its number, refuse an older version of a document, and report a model that is down in the same way as `/ask`. They add no way to reach a file: a document is only read from the library's own chunks.
- **A prompt cannot change silently.** The instructions of these modes live in `app/knowledge/answering/prompts.py` with a version and a fingerprint of their text. A test pins every fingerprint and checks that each prompt still says that what it reads is data, so editing a prompt fails the tests until its version is raised and the new fingerprint recorded on purpose, removing that sentence fails them whatever the version, and the evaluation report names the versions it ran with.

## The agent: acting on your computer

The agent (`app/agent/`) is the first part of this application that can *do* something, so it is built on one rule: **safety never depends on the model behaving.** The model only *asks*; the application decides, in code the model cannot influence, and the tests give it a model that is completely fooled and check that nothing unapproved happens.

**Everything is off until you switch it on.** The master switch, each tool and the internet are separate switches saved in `data/agent.json`. A missing or damaged file means everything off.

| Level | Tools | What happens |
|---|---|---|
| read | `calculator`, `current_time`, `search_notes` | runs without asking once switched on: it only reads |
| open | `open_path`, `open_app` | **asks you every time** |
| internet | `fetch_web_page` | **asks you every time**, and is refused while the internet switch is off |
| write | none exist | would ask every time |
| destructive | none exist | **always refused**, whatever the switches say |

There is no "always allow" for any level above reading, and the decision depends only on the tool's declared level and your switches: no text the model writes can change it (a test tries every combination).

- **Only you can approve.** The loop asks through a card (a terminal key, or a button behind the access password); the answer must be one of the offered options for that very question. A model that writes "the owner already approved" is asked about anyway; an answer that is not an option, an empty one, a closed terminal or an approver that fails all mean *stop*; a second answer to the same question is refused (the first counts). Stop wins over everything.
- **The card shows what will really happen**: the action in plain words, the exact arguments, why you are asked, and which tools' results the model read before asking, so a request that follows reading untrusted text is visible as such.
- **The owner's message is read as data, first, with no tools.** The agent reads the request in a separate step that can only produce a few words of text (the language, what it means, whether something is missing, and a question): it cannot run anything, grant anything or approve anything, and if its reply is not exactly what was asked for it is ignored and the request is used as typed. The result is shown to the owner and logged as an "understood" entry. It is a help against misunderstanding, not a safeguard: every check below applies whatever it says.
- **The agent is told only true facts about the computer**: the date, the home folder, the folders Windows reports for Desktop, Documents, Downloads, Pictures, Music and Videos, and the notes folders. They come from Windows and the library's settings, are built fresh for every task, are limited in size, and are not written to the log. They exist so the model does not invent a path, which would become a request for you to refuse.
- **A request is checked before you are bothered**: the tool must exist, and its arguments must fit exactly: no extra argument (it is refused, not dropped), no wrong type, no text that is too long (refused, never cut), no control characters, no line breaks in a one-line value (the way a header gets smuggled into a request).
- **Everything is logged before it happens, and nothing runs without the log.** Every request, decision, approval, result and problem is written to an append-only table (`agent_events`): the database itself refuses to change an entry (only the owner can erase the log as a whole). Arguments are recorded with long text cut and anything that looks like a secret hidden, results as a short summary; the text is encrypted with the library. If an entry cannot be written, the run stops before acting.
- **What goes wrong becomes a question, not a retry.** A failing or slow tool, a model that fails or says nothing, repeated requests that cannot be carried out, and the step and time limits all stop the run and ask you. A tool that crashes does not end the run and its message is never shown or logged (only the kind of error is).
- **Tool results are untrusted data.** They reach the model between delimiter lines with a random marker no result can predict, are cut to a limit, and the prompt says they are data, not instructions. The point is that even a model that obeys them can only *ask*: the checks above still hold.
- **Opening things** (`open_path`, `open_app`): a path must be written in full and exist, and is resolved first (so `sub\..\setup.exe`, `setup.exe.` or `setup.exe::$DATA` are judged as the real file, and the file opened is the checked one). Files that run code when opened are refused: programs and scripts, links, installers, macro-enabled Office files, web pages and vector images that run script in a browser, remote-connection files. Paths on another computer or a device are refused. Programs are started only from a short fixed list, alone, with no arguments.
- **Reading the web** (`fetch_web_page`): a plain GET, no cookies, no login, no script run. Only the public internet: the site's name is looked up and *every* address it has must be public (not this computer, not a home or office network, not the link-local range cloud services keep credentials in, IPv4 addresses dressed as IPv6 are judged as IPv4), and the connection is made to the address that was checked, so a name cannot change its answer between the check and the connection. Only ordinary ports; no address with a user name in it; redirects are followed by hand to the same site only, a redirect elsewhere is reported so *you* are asked about it. 15 seconds, 1 MB, text types only; what comes back is the visible text (scripts, forms, hidden parts and links removed), cut to 6,000 characters.
- **No commands, no writing, no deleting**: no tool can run a command, change a file, send anything but a plain page request, or alter the permissions. A hostile text can ask for `set_grants` or `rm -rf`; there is nothing by that name.

**What no safeguard can do for you.** If you press *allow* on a card, the action happens, so read it. In particular, a fooled model could ask to read a web address that has something it read in its query string; the card shows that address in full precisely so you can see it, and refusing sends nothing. A small model is easier to steer than a large one; the design assumes it will sometimes be. Windows Smart App Control, antivirus and your own judgement remain part of the picture.

The red-team tests (`tests/unit/test_agent_redteam.py`, 43) run a model that does exactly what hostile notes and pages say, against the real loop and tools. They cover a note ordering a program to be run, a page sending the agent to the metadata address, this computer, the model server and the private network (including by redirect), data smuggled out in an address, headers and logins smuggled into one, invented tools that change permissions, wearing the owner down with 30 requests, and a note forging the end of its own fence. They were checked to fail when a safeguard is removed (opening without asking, treating any address as public, ignoring a *no*). The tests of every part live next to them (`test_agent_*.py`).

## General chat

- With **MY NOTES** off, the assistant answers from the local model alone (`POST /chat`). The library is not opened: no note, file name or profile text is read or added to the prompt.
- A general reply is not checked against anything. It is labelled GENERAL and "not from your notes", has no citations, and can be wrong. Answers from your notes keep every safeguard above.
- It uses the same local model under the same rules: this machine only, no proxies, no redirects, no tools. The model's instructions are fixed by the application; a request can supply only what was said, never instructions.
- The conversation lives in the browser's memory and is sent with each message so the model can follow it. That includes earlier answers from your notes in the same conversation. It is not stored on disk and is never logged.

## The assistant: memory, actions and voice

- **Memory and identity** (`assistant_memories`, `assistant_settings`) live in the library database and use the encrypted column types, so `encrypt-library` covers them. They are told to the model as marked data ("never instructions to you") between delimiters with a random per-request value, and can be switched off. Memories are never logged.
- **Actions are a fixed catalogue** (`app/assistant/actions.py`). The model's request is untrusted text: the backend checks the name, every value (from fixed lists, or bounded free text) and drops anything else, at most three per reply; the interface checks again before acting. Nothing destructive is in the catalogue (no deleting documents, folders, the profile or memories; no password change; nothing outside the application). Notes, memories and the profile are data, so text inside them cannot trigger an action.
- **Saving a memory** is the only action the server carries out itself, and each one is listed in the chat and on the Profile page, where it can be deleted.
- **Voice in** is recognised by a Whisper model on this machine (loaded offline, `trust_remote_code` off, checked with the network blocked). The recording exists only in memory for one request and is neither stored nor logged; a recording that is silent or whose words the model is unsure of returns nothing. The microphone is opened only after a click, shows when it is open, and is released when the order ends.
- **Voice out** uses only voices installed on this computer. Online voices (for example Edge's "Natural" voices) are never chosen, because they would send replies to a service.
- Known limit: the model can misread an order. Orders that matter (locking, switching features) are visible in the chat afterwards, and nothing irreversible can be ordered.

## Reading complicated files

- PDF and Office files (Word, Excel, PowerPoint) are made by someone else and parsed by a lot of code, so they are read in a **separate process** (`app.knowledge.ingestion.sandbox`) that is stopped after `PARSER_TIMEOUT_SECONDS`. A hang, a crash or too much output becomes a failed file with a reason code; nothing else is affected.
- The file goes to that process on stdin and one JSON document comes back; what happened inside is never reported, because it can contain pieces of the file. The process gets only a few system environment variables (not this program's, so never `REYLEIGHT_PASSPHRASE`), and it cannot open network connections.
- Office files are read with the standard library only, from memory (nothing is unpacked to disk). Refused: archives that unpack to far more than they should (zip bombs: a ratio and size check before anything is read), entry names that could escape a folder (absolute paths, `..`, drive letters, backslashes), archives marked as password-protected, and any XML that declares a DOCTYPE or entity (how entity-expansion attacks work). Macros are never read or run, links inside a file are never followed, and formulas are never evaluated.
- Not applied: a memory limit on the process (it would need a Windows job object). The file-size limit bounds what goes in, and the time limit stops the rest.
- The size of what comes back is capped, so a file that expands into gigabytes of text is refused.

## Searching by words

- The keyword index is an in-memory SQLite FTS5 table, built from the notes through the normal (decrypting) columns and thrown away with the process. Nothing about it is written to disk, so an encrypted library never has its words in a plaintext index. A test searches an encrypted library by an exact code and checks that no word of the notes is readable anywhere in the data folder afterwards.
- A question is cut into plain words, each word is quoted, and the words are joined with OR, so FTS5 operators, quotes and column filters in a question are only words and cannot change the query.
- The relevance gate always uses meaning similarity. A strong keyword match never makes a note "relevant enough" to answer from.

## Saved conversations

- A conversation is made of your own words and of answers built from your notes, so it is treated like note text: it lives in the library database, in encrypted columns (title, message text, the details shown beside an answer) when the library is encrypted, and it is never logged. Conversations are included in backups like the rest of the database.
- Deleting a conversation deletes its messages in the same step, and the database overwrites deleted rows, so the words do not stay readable inside the file. Saving can be turned off in the chat (**SAVE**). `CONVERSATION_RETENTION_DAYS` removes conversations that were untouched for that long; the default keeps them until you delete them.
- Removing a note from the library replaces every answer that quoted it by a notice. The check uses a plain column that holds only document numbers (which note was quoted, never what it said), so it does not need to decrypt every message. Not covered: an answer that never quoted the note but still repeats what it says, and older copies the file system or a backup may hold.
- The earlier turns of a conversation are sent to the local model to rewrite a follow-up question. They are shortened, fenced with a random marker and described to the model as data, never instructions; the rewrite must come back as one short line or it is ignored, and it only changes what is searched for. What can be answered is still decided by the relevance gate and the citation check.
- **Marks on answers** (HELPFUL, NOT HELPFUL, WRONG SOURCE, MISSING INFO) keep the question and the answer, so they are treated like note text: encrypted columns when the library is encrypted, never logged, and overwritten when deleted. They are made only when you press a button, whatever the SAVE switch says about conversations. The sources are kept by file name, heading and number, never by their text. Removing a note from the library replaces the answer of every mark that quoted it by the same notice used in conversations, drops your note on the mark (it may be about the quoted text) and keeps your question, which is yours. `python -m app feedback export` writes the failures to `eval-private/feedback-candidates.json`, a folder git ignores, because it holds your own words; the file is replaced in one step so an interruption cannot leave half of it, and an entry that is not reviewed cannot be loaded as part of an evaluation set.
- A streamed answer is not verified until it is complete. The interface marks it NOT CHECKED YET, and the final checked answer replaces it. Closing the connection stops the model.

## Edited and deleted files

- A sync never deletes anything. An edit replaces a document with a new version and keeps the old text as history that is never searched; a file that stays away from two syncs in a row marks its document as missing, and a missing document is not searched. Only `python -m app prune`, after showing what it would remove and asking, deletes documents (and their text, vectors and stored copies).
- Where each file was found (folder and path) is stored in the database in encrypted columns. Because encrypted values cannot be searched in SQL, the sync loads them and compares in memory. `data/library.json` no longer keeps the path of every document; it still holds the folders you added in the interface, in plain text.
- Deleted rows are overwritten with zeros (`secure_delete`), and `prune` compacts the database file afterwards, so the text of a removed note is not left readable inside it. This does not reach older copies the file system or an SSD may keep, nor backups you made earlier.

## Logs and the drive

- **Logs hold counts and kinds, never words.** A log line says that a search finished with 4 results, or that an answer was refused for a given reason; it never contains a question, an answer, a note, a file name, a folder, a conversation or a profile value. An error is logged by its type only. `tests/unit/test_log_scrubber.py` puts a distinctive marker into each kind of your own text, drives the main routes with them and fails if any marker appears in any log, so a careless new log line is caught. `python -m app serve` also turns off the per-request lines, because a request line carries its query string (a folder path, for one).
- **BitLocker.** `doctor` asks Windows whether the drive that holds the library is protected and warns when it is not, unless the library is encrypted by Reyleight itself. It only reports; it never changes the setting. The library encryption below protects the notes inside the database and stored files; BitLocker also covers everything else on the drive (the search vectors, the key file, backups kept there). Windows can answer this without administrator rights; where it cannot, `doctor` says how to check by hand. An encrypted database engine (SQLCipher) is not used: it is a compiled dependency Smart App Control may block, and BitLocker plus the library encryption cover the same ground.
- **Files Windows blocks.** `doctor` reads the Code Integrity log (read-only) and lists the files of this program that Windows refused to load in the last 14 days, so a block is diagnosed in seconds instead of from a cryptic import error. Reyleight never turns Smart App Control off or changes any Windows security setting.

## Encryption at rest

- `python -m app encrypt-library` encrypts note text, headings, file names and the stored copies of files with AES-256-GCM, using Windows' own cryptography (no extra dependency). Backups contain only the encrypted forms.
- A random 256-bit library key is kept in `security.json`, wrapped twice: by the Windows account (automatic unlock) and by a recovery passphrase (scrypt, then AES-256-GCM). The key file is not secret on its own.
- Every value is authenticated and bound to its column (or file name). A modified value fails to decrypt; a value moved to another column or file fails too.
- Safe failure: an encrypted library is never opened without its key, never written to in plaintext, and a half-finished encryption is refused until it is resumed. A library that contains encrypted data but lost its key file is refused, not treated as plaintext.
- Not covered: the search vectors (needed readable for search), structure such as sizes, line numbers and file hashes, malware running as the signed-in user, and plaintext that was on disk before encryption (see the README for `cipher /w`). Keys cannot be wiped from memory in Python.
- The keyword index (SQLite full-text search) is built in memory and never written to disk, so it does not break this (see "Searching by words").

## Threats to keep in mind

- Path traversal through filenames.
- Prompt injection through document content.
- Accidental indexing of files outside the allow-list.
- Leaking personal data through logs. Logs must not contain document or query text.
