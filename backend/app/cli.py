"""Command-line interface: `python -m app <command>`. Run `python -m app --help` for the list."""

import argparse
import getpass
import io
import ipaddress
import json
import os
import socket
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.agent.console import (
    ACTIONS,
    ConsoleError,
    Environment,
    LazySearch,
    agent_command,
)
from app.agent.tools import ToolError
from app.ai.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRuntimeError,
    ModelNotAvailableError,
    is_model_downloaded,
    model_dir_for,
)
from app.ai.llm.base import LLMError, LLMUnavailableError
from app.ai.speech.base import is_speech_model_downloaded
from app.assistant.feedback import FAILURES, KINDS, count_by_kind, list_feedback
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.evaluation.candidates import DEFAULT_FILE, CandidatesFileError, export_candidates
from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.network_guard import NetworkBlocked, NetworkGuard
from app.evaluation.report import format_report, to_dict
from app.evaluation.runner import hit_rate, run_evaluation
from app.knowledge.answering.service import Answer, Source, compose_answer
from app.knowledge.answering.tasks import (
    Extraction,
    compare,
    extract,
    load_document_text,
    summarize,
)
from app.knowledge.chunking.service import rechunk_all
from app.knowledge.components import (
    create_llm,
    load_embedder,
    open_vector_store,
    vector_store_path,
)
from app.knowledge.indexing.service import (
    IndexProgress,
    IndexSummary,
    count_pending,
    count_stale_vectors,
    index_pending,
    library_counts,
    reset_index,
)
from app.knowledge.ingestion.file_types import FILE_TYPES
from app.knowledge.ingestion.service import IngestSummary, ingest_folders
from app.knowledge.library.documents import clear_legacy_sources
from app.knowledge.library.folders import allowed_folders
from app.knowledge.library.prune import find_candidates, prune
from app.knowledge.library.state import load_state
from app.knowledge.retrieval.keyword import KeywordSearchUnavailable
from app.knowledge.retrieval.service import (
    SEARCH_MODES,
    RetrievedChunk,
    describe_location,
    retrieve,
)
from app.operations.backup import BackupError, create_backup, restore_backup
from app.operations.doctor import format_checks, run_doctor
from app.operations.upgrade import UpgradeError, UpgradeResult, upgrade_database
from app.security import keystore
from app.security.errors import KeystoreError, SecurityError, WrongPassphraseError
from app.security.migrate import encrypt_library
from app.storage.database import create_db_engine, vacuum_database
from app.storage.migrations import database_is_up_to_date
from app.storage.models import DOC_ACTIVE, Document
from app.storage.vector_store import (
    QdrantVectorStore,
    VectorStoreError,
    collection_name,
    count_local_vectors,
)


class CliError(Exception):
    """A problem reported to the user as one line, with exit code 1."""


Handler = Callable[[argparse.Namespace, Settings], int]


PASSPHRASE_ENV = "REYLEIGHT_PASSPHRASE"
NEW_PASSPHRASE_ENV = "REYLEIGHT_NEW_PASSPHRASE"


def _ask_secret(prompt: str, env_name: str) -> str:
    """A passphrase from an environment variable (for automation), or typed without echo."""
    value = os.environ.get(env_name)
    if value:
        return value
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    raise CliError(f"a passphrase is needed but this is not a terminal; set {env_name}")


def _passphrase_for_unlock(settings: Settings, *, required: bool = True) -> str | None:
    """The recovery passphrase, asked only if Windows cannot unlock the encrypted library."""
    try:
        if keystore.library_state(settings.data_dir) != "encrypted":
            return None
        if keystore.windows_unlock_works(settings.data_dir):
            return None
    except KeystoreError as exc:
        raise CliError(str(exc)) from exc
    try:
        return _ask_secret("Recovery passphrase to unlock the library: ", PASSPHRASE_ENV)
    except CliError:
        if required:
            raise
        return None


def _open_engine(settings: Settings) -> Engine:
    passphrase = _passphrase_for_unlock(settings)
    try:
        engine = create_db_engine(settings.data_dir, passphrase=passphrase)
    except SecurityError as exc:
        raise CliError(str(exc)) from exc
    if passphrase is not None and keystore.windows_unlock_works(settings.data_dir):
        print(
            "Unlocked. Windows will now unlock this library automatically for your account.",
            file=sys.stderr,
        )
    if not database_is_up_to_date(engine):
        raise CliError("database missing or out of date. Run `python -m app migrate` first.")
    return engine


def _load_embedder(settings: Settings) -> EmbeddingProvider:
    try:
        return load_embedder(settings.embedding_model, settings.models_dir)
    except (ModelNotAvailableError, EmbeddingRuntimeError) as exc:
        raise CliError(str(exc)) from exc


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    hours, rest = divmod(int(seconds / 60), 60)
    return f"{hours} h {rest} min"


def _progress_printer() -> tuple[Callable[[IndexProgress], None], Callable[[], None]]:
    """A progress line on stderr: it rewrites itself on a terminal and prints rarely elsewhere.

    Returns (show, finish). Only counts and times are shown, never note text or names.
    """
    tty = sys.stderr.isatty()
    interval = 0.5 if tty else 15.0
    last_shown = [0.0]
    shown_any = [False]

    def show(progress: IndexProgress) -> None:
        if progress.chunks_total == 0:
            return
        now = time.monotonic()
        finished = progress.chunks_done >= progress.chunks_total
        if not finished and now - last_shown[0] < interval:
            return
        last_shown[0] = now
        line = (
            f"indexing: {progress.chunks_done:,} of {progress.chunks_total:,} chunks "
            f"({progress.chunks_done * 100 // progress.chunks_total}%)"
        )
        if progress.chunks_per_second:
            line += f", {progress.chunks_per_second:.0f} chunks/s"
        left = progress.seconds_left
        if left is not None and not finished:
            line += f", about {_duration(left)} left"
        if tty:
            print("\r" + line.ljust(79), end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        shown_any[0] = True

    def finish() -> None:
        if tty and shown_any[0]:
            print(file=sys.stderr)

    return show, finish


def _run_index(session: Session, settings: Settings, *, rebuild: bool) -> IndexSummary:
    embedder = _load_embedder(settings)
    show, finish = _progress_printer()
    try:
        with _open_store(settings, embedder) as store:
            if rebuild:
                reset_index(session, store)
            return index_pending(session, embedder, store, on_progress=show)
    except VectorStoreError as exc:
        raise CliError(str(exc)) from exc
    finally:
        finish()


def _index_after_changes(session: Session, settings: Settings) -> None:
    """Make new or re-chunked documents searchable, when the model is available."""
    pending = count_pending(session, settings.embedding_model)
    stale = count_stale_vectors(session)
    if pending == 0 and stale == 0:
        print("search index:              up to date")
        return
    if not is_model_downloaded(settings.models_dir, settings.embedding_model):
        if pending:
            print(
                f"search index:              {pending} document(s) not searchable yet; run "
                "`python -m app download-model`, then `python -m app index`"
            )
        else:
            print("search index:              up to date")
        return
    summary = _run_index(session, settings, rebuild=False)
    print(
        f"search index:              {summary.documents_indexed} document(s) indexed, "
        f"{summary.chunks_embedded} chunks embedded"
    )
    if summary.documents_purged:
        print(
            f"search index:              {summary.documents_purged} replaced or missing "
            "document(s) removed from the index"
        )


# What each failure reason code means, for the person reading the summary.
FAILURE_HELP = {
    "no_text": "no text to read (probably a scan, a picture of text, which would need OCR)",
    "encrypted": "protected by a password",
    "corrupt": "damaged, or not really this type of file",
    "unsafe_archive": "an archive that looks like an attack (huge when unpacked, or bad paths)",
    "old_format": "an old binary Office file (.doc, .xls, .ppt) or a password-protected one",
    "too_many_pages": "more pages than PDF_MAX_PAGES allows",
    "timeout": "took longer than PARSER_TIMEOUT_SECONDS to read, so it was stopped",
    "crashed": "the reader stopped unexpectedly on this file",
    "too_large_output": "holds far more text than a reader may return",
    "not_utf8": "text that is not UTF-8",
    "binary": "a binary file, not text",
    "invalid_json": "not valid JSON",
    "unreadable": "could not be read (is it open in another program?)",
    "chunking": "its text could not be split into pieces",
}


def _print_ingest_summary(summary: IngestSummary) -> None:
    print(f"added:                     {summary.added}")
    print(f"unchanged:                 {summary.unchanged}")
    print(f"chunks created:            {summary.chunks_created}")
    print(f"skipped (empty):           {summary.skipped_empty}")
    print(f"skipped (too large):       {summary.skipped_too_large}")
    print(f"skipped (unsupported):     {summary.skipped_unsupported}")
    if summary.unsupported_by_extension:
        breakdown = ", ".join(
            f"{ext}={count}" for ext, count in sorted(summary.unsupported_by_extension.items())
        )
        print(f"  by type:                 {breakdown}")
    print(f"skipped (outside allowed): {summary.skipped_outside_allowlist}")
    print(f"allowed folders missing:   {summary.folders_missing}")
    failures = ", ".join(f"{reason}={count}" for reason, count in sorted(summary.failed.items()))
    print(f"failed:                    {sum(summary.failed.values())} {failures}".rstrip())
    for reason in sorted(summary.failed):
        if reason in FAILURE_HELP:
            print(f"  {reason + ':':<24} {FAILURE_HELP[reason]}")
    sync = summary.sync
    if sync.superseded:
        print(f"replaced by an edit:       {sync.superseded} (older versions are kept as history)")
    if sync.reactivated:
        print(f"back again:                {sync.reactivated}")
    if sync.newly_missing:
        print(
            f"files not found again:     {sync.newly_missing} document(s) now marked missing "
            "(`python -m app prune` removes them)"
        )
    if sync.paused_folders:
        print(
            f"folders not judged:        {sync.paused_folders} (many files vanished at once; "
            "is the folder available?)"
        )


# --- commands ---------------------------------------------------------------------------------


def _print_upgrade(result: UpgradeResult) -> None:
    if not result.upgraded:
        print("the database is up to date.")
        return
    before = (
        "a new database" if result.from_revision is None else f"revision {result.from_revision}"
    )
    print(f"database upgraded: {before} -> {result.to_revision}")
    if result.backup is not None:
        print(f"a copy from before the upgrade is saved at: {result.backup}")


def _cmd_migrate(_args: argparse.Namespace, settings: Settings) -> int:
    """Bring the database up to the version this program needs (after saving a copy of it)."""
    try:
        _print_upgrade(upgrade_database(settings))
    except UpgradeError as exc:
        raise CliError(str(exc)) from exc
    return 0


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Start the API (and the interface behind it) on this computer, after making sure the database
    is up to date. It refuses to listen on a network address: that would let other computers read
    your notes."""
    if not _is_loopback(args.host):
        if not args.unsafe_expose_to_network:
            raise CliError(
                f"refusing to listen on {args.host}: that would let other computers on the network "
                "reach your notes. Use 127.0.0.1 (the default). If you really mean it, add "
                "--unsafe-expose-to-network, and read docs/security.md first."
            )
        print(
            f"WARNING: listening on {args.host}. Anyone who can reach this computer can reach the "
            "sign-in page. The access password is the only protection.",
            file=sys.stderr,
        )
    if not args.no_migrate:
        try:
            _print_upgrade(upgrade_database(settings))
        except UpgradeError as exc:
            raise CliError(str(exc)) from exc
    import uvicorn

    print(f"Reyleight is starting at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    # No per-request log lines: they would carry the query string of each request (a folder path,
    # for one). The application logs what it does, in counts and kinds, never text.
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_config=None, access_log=False)
    return 0


def _cmd_types(_args: argparse.Namespace, _settings: Settings) -> int:
    print("Supported file types:")
    for file_type in FILE_TYPES:
        extensions = " ".join(file_type.extensions)
        print(f"  {file_type.name:<18} {extensions:<16} {file_type.description}")
    return 0


def _cmd_ingest(_args: argparse.Namespace, settings: Settings) -> int:
    folders = [entry.path for entry in allowed_folders(settings)]
    if not folders:
        print("ALLOWED_FOLDERS is empty. Nothing to ingest. Set it in .env.")
        return 0
    engine = _open_engine(settings)
    with Session(engine) as session:
        state = load_state(settings.data_dir)
        summary = ingest_folders(
            session,
            settings.data_dir,
            folders,
            settings.max_file_size_mb * 1024 * 1024,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
            excluded_hashes=frozenset(state.excluded),
            legacy_sources=state.sources,
        )
        clear_legacy_sources(settings.data_dir)  # now recorded, encrypted, in the database
        _print_ingest_summary(summary)
        _index_after_changes(session, settings)
    return 0


FEEDBACK_LIST_LIMIT = 10
_KIND_LABELS = {
    "helpful": "helpful",
    "not_helpful": "not helpful",
    "wrong_source": "wrong source",
    "missing_info": "missing information",
}


def _cmd_feedback(args: argparse.Namespace, settings: Settings) -> int:
    """Show the marks you put on answers, or turn the failures into evaluation questions."""
    engine = _open_engine(settings)
    with Session(engine) as session:
        if args.action == "export":
            target = Path(args.to) if args.to else DEFAULT_FILE
            try:
                result = export_candidates(session, target)
            except CandidatesFileError as exc:
                raise CliError(str(exc)) from exc
            if result.added:
                print(f"added {result.added} evaluation question(s) to review in {result.path}")
            else:
                print("nothing new to add: every marked failure is already in the file.")
            if result.kept:
                print(f"already in the file, left as they were: {result.kept}")
            if result.skipped_general:
                print(
                    f"{result.skipped_general} failure(s) on general chat were not added: the "
                    "evaluation checks answers from your notes."
                )
            if result.added:
                print(
                    "Open the file, name the note that should answer each question "
                    "(expected_sources) and say what a good answer contains (answer_contains), "
                    "then copy the entries into the questions.json of your private evaluation set "
                    "and run `python -m app eval --set <folder>`."
                )
            return 0
        counts = count_by_kind(session)
        if not any(counts.values()):
            print("no marks yet. Mark an answer in the chat: helpful, not helpful, wrong source,")
            print("or missing information.")
            return 0
        print("marks: " + ", ".join(f"{counts[k]} {_KIND_LABELS[k]}" for k in KINDS))
        for mark in list_feedback(session, kinds=FAILURES, limit=FEEDBACK_LIST_LIMIT):
            question = " ".join(mark.question.split())
            question = question if len(question) <= 70 else question[:67] + "..."
            print(f"  #{mark.id:<4} {_KIND_LABELS[mark.kind]:<19} {question}")
        print("`python -m app feedback export` turns the failures into evaluation questions.")
    return 0


PRUNE_LIST_LIMIT = 40

# How long a command waits for the search index when the server has it open.
STORE_WAIT_SECONDS = 20.0


def _open_store(
    settings: Settings, embedder: EmbeddingProvider
) -> AbstractContextManager[QdrantVectorStore]:
    """The search index, waiting a little if the server is using it right now."""
    return open_vector_store(
        settings,
        embedder,
        wait_seconds=STORE_WAIT_SECONDS,
        on_wait=lambda: print(
            "the search index is in use (by the server?); waiting for it ...",
            file=sys.stderr,
            flush=True,
        ),
    )


def _cmd_prune(args: argparse.Namespace, settings: Settings) -> int:
    """Remove documents whose files are gone. Shows them first and asks before deleting."""
    engine = _open_engine(settings)
    with Session(engine) as session:
        found = find_candidates(session, include_superseded=args.superseded)
        if found.total == 0:
            print("nothing to remove: no document is marked missing.")
            if not args.superseded:
                print("(--superseded also removes older versions of files that were edited.)")
            return 0
        listing = [("missing", d) for d in found.missing] + [
            ("replaced", d) for d in found.superseded
        ]
        for label, document in listing[:PRUNE_LIST_LIMIT]:
            print(f"  {label:<9} {document.original_filename}  (id {document.id})")
        if len(listing) > PRUNE_LIST_LIMIT:
            print(f"  ... and {len(listing) - PRUNE_LIST_LIMIT} more")
        print(f"{len(found.missing)} missing, {len(found.superseded)} replaced version(s).")
        print("Their stored copies, text and search entries would be removed.")
        print("Your original files in your folders are never touched.")
        if args.dry_run:
            print("dry run: nothing was removed.")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print("not removing anything: run again with --yes to confirm.", file=sys.stderr)
                return 1
            if input("Remove them? Type yes to confirm: ").strip().lower() != "yes":
                print("nothing was removed.")
                return 0
        try:
            if is_model_downloaded(settings.models_dir, settings.embedding_model):
                with _open_store(settings, _load_embedder(settings)) as store:
                    removed = prune(session, settings.data_dir, found, store)
            else:
                removed = prune(session, settings.data_dir, found, None)
        except VectorStoreError as exc:
            raise CliError(str(exc)) from exc
    vacuum_database(engine)  # so the removed text does not stay inside the database file
    print(
        f"removed {removed.documents} document(s), {removed.chunks} chunks and "
        f"{removed.stored_files} stored file(s); the database file was compacted."
    )
    return 0


def _cmd_rechunk(_args: argparse.Namespace, settings: Settings) -> int:
    engine = _open_engine(settings)
    with Session(engine) as session:
        total = rechunk_all(
            session, settings.data_dir, settings.chunk_size_chars, settings.chunk_overlap_chars
        )
        print(f"rechunked all stored documents: {total} chunks")
        _index_after_changes(session, settings)
    return 0


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    engine = _open_engine(settings)
    with Session(engine) as session:
        summary = _run_index(session, settings, rebuild=args.rebuild)
    print(
        f"indexed {summary.documents_indexed} document(s), "
        f"{summary.chunks_embedded} chunks embedded"
    )
    return 0


def _cmd_status(_args: argparse.Namespace, settings: Settings) -> int:
    engine = _open_engine(settings)
    with Session(engine) as session:
        counts = library_counts(session, settings.embedding_model)
    downloaded = is_model_downloaded(settings.models_dir, settings.embedding_model)
    try:
        vectors: int | str = count_local_vectors(
            vector_store_path(settings), collection_name(settings.embedding_model)
        )
    except VectorStoreError as exc:
        vectors = f"unknown: {exc}"

    print(f"embedding model: {settings.embedding_model}")
    print(f"  downloaded:    {'yes' if downloaded else 'no (run `python -m app download-model`)'}")
    print(f"documents:       {counts.active}")
    print(f"chunks:          {counts.chunks}")
    print(f"searchable:      {counts.searchable} of {counts.active} documents")
    if counts.missing:
        print(f"missing:         {counts.missing} (file not found; `python -m app prune`)")
    if counts.superseded:
        print(f"older versions:  {counts.superseded} (kept as history)")
    print(f"vectors:         {vectors}")
    print(f"llm model:       {settings.llm_model}")
    print(f"  ollama:        {_llm_state(settings)}")
    return 0


def _llm_state(settings: Settings) -> str:
    try:
        installed = create_llm(settings, timeout_seconds=5).list_models()
    except LLMUnavailableError:
        return f"not running at {settings.ollama_url} (open the Ollama app or run `ollama serve`)"
    except LLMError as exc:
        return f"problem: {exc}"
    wanted = settings.llm_model if ":" in settings.llm_model else f"{settings.llm_model}:latest"
    if wanted in installed:
        return "running, model installed"
    return f"running, but the model is not installed (run `ollama pull {settings.llm_model}`)"


def _cmd_check_llm(_args: argparse.Namespace, settings: Settings) -> int:
    print(f"asking {settings.llm_model} at {settings.ollama_url} ...")
    started = time.monotonic()
    try:
        reply = create_llm(settings).generate(
            "This is a connection test. Reply with the single word: ready", "Are you ready?"
        )
    except LLMError as exc:
        raise CliError(str(exc)) from exc
    print(f"reply: {' '.join(reply.split())[:200]}")
    print(f"took {time.monotonic() - started:.1f}s")
    return 0


def _where(item: RetrievedChunk | Source) -> str:
    """Pages for a file that has them, lines otherwise."""
    return describe_location(item.start_line, item.end_line, item.start_page, item.end_page)


def _print_result(rank: int, result: RetrievedChunk) -> None:
    heading = f"  [{result.heading_path}]" if result.heading_path else ""
    print(
        f"{rank}. score {result.score:.3f}  {result.source}{heading}  "
        f"{_where(result)}  (id {result.citation_id})"
        + (f"  keyword {result.keyword_score:.1f}" if result.keyword_score is not None else "")
    )
    snippet = " ".join(result.text.split())
    print(f"   {snippet[:240]}{'...' if len(snippet) > 240 else ''}")


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    query = " ".join(args.query)
    engine = _open_engine(settings)
    embedder = _load_embedder(settings)
    with Session(engine) as session:
        try:
            with _open_store(settings, embedder) as store:
                results = retrieve(
                    session,
                    embedder,
                    store,
                    query,
                    top_k=settings.retrieval_top_k if args.top_k is None else args.top_k,
                    document_ids=args.document,
                    file_types=args.type,
                    mode=args.mode or settings.search_mode,
                )
        except (ValueError, VectorStoreError, KeywordSearchUnavailable) as exc:
            raise CliError(str(exc)) from exc
        pending = count_pending(session, settings.embedding_model)

    if not results:
        print("no matching notes found")
    for rank, result in enumerate(results, start=1):
        _print_result(rank, result)
    if pending:
        print(f"note: {pending} document(s) are not indexed yet; run `python -m app index`")
    return 0


def _print_answer(answer: Answer) -> None:
    print(answer.text)
    _print_sources(answer.sources)


def _print_sources(sources: Sequence[Source]) -> None:
    if not sources:
        return
    print()
    print("Sources:")
    for source in sources:
        heading = f" > {source.heading_path}" if source.heading_path else ""
        print(
            f"  [{source.marker}] {source.source}{heading}, "
            f"{_where(source)}  (id {source.citation_id})"
        )


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    question = " ".join(args.question)
    engine = _open_engine(settings)
    embedder = _load_embedder(settings)
    with Session(engine) as session:
        try:
            with _open_store(settings, embedder) as store:
                retrieved = retrieve(
                    session,
                    embedder,
                    store,
                    question,
                    top_k=settings.retrieval_top_k if args.top_k is None else args.top_k,
                    document_ids=args.document,
                    file_types=args.type,
                    mode=args.mode or settings.search_mode,
                )
            print("thinking ...", file=sys.stderr, flush=True)
            answer = compose_answer(
                create_llm(settings), question, retrieved, settings.answer_min_score
            )
        except (ValueError, VectorStoreError, LLMError, KeywordSearchUnavailable) as exc:
            raise CliError(str(exc)) from exc
        pending = count_pending(session, settings.embedding_model)

    _print_answer(answer)
    if pending:
        print()
        print(f"note: {pending} document(s) are not indexed yet; run `python -m app index`")
    return 0


def _document_id(session: Session, reference: str) -> int:
    """A document given as its number, or as its file name (names are encrypted, so they are
    compared here and not in the database)."""
    if reference.isdigit():
        return int(reference)
    wanted = reference.casefold()
    found = [
        d.id
        for d in session.scalars(select(Document).where(Document.status == DOC_ACTIVE))
        if d.original_filename.casefold() == wanted
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise CliError(
            f"no current document is named {reference!r}. Use its number: `python -m app search` "
            "shows it in the id of each result (id 4:0 is document 4)."
        )
    raise CliError(
        f"{len(found)} documents are named {reference!r} (numbers {', '.join(map(str, found))}); "
        "use a number"
    )


def _cmd_summarize(args: argparse.Namespace, settings: Settings) -> int:
    """A short summary of one document, written from its own text (no search needed)."""
    engine = _open_engine(settings)
    with Session(engine) as session:
        try:
            document = load_document_text(session, _document_id(session, args.document))
            print("reading ...", file=sys.stderr, flush=True)
            summary = summarize(create_llm(settings), document)
        except (LookupError, ValueError, LLMError) as exc:
            raise CliError(str(exc)) from exc
    print(f"Summary of {summary.name}:")
    print(summary.text)
    if summary.truncated:
        print()
        print(
            f"note: the document is long. Only the first {summary.covered_parts} of its "
            f"{summary.parts} parts were summarized."
        )
    return 0


def _cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    """What two to four documents have in common and where they differ, with citations."""
    engine = _open_engine(settings)
    with Session(engine) as session:
        try:
            documents = [
                load_document_text(session, _document_id(session, ref)) for ref in args.documents
            ]
            print("reading ...", file=sys.stderr, flush=True)
            comparison = compare(create_llm(settings), documents)
        except (LookupError, ValueError, LLMError) as exc:
            raise CliError(str(exc)) from exc
    print(comparison.text)
    if comparison.sources:
        print()
        print("Sources:")
        for entry in comparison.sources:
            print(f"  [{entry.marker}] {entry.name}  (document {entry.document_id})")
    partial = [d.name for d in comparison.documents if d.document_id in comparison.truncated]
    if partial:
        print()
        print(f"note: these are long, so only their start was read: {', '.join(partial)}.")
    return 0


_EXTRACTION_MESSAGES = {
    "no_relevant_notes": "I don't have enough information in your notes to match that.",
    "nothing_found": "No note holds a fact that matches the request.",
    "unreadable": "The model's reply could not be read as a table. Try again or rephrase.",
    "no_valid_row": (
        "None of the rows the model wrote could be checked against your notes, so none is shown."
    ),
}


def _print_extraction(result: Extraction) -> None:
    if not result.rows:
        print(_EXTRACTION_MESSAGES[result.reason])
        return
    width = max(len(row.item) for row in result.rows)
    for row in result.rows:
        print(f"  {row.item:<{width}}  {row.value}  [{row.marker}]")
    _print_sources(result.sources)
    if result.dropped:
        print()
        print(f"note: {result.dropped} row(s) the model wrote were refused (no such note).")


def _cmd_extract(args: argparse.Namespace, settings: Settings) -> int:
    """A table of the facts your notes hold that match a request, each with its source."""
    request = " ".join(args.request)
    engine = _open_engine(settings)
    embedder = _load_embedder(settings)
    with Session(engine) as session:
        try:
            with _open_store(settings, embedder) as store:
                retrieved = retrieve(
                    session,
                    embedder,
                    store,
                    request,
                    top_k=args.top_k or max(settings.retrieval_top_k, 8),
                    document_ids=args.document,
                    file_types=args.type,
                    mode=args.mode or settings.search_mode,
                )
            print("reading ...", file=sys.stderr, flush=True)
            result = extract(create_llm(settings), request, retrieved, settings.answer_min_score)
        except (ValueError, VectorStoreError, LLMError, KeywordSearchUnavailable) as exc:
            raise CliError(str(exc)) from exc
    _print_extraction(result)
    return 0


def _cmd_agent(args: argparse.Namespace, settings: Settings) -> int:
    """Let the assistant carry out a task by itself, within what you allow, and see what it did."""
    engine = _open_engine(settings)

    def load_embedder() -> EmbeddingProvider:
        try:
            return _load_embedder(settings)
        except CliError as exc:
            raise ToolError(str(exc)) from exc

    def retrieve_notes(embedder: EmbeddingProvider, query: str, count: int) -> list[RetrievedChunk]:
        # A tool runs in its own thread, so the search opens its own session.
        with Session(engine) as own, _open_store(settings, embedder) as store:
            return retrieve(own, embedder, store, query, top_k=count, mode=settings.search_mode)

    with Session(engine) as session:
        env = Environment(
            data_dir=settings.data_dir,
            session=session,
            make_llm=lambda: create_llm(settings),
            search=LazySearch(load_embedder, retrieve_notes),
            min_score=settings.answer_min_score,
            progress=lambda text: print(f"  {text}", file=sys.stderr, flush=True),
            input_fn=lambda prompt: input(prompt),
        )
        try:
            return agent_command(
                env,
                args.action,
                args.words,
                run_id=args.run,
                limit=args.limit,
                export=args.export,
                erase=args.erase,
                yes=args.yes,
            )
        except ConsoleError as exc:
            raise CliError(str(exc)) from exc


def _evaluation_inputs(args: argparse.Namespace, settings: Settings) -> tuple[int, int, int, float]:
    chunk_size = args.chunk_size or settings.chunk_size_chars
    chunk_overlap = (
        settings.chunk_overlap_chars if args.chunk_overlap is None else args.chunk_overlap
    )
    top_k = args.top_k or settings.retrieval_top_k
    min_score = settings.answer_min_score if args.min_score is None else args.min_score
    if chunk_size < 100:
        raise CliError("--chunk-size must be at least 100")
    if not 0 <= chunk_overlap < chunk_size:
        raise CliError("--chunk-overlap must be at least 0 and smaller than the chunk size")
    if not 1 <= top_k <= 50:
        raise CliError("--top-k must be between 1 and 50")
    if not 0.0 <= min_score <= 1.0:
        raise CliError("--min-score must be between 0 and 1")
    return chunk_size, chunk_overlap, top_k, min_score


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    chunk_size, chunk_overlap, top_k, min_score = _evaluation_inputs(args, settings)
    try:
        dataset = load_dataset(Path(args.set)) if args.set else load_dataset()
    except DatasetError as exc:
        raise CliError(f"evaluation set problem: {exc}") from exc
    embedder = _load_embedder(settings)

    llm = None
    if args.answers:
        llm = create_llm(settings)
        try:
            llm.list_models()  # fail early, with a clear message, if Ollama is not running
        except LLMError as exc:
            raise CliError(str(exc)) from exc

    note = " and answers (this takes a few minutes)" if llm else ""
    print(f"evaluating retrieval{note} on the built-in test set ...", file=sys.stderr, flush=True)
    try:
        run = run_evaluation(
            embedder,
            dataset,
            llm=llm,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode=args.mode or settings.search_mode,
        )
    except (ValueError, VectorStoreError, KeywordSearchUnavailable) as exc:
        raise CliError(str(exc)) from exc

    print(format_report(run))
    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(to_dict(run), indent=2), encoding="utf-8")
        print(f"\nfull results written to {path}")
    return 0


def _cmd_offline_check(_args: argparse.Namespace, settings: Settings) -> int:
    """Run the whole pipeline with all non-local network access blocked, and report the result."""
    try:
        dataset = load_dataset()
    except DatasetError as exc:
        raise CliError(f"evaluation set problem: {exc}") from exc

    guard = NetworkGuard()
    print("OFFLINE CHECK")
    with guard:
        # The guard must really be on: reaching a public test address has to be refused.
        try:
            socket.create_connection(("203.0.113.1", 80), timeout=1).close()
            raise CliError("the network guard did not engage; cannot prove anything")
        except NetworkBlocked:
            print("  network guard self-test: a connection to 203.0.113.1 was refused (OK)")
        except OSError as exc:
            raise CliError(f"the network guard did not engage ({exc})") from exc
        guard.blocked.clear()
        guard.local.clear()

        embedder = _load_embedder(settings)
        flags = ", ".join(
            f"{k}={os.environ.get(k, 'unset')}" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        print(f"  embedding model loaded from disk ({flags})")

        llm = create_llm(settings)
        try:
            llm.list_models()
        except LLMError as exc:
            raise CliError(str(exc)) from exc

        smoke = {q.id for q in dataset.questions if q.smoke}
        run = run_evaluation(
            embedder,
            dataset,
            llm=llm,
            top_k=settings.retrieval_top_k,
            min_score=settings.answer_min_score,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
            answer_ids=smoke,
        )

    passed = sum(a.passed for a in run.answers)
    print(f"  corpus ingested and indexed: {run.documents} documents, {run.chunks} chunks")
    print(
        f"  searched {len(run.retrieval)} questions "
        f"(right note in the top {run.top_k} for {hit_rate(run.retrieval, run.top_k) * 100:.0f}%)"
    )
    print(
        f"  answered {len(run.answers)} questions with {run.llm_model} "
        f"({passed} correct, informational only)"
    )
    for line in guard.summary().splitlines():
        print(f"  {line}")
    if guard.blocked:
        print("  RESULT: FAIL - something tried to reach outside this computer.")
        return 1
    print("  RESULT: PASS - nothing tried to reach outside this computer.")
    print("  (This watches Python. For the strongest proof, also run it with Wi-Fi off:")
    print("   see docs/offline-check.md.)")
    return 0


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    passphrase = _passphrase_for_unlock(settings, required=False)
    checks = run_doctor(settings, repair=args.fix, quick=args.quick, passphrase=passphrase)
    print(format_checks(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def _default_backup_path() -> Path:
    """A new file under the local application data folder: outside the repository and not synced."""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / "Reyleight" / "backups" / f"reyleight-backup-{stamp}.zip"


def _cmd_backup(args: argparse.Namespace, settings: Settings) -> int:
    destination = Path(args.to) if args.to else _default_backup_path()
    try:
        summary = create_backup(settings.data_dir, destination)
    except BackupError as exc:
        raise CliError(str(exc)) from exc
    print(f"backup saved:   {summary.path}")
    print(f"documents:      {summary.documents}")
    print(f"chunks:         {summary.chunks}")
    print(f"stored files:   {summary.files}")
    print(f"size:           {summary.archive_bytes / (1024 * 1024):.1f} MB")
    print(f"schema version: {summary.schema_revision}")
    for warning in summary.warnings:
        print(f"warning: {warning}")
    print("The search vectors are not saved; after a restore run `python -m app index`.")
    if summary.encrypted:
        print("encryption:     yes (notes, headings and file names are encrypted in this backup)")
        print("Restoring it on another computer needs your recovery passphrase.")
    else:
        print("encryption:     NO. This file contains your notes in readable form.")
        print(
            "Run `python -m app encrypt-library` first to protect your notes, then back up again."
        )
    return 0


def _cmd_restore(args: argparse.Namespace, _settings: Settings) -> int:
    target = Path(args.to)
    try:
        summary = restore_backup(Path(args.archive), target)
    except BackupError as exc:
        raise CliError(str(exc)) from exc
    print(f"restored to:    {summary.target}")
    print(f"documents:      {summary.documents}")
    print(f"chunks:         {summary.chunks}")
    print(f"stored files:   {summary.files}")
    print("checksums verified; nothing is searchable until you run `python -m app index`.")
    if summary.encrypted:
        print("This library is encrypted: the first command that opens it asks for the recovery")
        print("passphrase (unless this Windows account can unlock it).")
    print(f"To use it, set DATA_DIR={summary.target} (in .env or the environment)")
    if summary.needs_migration:
        print("then run `python -m app migrate`: this backup is from an older version.")
    return 0


def _choose_new_passphrase() -> str:
    """Ask for a new recovery passphrase: typed twice, or a generated random key."""
    value = os.environ.get(NEW_PASSPHRASE_ENV)
    if value:
        return value
    if not sys.stdin.isatty():
        raise CliError(
            f"a passphrase is needed but this is not a terminal; set {NEW_PASSPHRASE_ENV}"
        )
    print("Choose a recovery passphrase. It unlocks your library on another computer, or if")
    print("Windows is reinstalled. Without it (and this Windows account) the data is gone.")
    print(f"Use {keystore.MIN_PASSPHRASE_LENGTH}+ characters, or press Enter for a random key.")
    while True:
        first = getpass.getpass("Recovery passphrase (Enter = generate one): ")
        if not first:
            key = keystore.generate_recovery_passphrase()
            print("")
            print("Your recovery key. Write it down and keep it away from this computer:")
            print(f"    {key}")
            again = getpass.getpass("Type the key again to confirm you wrote it down: ")
            if keystore.normalize_passphrase(key) in keystore.passphrase_candidates(again):
                return key
            print("That did not match. Starting again.")
            continue
        try:
            keystore.validate_new_passphrase(first)
        except ValueError as exc:
            print(f"{exc}. Try again.")
            continue
        if getpass.getpass("Type it again: ") != first:
            print("The two did not match. Try again.")
            continue
        return first


def _cmd_encrypt_library(args: argparse.Namespace, settings: Settings) -> int:
    data_dir = settings.data_dir
    try:
        state = keystore.library_state(data_dir)
    except KeystoreError as exc:
        raise CliError(str(exc)) from exc
    if state == "encrypted":
        print("This library is already encrypted.")
        return 0

    if state == "migrating":
        print("A previous encryption was not finished. It will be resumed.")
    else:
        print("This will encrypt your library with AES-256-GCM:")
        print("  - note text, headings and file names in the database")
        print("  - the stored copies of your files")
        print("  - backups made afterwards (they hold only the encrypted forms)")
        print("The search vectors stay readable because search needs them (BitLocker for the")
        print("drive protects those). Stop the API server and any other Reyleight program first.")
        if not args.no_backup:
            print("A backup of the current (unencrypted) library is made first.")
    if not args.yes:
        if not sys.stdin.isatty():
            raise CliError("this changes your library; run it in a terminal, or add --yes")
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing was changed.")
            return 0

    if state == "migrating":
        passphrase = (
            ""
            if keystore.windows_unlock_works(data_dir)
            else _ask_secret("Recovery passphrase: ", PASSPHRASE_ENV)
        )
    else:
        passphrase = _choose_new_passphrase()
    backup_to = None if args.no_backup or state == "migrating" else _default_backup_path()
    try:
        report = encrypt_library(
            data_dir,
            passphrase,
            backup_to=backup_to,
            progress=lambda message: print(f"  {message}"),
        )
    except (SecurityError, ValueError) as exc:
        raise CliError(str(exc)) from exc

    print("")
    print("The library is encrypted.")
    print(f"  values encrypted: {report.values_encrypted}")
    print(f"  stored files encrypted: {report.files_encrypted}")
    for warning in sorted(set(report.warnings)):
        print(f"  warning: {warning}")
    print("Next steps:")
    print("  1. Run `python -m app recovery-check` to confirm your recovery passphrase works.")
    print("  2. Run `python -m app doctor`, then try a search or question.")
    if report.backup_path is not None:
        print(f"  3. Delete the safety backup once you are satisfied: {report.backup_path}")
        print("     (it is the unencrypted original, so it is not protected)")
    print("Data written to disk before today may survive in unused disk space. To overwrite it,")
    print(f"run: cipher /w:{data_dir}   (Windows built-in; it takes a while)")
    return 0


def _cmd_security(_args: argparse.Namespace, settings: Settings) -> int:
    data_dir = settings.data_dir
    try:
        keyfile = keystore.read_keyfile(data_dir)
    except KeystoreError as exc:
        raise CliError(str(exc)) from exc
    if keyfile is None:
        print("encryption:     off (the library is not encrypted)")
        print("Run `python -m app encrypt-library` to encrypt it.")
        return 0
    windows = keystore.windows_unlock_works(data_dir)
    print(f"encryption:     {keyfile.state}")
    print("cipher:         AES-256-GCM (Windows cryptography)")
    print(f"key file:       {data_dir / keystore.SECURITY_FILENAME}")
    print(f"created:        {keyfile.created_at}")
    if windows:
        print("windows unlock: works for this account")
    else:
        print("windows unlock: does NOT work; the recovery passphrase is needed")
    print(f"passphrase kdf: scrypt (n={keyfile.kdf_n}, r={keyfile.kdf_r}, p={keyfile.kdf_p})")
    print("Run `python -m app recovery-check` to confirm the recovery passphrase still works.")
    return 0


def _cmd_recovery_check(_args: argparse.Namespace, settings: Settings) -> int:
    try:
        passphrase = _ask_secret("Recovery passphrase to check: ", PASSPHRASE_ENV)
        result = keystore.check_recovery(settings.data_dir, passphrase)
    except SecurityError as exc:
        raise CliError(str(exc)) from exc
    print(f"recovery passphrase: {'correct' if result.passphrase_works else 'NOT correct'}")
    print(f"windows unlock:      {'works' if result.windows_unlock_works else 'does not work'}")
    if result.same_key_as_windows is not None:
        same = "yes" if result.same_key_as_windows else "NO (something is wrong)"
        print(f"same key as Windows: {same}")
    ok = result.passphrase_works and result.same_key_as_windows is not False
    return 0 if ok else 1


def _cmd_change_passphrase(_args: argparse.Namespace, settings: Settings) -> int:
    data_dir = settings.data_dir
    try:
        if keystore.library_state(data_dir) != "encrypted":
            raise CliError("the library is not encrypted")
        unlocked = keystore.unlock(data_dir, passphrase=_passphrase_for_unlock(settings))
        new_passphrase = _choose_new_passphrase()
        keystore.change_passphrase(data_dir, unlocked.vault, new_passphrase)
    except WrongPassphraseError as exc:
        raise CliError(str(exc)) from exc
    except (SecurityError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    print("The recovery passphrase was changed. The old one no longer works.")
    print("Run `python -m app recovery-check` to confirm the new one, and update any old backups'")
    print("passphrase notes: backups made before today still need the OLD passphrase.")
    return 0


def _cmd_download_model(_args: argparse.Namespace, settings: Settings) -> int:
    target = model_dir_for(settings.models_dir, settings.embedding_model)
    if is_model_downloaded(settings.models_dir, settings.embedding_model):
        print(f"model already downloaded: {target}")
        return 0

    from app.ai.embeddings.download import download_model

    print(f"downloading {settings.embedding_model} (one time, needs internet) ...")
    try:
        download_model(settings.embedding_model, settings.models_dir)
    except Exception as exc:  # network, disk or unknown-model errors: report, don't crash
        raise CliError(f"download failed: {type(exc).__name__}: {exc}") from exc
    print(f"saved to {target}")
    return 0


def _cmd_download_voice_model(_args: argparse.Namespace, settings: Settings) -> int:
    target = model_dir_for(settings.models_dir, settings.speech_model)
    if is_speech_model_downloaded(settings.models_dir, settings.speech_model):
        print(f"speech model already downloaded: {target}")
        return 0

    from app.ai.speech.download import download_speech_model

    print(f"downloading {settings.speech_model} (one time, needs internet) ...")
    try:
        download_speech_model(settings.speech_model, settings.models_dir)
    except Exception as exc:  # network, disk or unknown-model errors: report, don't crash
        raise CliError(f"download failed: {type(exc).__name__}: {exc}") from exc
    print(f"saved to {target}")
    return 0


# --- entry point ------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app", description="Reyleight command line.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run the command with all non-local network access blocked, and report any attempt",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, handler: Handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    serve = add("serve", _cmd_serve, "start the interface and the API (on this computer only)")
    serve.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)"
    )
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    serve.add_argument(
        "--no-migrate", action="store_true", help="do not upgrade the database before starting"
    )
    serve.add_argument(
        "--unsafe-expose-to-network",
        action="store_true",
        help="allow a non-local --host (lets other computers reach your notes; not recommended)",
    )
    add("migrate", _cmd_migrate, "upgrade the database to this version (saves a copy first)")
    add("ingest", _cmd_ingest, "ingest, chunk and index files from your allowed folders")
    add("sync", _cmd_ingest, "same as ingest: also notices edited and deleted files")
    prune_command = add(
        "prune", _cmd_prune, "remove documents whose files are gone (asks before deleting)"
    )
    prune_command.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    prune_command.add_argument(
        "--dry-run", action="store_true", help="only list what would be removed"
    )
    prune_command.add_argument(
        "--superseded",
        action="store_true",
        help="also remove older versions of files that were edited",
    )
    feedback = add(
        "feedback",
        _cmd_feedback,
        "see your marks on answers, or turn the failures into evaluation questions",
    )
    feedback.add_argument(
        "action",
        nargs="?",
        choices=("list", "export"),
        default="list",
        help="list the marks (default), or export the failures as evaluation questions",
    )
    feedback.add_argument(
        "--to",
        metavar="FILE",
        help="where export writes (default: eval-private/feedback-candidates.json)",
    )
    summarize_command = add(
        "summarize", _cmd_summarize, "a short summary of one document (by number or file name)"
    )
    summarize_command.add_argument("document", help="the document's number, or its file name")
    compare_command = add(
        "compare", _cmd_compare, "what two to four documents have in common and where they differ"
    )
    compare_command.add_argument(
        "documents", nargs="+", metavar="document", help="two to four numbers or file names"
    )
    extract_command = add(
        "extract", _cmd_extract, "a table of the facts your notes hold that match a request"
    )
    extract_command.add_argument("request", nargs="+", help="which facts are wanted")
    extract_command.add_argument("--top-k", type=int, help="notes to search (default: at least 8)")
    extract_command.add_argument(
        "--type", action="append", help="only search this file type, e.g. .md (repeatable)"
    )
    extract_command.add_argument(
        "--document", type=int, action="append", help="only search this document number"
    )
    extract_command.add_argument(
        "--mode", choices=("vector", "keyword", "hybrid"), help="how notes are found"
    )
    agent = add(
        "agent",
        _cmd_agent,
        "let the assistant carry out a task by itself, within what you allow",
    )
    agent.add_argument(
        "action",
        choices=ACTIONS,
        help="run a task; status; enable or disable the agent; allow or revoke a tool; "
        "web on or off; log",
    )
    agent.add_argument(
        "words",
        nargs="*",
        help="run: the task. allow, revoke: tool names. web: on or off",
    )
    agent.add_argument("--run", metavar="ID", help="log: show one run")
    agent.add_argument("--limit", type=int, default=40, help="log: how many entries (default 40)")
    agent.add_argument("--export", metavar="FILE", help="log: write every entry to a JSON file")
    agent.add_argument("--erase", action="store_true", help="log: erase the whole log")
    agent.add_argument("--yes", action="store_true", help="log --erase: do not ask to confirm")
    add("rechunk", _cmd_rechunk, "rebuild all chunks (after changing chunk settings)")
    index = add("index", _cmd_index, "embed documents that are not searchable yet")
    index.add_argument(
        "--rebuild", action="store_true", help="drop all vectors and re-embed every document"
    )
    add("status", _cmd_status, "show documents, chunks and search index state")
    search = add("search", _cmd_search, "find the chunks most relevant to a question")
    search.add_argument("query", nargs="+", help="what to look for (quotes are optional)")
    search.add_argument("--top-k", type=int, help="number of results (default: RETRIEVAL_TOP_K)")
    search.add_argument(
        "--type", action="append", metavar="EXT", help="only this file type, e.g. .md (repeatable)"
    )
    search.add_argument(
        "--document", action="append", type=int, metavar="ID", help="only this document id"
    )
    search.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        help="find notes by meaning, by the words in the question, or both (default: SEARCH_MODE)",
    )
    ask = add("ask", _cmd_ask, "answer a question from your notes, with citations")
    ask.add_argument("question", nargs="+", help="your question (quotes are optional)")
    ask.add_argument("--top-k", type=int, help="notes to consider (default: RETRIEVAL_TOP_K)")
    ask.add_argument(
        "--type", action="append", metavar="EXT", help="only this file type, e.g. .md (repeatable)"
    )
    ask.add_argument(
        "--document", action="append", type=int, metavar="ID", help="only this document id"
    )
    ask.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        help="find notes by meaning, by the words in the question, or both (default: SEARCH_MODE)",
    )
    add("check-llm", _cmd_check_llm, "send a test prompt to the local LLM (Ollama)")
    evaluate = add("eval", _cmd_eval, "measure retrieval and answer quality on a built-in test set")
    evaluate.add_argument("--answers", action="store_true", help="also test answers (uses the LLM)")
    evaluate.add_argument("--top-k", type=int, help="notes retrieved per question")
    evaluate.add_argument("--chunk-size", type=int, help="try another chunk size (characters)")
    evaluate.add_argument("--chunk-overlap", type=int, help="try another chunk overlap")
    evaluate.add_argument("--min-score", type=float, help="try another ANSWER_MIN_SCORE")
    evaluate.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        help="search mode for the gate and answers (default: SEARCH_MODE); every mode is compared",
    )
    evaluate.add_argument("--output", metavar="FILE", help="also write full results as JSON")
    evaluate.add_argument(
        "--set",
        metavar="FOLDER",
        help="use your own evaluation set (a folder with corpus/ and questions.json), "
        "for example to test real questions kept outside git",
    )
    add("offline-check", _cmd_offline_check, "run the pipeline with all non-local network blocked")
    encrypt = add(
        "encrypt-library",
        _cmd_encrypt_library,
        "encrypt the library (notes, headings, file names, files)",
    )
    encrypt.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    encrypt.add_argument(
        "--no-backup", action="store_true", help="skip the safety backup (not recommended)"
    )
    add("security", _cmd_security, "show whether the library is encrypted and how it unlocks")
    add("recovery-check", _cmd_recovery_check, "check that your recovery passphrase unlocks it")
    add("change-passphrase", _cmd_change_passphrase, "set a new recovery passphrase")
    doctor = add("doctor", _cmd_doctor, "check the whole setup and say how to fix problems")
    doctor.add_argument(
        "--fix", action="store_true", help="safe repairs only: mark for re-indexing"
    )
    doctor.add_argument("--quick", action="store_true", help="skip verifying stored-file checksums")
    backup = add("backup", _cmd_backup, "save the library (database and files) to one archive")
    backup.add_argument(
        "--to", metavar="FILE", help="where to write the archive (never overwritten)"
    )
    restore = add("restore", _cmd_restore, "restore a backup archive into an empty folder")
    restore.add_argument("archive", help="the backup archive to restore")
    restore.add_argument("--to", metavar="FOLDER", required=True, help="an empty folder")
    add("types", _cmd_types, "list supported file types")
    add("download-model", _cmd_download_model, "download the embedding model (needs internet)")
    add(
        "download-voice-model",
        _cmd_download_voice_model,
        "download the speech model for voice orders (needs internet)",
    )
    return parser


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        # never crash on characters the console cannot show
        sys.stdout.reconfigure(errors="replace")
    try:
        settings = settings or get_settings()
    except ValidationError as exc:
        print("error: invalid settings in .env or the environment:", file=sys.stderr)
        for problem in exc.errors():
            name = ".".join(str(part) for part in problem["loc"]).upper() or "settings"
            print(f"  {name}: {problem['msg']}", file=sys.stderr)
        return 1
    configure_logging(settings.log_level)

    def run() -> int:
        try:
            code: int = args.handler(args, settings)
            return code
        except CliError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except SecurityError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            # Indexing saves its progress after every batch, so nothing is lost.
            print(
                "\ninterrupted. Progress is saved: run the command again to continue.",
                file=sys.stderr,
            )
            return 130

    if not args.offline:
        return run()
    guard = NetworkGuard()
    with guard:
        code = run()
    print(guard.summary(), file=sys.stderr)
    return 1 if guard.blocked and code == 0 else code
