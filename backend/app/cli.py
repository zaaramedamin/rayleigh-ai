"""Command-line interface: `python -m app <command>`. Run `python -m app --help` for the list."""

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.ai.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRuntimeError,
    ModelNotAvailableError,
    is_model_downloaded,
    model_dir_for,
)
from app.ai.llm.base import LLMError, LLMUnavailableError
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.network_guard import NetworkBlocked, NetworkGuard
from app.evaluation.report import format_report, to_dict
from app.evaluation.runner import hit_rate, run_evaluation
from app.knowledge.answering.service import Answer, compose_answer
from app.knowledge.chunking.service import rechunk_all
from app.knowledge.components import (
    create_llm,
    load_embedder,
    open_vector_store,
    vector_store_path,
)
from app.knowledge.indexing.service import (
    IndexSummary,
    count_pending,
    index_pending,
    reset_index,
)
from app.knowledge.ingestion.file_types import FILE_TYPES
from app.knowledge.ingestion.service import IngestSummary, ingest_folders
from app.knowledge.retrieval.service import RetrievedChunk, retrieve
from app.storage.database import create_db_engine
from app.storage.migrations import database_is_up_to_date
from app.storage.models import Chunk, Document
from app.storage.vector_store import VectorStoreError, collection_name, count_local_vectors


class CliError(Exception):
    """A problem reported to the user as one line, with exit code 1."""


Handler = Callable[[argparse.Namespace, Settings], int]


def _open_engine(settings: Settings) -> Engine:
    engine = create_db_engine(settings.data_dir)
    if not database_is_up_to_date(engine):
        raise CliError(
            "database missing or out of date. Run `alembic upgrade head` from backend/ first."
        )
    return engine


def _load_embedder(settings: Settings) -> EmbeddingProvider:
    try:
        return load_embedder(settings.embedding_model, settings.models_dir)
    except (ModelNotAvailableError, EmbeddingRuntimeError) as exc:
        raise CliError(str(exc)) from exc


def _run_index(session: Session, settings: Settings, *, rebuild: bool) -> IndexSummary:
    embedder = _load_embedder(settings)
    try:
        with open_vector_store(settings, embedder) as store:
            if rebuild:
                reset_index(session, store)
            return index_pending(session, embedder, store)
    except VectorStoreError as exc:
        raise CliError(str(exc)) from exc


def _index_after_changes(session: Session, settings: Settings) -> None:
    """Make new or re-chunked documents searchable, when the model is available."""
    pending = count_pending(session, settings.embedding_model)
    if pending == 0:
        print("search index:              up to date")
        return
    if not is_model_downloaded(settings.models_dir, settings.embedding_model):
        print(
            f"search index:              {pending} document(s) not searchable yet; run "
            "`python -m app download-model`, then `python -m app index`"
        )
        return
    summary = _run_index(session, settings, rebuild=False)
    print(
        f"search index:              {summary.documents_indexed} document(s) indexed, "
        f"{summary.chunks_embedded} chunks embedded"
    )


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


# --- commands ---------------------------------------------------------------------------------


def _cmd_types(_args: argparse.Namespace, _settings: Settings) -> int:
    print("Supported file types:")
    for file_type in FILE_TYPES:
        extensions = " ".join(file_type.extensions)
        print(f"  {file_type.name:<18} {extensions:<16} {file_type.description}")
    return 0


def _cmd_ingest(_args: argparse.Namespace, settings: Settings) -> int:
    if not settings.allowed_folders:
        print("ALLOWED_FOLDERS is empty. Nothing to ingest. Set it in .env.")
        return 0
    engine = _open_engine(settings)
    with Session(engine) as session:
        summary = ingest_folders(
            session,
            settings.data_dir,
            settings.allowed_folders,
            settings.max_file_size_mb * 1024 * 1024,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
        )
        _print_ingest_summary(summary)
        _index_after_changes(session, settings)
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
        documents = session.scalar(select(func.count()).select_from(Document)) or 0
        chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
        pending = count_pending(session, settings.embedding_model)
    downloaded = is_model_downloaded(settings.models_dir, settings.embedding_model)
    try:
        vectors: int | str = count_local_vectors(
            vector_store_path(settings), collection_name(settings.embedding_model)
        )
    except VectorStoreError as exc:
        vectors = f"unknown: {exc}"

    print(f"embedding model: {settings.embedding_model}")
    print(f"  downloaded:    {'yes' if downloaded else 'no (run `python -m app download-model`)'}")
    print(f"documents:       {documents}")
    print(f"chunks:          {chunks}")
    print(f"searchable:      {documents - pending} of {documents} documents")
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


def _print_result(rank: int, result: RetrievedChunk) -> None:
    heading = f"  [{result.heading_path}]" if result.heading_path else ""
    print(
        f"{rank}. score {result.score:.3f}  {result.source}{heading}  "
        f"lines {result.start_line}-{result.end_line}  (id {result.citation_id})"
    )
    snippet = " ".join(result.text.split())
    print(f"   {snippet[:240]}{'...' if len(snippet) > 240 else ''}")


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    query = " ".join(args.query)
    engine = _open_engine(settings)
    embedder = _load_embedder(settings)
    with Session(engine) as session:
        try:
            with open_vector_store(settings, embedder) as store:
                results = retrieve(
                    session,
                    embedder,
                    store,
                    query,
                    top_k=settings.retrieval_top_k if args.top_k is None else args.top_k,
                    document_ids=args.document,
                    file_types=args.type,
                )
        except (ValueError, VectorStoreError) as exc:
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
    if not answer.sources:
        return
    print()
    print("Sources:")
    for source in answer.sources:
        heading = f" > {source.heading_path}" if source.heading_path else ""
        print(
            f"  [{source.marker}] {source.source}{heading}, "
            f"lines {source.start_line}-{source.end_line}  (id {source.citation_id})"
        )


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    question = " ".join(args.question)
    engine = _open_engine(settings)
    embedder = _load_embedder(settings)
    with Session(engine) as session:
        try:
            with open_vector_store(settings, embedder) as store:
                retrieved = retrieve(
                    session,
                    embedder,
                    store,
                    question,
                    top_k=settings.retrieval_top_k if args.top_k is None else args.top_k,
                    document_ids=args.document,
                    file_types=args.type,
                )
            print("thinking ...", file=sys.stderr, flush=True)
            answer = compose_answer(
                create_llm(settings), question, retrieved, settings.answer_min_score
            )
        except (ValueError, VectorStoreError, LLMError) as exc:
            raise CliError(str(exc)) from exc
        pending = count_pending(session, settings.embedding_model)

    _print_answer(answer)
    if pending:
        print()
        print(f"note: {pending} document(s) are not indexed yet; run `python -m app index`")
    return 0


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
        dataset = load_dataset()
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
        )
    except (ValueError, VectorStoreError) as exc:
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

    add("ingest", _cmd_ingest, "ingest, chunk and index files from ALLOWED_FOLDERS")
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
    ask = add("ask", _cmd_ask, "answer a question from your notes, with citations")
    ask.add_argument("question", nargs="+", help="your question (quotes are optional)")
    ask.add_argument("--top-k", type=int, help="notes to consider (default: RETRIEVAL_TOP_K)")
    ask.add_argument(
        "--type", action="append", metavar="EXT", help="only this file type, e.g. .md (repeatable)"
    )
    ask.add_argument(
        "--document", action="append", type=int, metavar="ID", help="only this document id"
    )
    add("check-llm", _cmd_check_llm, "send a test prompt to the local LLM (Ollama)")
    evaluate = add("eval", _cmd_eval, "measure retrieval and answer quality on a built-in test set")
    evaluate.add_argument("--answers", action="store_true", help="also test answers (uses the LLM)")
    evaluate.add_argument("--top-k", type=int, help="notes retrieved per question")
    evaluate.add_argument("--chunk-size", type=int, help="try another chunk size (characters)")
    evaluate.add_argument("--chunk-overlap", type=int, help="try another chunk overlap")
    evaluate.add_argument("--min-score", type=float, help="try another ANSWER_MIN_SCORE")
    evaluate.add_argument("--output", metavar="FILE", help="also write full results as JSON")
    add("offline-check", _cmd_offline_check, "run the pipeline with all non-local network blocked")
    add("types", _cmd_types, "list supported file types")
    add("download-model", _cmd_download_model, "download the embedding model (needs internet)")
    return parser


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")  # never crash on characters the console lacks
    except (AttributeError, ValueError):
        pass
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
            return args.handler(args, settings)
        except CliError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if not args.offline:
        return run()
    guard = NetworkGuard()
    with guard:
        code = run()
    print(guard.summary(), file=sys.stderr)
    return 1 if guard.blocked and code == 0 else code
