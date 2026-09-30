"""Command-line interface: `python -m app <command>`. Run `python -m app --help` for the list."""

import argparse
import sys
from collections.abc import Callable

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.ai.embeddings.base import is_model_downloaded, model_dir_for
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.knowledge.chunking.service import rechunk_all
from app.knowledge.ingestion.file_types import FILE_TYPES
from app.knowledge.ingestion.service import IngestSummary, ingest_folders
from app.storage.database import create_db_engine
from app.storage.migrations import database_is_up_to_date


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
    return 0


def _cmd_rechunk(_args: argparse.Namespace, settings: Settings) -> int:
    engine = _open_engine(settings)
    with Session(engine) as session:
        total = rechunk_all(
            session, settings.data_dir, settings.chunk_size_chars, settings.chunk_overlap_chars
        )
    print(f"rechunked all stored documents: {total} chunks")
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
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, handler: Handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("ingest", _cmd_ingest, "ingest and chunk files from ALLOWED_FOLDERS")
    add(
        "rechunk", _cmd_rechunk, "rebuild chunks for all stored documents (after changing settings)"
    )
    add("types", _cmd_types, "list supported file types")
    add("download-model", _cmd_download_model, "download the embedding model (needs internet)")
    return parser


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")  # never crash on characters the console lacks
    except (AttributeError, ValueError):
        pass
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    try:
        return args.handler(args, settings)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
