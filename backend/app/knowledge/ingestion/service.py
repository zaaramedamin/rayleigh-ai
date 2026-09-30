import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.knowledge.chunking.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.knowledge.chunking.service import chunk_document, has_chunks
from app.knowledge.ingestion.file_types import extract_text
from app.knowledge.ingestion.parsers import ParseError
from app.knowledge.ingestion.scanner import is_within_allowed, resolve_roots, scan_allowed_folders
from app.storage.files import UnsafePathError, store_file

logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    added: int = 0
    unchanged: int = 0
    chunks_created: int = 0
    skipped_empty: int = 0
    skipped_too_large: int = 0
    skipped_unsupported: int = 0
    skipped_outside_allowlist: int = 0
    # Extension -> count, for files skipped as an unsupported type.
    unsupported_by_extension: Counter[str] = field(default_factory=Counter)
    folders_missing: int = 0
    # Failure reason code -> count. Reasons never include file names or contents.
    failed: Counter[str] = field(default_factory=Counter)


def ingest_folders(
    session: Session,
    data_dir: Path,
    allowed_folders: list[Path],
    max_file_size_bytes: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestSummary:
    """Ingest supported files from the allow-listed folders into local storage and chunk them.

    Re-ingesting unchanged files stores nothing new (storage dedupes by content hash). A stored
    document that has no chunks yet (e.g. ingested before chunking existed) is chunked now.
    """
    summary = IngestSummary()
    scan = scan_allowed_folders(allowed_folders)
    summary.folders_missing = scan.folders_missing
    summary.skipped_unsupported = scan.skipped_unsupported
    summary.unsupported_by_extension = scan.unsupported_by_extension
    summary.skipped_outside_allowlist = scan.skipped_outside_allowlist

    roots = resolve_roots(allowed_folders)

    for path in scan.files:
        # Re-check at read time, independent of the scanner.
        if not is_within_allowed(path, roots):
            summary.skipped_outside_allowlist += 1
            continue

        try:
            size = path.stat().st_size
            if size == 0:
                summary.skipped_empty += 1
                continue
            if size > max_file_size_bytes:
                summary.skipped_too_large += 1
                continue
            data = path.read_bytes()
        except OSError:
            summary.failed["unreadable"] += 1
            continue

        try:
            text = extract_text(path.suffix, data)
        except ParseError as exc:
            summary.failed[str(exc)] += 1
            continue
        if not text.strip():
            # e.g. a whitespace-only note, or an HTML page with no visible text
            summary.skipped_empty += 1
            continue

        document, created = store_file(session, data_dir, data, path.name)
        if created:
            summary.added += 1
        else:
            summary.unchanged += 1

        if created or not has_chunks(session, document):
            try:
                summary.chunks_created += chunk_document(
                    session, data_dir, document, chunk_size, chunk_overlap
                )
            except (ParseError, UnsafePathError, OSError):
                summary.failed["chunking"] += 1

    logger.info(
        "ingest finished added=%d unchanged=%d chunks=%d failed=%d",
        summary.added,
        summary.unchanged,
        summary.chunks_created,
        sum(summary.failed.values()),
    )
    return summary
