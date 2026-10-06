import hashlib
import logging
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.knowledge.chunking.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.knowledge.chunking.service import chunk_document, has_chunks
from app.knowledge.ingestion.file_types import extract_document
from app.knowledge.ingestion.parsers import ParseError
from app.knowledge.ingestion.scanner import is_within_allowed, resolve_roots, scan_allowed_folders
from app.knowledge.ingestion.sync import LocationTracker, SyncReport, legacy_locations
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
    # Files whose content the owner removed from the library on purpose.
    skipped_excluded: int = 0
    # Extension -> count, for files skipped as an unsupported type.
    unsupported_by_extension: Counter[str] = field(default_factory=Counter)
    folders_missing: int = 0
    # Failure reason code -> count. Reasons never include file names or contents.
    failed: Counter[str] = field(default_factory=Counter)
    # What the scan found out about where documents come from (edits, deletions).
    sync: SyncReport = field(default_factory=SyncReport)


def ingest_folders(
    session: Session,
    data_dir: Path,
    allowed_folders: list[Path],
    max_file_size_bytes: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    excluded_hashes: Collection[str] = (),
    on_file: Callable[[str, Path], None] | None = None,
    legacy_sources: Mapping[str, str] | None = None,
) -> IngestSummary:
    """Ingest supported files from the allow-listed folders into local storage and chunk them.

    Re-ingesting unchanged files stores nothing new (storage dedupes by content hash). A stored
    document that has no chunks yet (e.g. ingested before chunking existed) is chunked now.

    Content whose hash is in `excluded_hashes` is skipped. `on_file(hash, path)` is called for
    every file that is stored or already stored.

    Where each file was found is recorded (see sync.py): an edited file replaces its old version,
    and a file that stays absent from two scans in a row marks its document as missing.
    `legacy_sources` (hash -> path) lets documents read before locations existed be tracked too.
    """
    summary = IngestSummary()
    scan = scan_allowed_folders(allowed_folders)
    summary.folders_missing = scan.folders_missing
    summary.skipped_unsupported = scan.skipped_unsupported
    summary.unsupported_by_extension = scan.unsupported_by_extension
    summary.skipped_outside_allowlist = scan.skipped_outside_allowlist

    roots = resolve_roots(allowed_folders)
    tracker = LocationTracker(session, roots)

    for path in scan.files:
        # Re-check at read time, independent of the scanner.
        if not is_within_allowed(path, roots):
            summary.skipped_outside_allowlist += 1
            continue
        tracker.saw(path)  # it exists, whatever becomes of its content below

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
            text = extract_document(path.suffix, data).text
        except ParseError as exc:
            summary.failed[str(exc)] += 1
            continue
        if not text.strip():
            # e.g. a whitespace-only note, or an HTML page with no visible text
            summary.skipped_empty += 1
            continue

        digest = hashlib.sha256(data).hexdigest()
        if digest in excluded_hashes:
            summary.skipped_excluded += 1
            continue

        document, created = store_file(session, data_dir, data, path.name)
        tracker.attach(path, document)
        if on_file is not None:
            on_file(digest, path)
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

    summary.sync = tracker.finish(
        {str(root) for root in roots if root.is_dir()},
        legacy_locations(session, legacy_sources or {}),
    )

    logger.info(
        "ingest finished added=%d unchanged=%d chunks=%d failed=%d",
        summary.added,
        summary.unchanged,
        summary.chunks_created,
        sum(summary.failed.values()),
    )
    return summary
