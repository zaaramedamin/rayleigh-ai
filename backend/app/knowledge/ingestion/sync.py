"""Keep the library in step with the folders it was read from.

Every file found during a scan is recorded as a *location* (an allowed folder plus a path inside
it). From those locations the library can tell:

- an edit: the same location now holds different content. The new content becomes a new document
  and the old one is *superseded* (kept as history, never searched);
- a rename or a copy: the same content at a second location. One document, two locations;
- a deletion: a location that no scan finds. It is only called *missing* after two syncs in a row
  failed to find it, so an offline cloud folder or an unplugged drive does not look like a deletion,
  and a document is *missing* only when none of its locations is left.

Nothing is ever deleted here. Removing documents is `prune`, which asks first.

Locations are encrypted like file names, so they cannot be looked up in SQL. They are loaded once
per sync and compared in Python.
"""

import logging
import os
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import (
    DOC_ACTIVE,
    DOC_MISSING,
    DOC_SUPERSEDED,
    LOCATION_MISSING,
    LOCATION_PRESENT,
    Document,
    DocumentSource,
)

logger = logging.getLogger(__name__)

# A file must be absent from this many consecutive syncs before its location counts as missing.
MISSES_BEFORE_MISSING = 2
# If at least this many files vanish from one folder at once, and they are more than this share of
# what was there, the folder is probably unavailable rather than emptied. Nothing is judged then.
MASS_DISAPPEARANCE_MIN = 5
MASS_DISAPPEARANCE_SHARE = 0.5


@dataclass
class SyncReport:
    """What a sync found out about the library, as counts. Never names or paths."""

    locations_added: int = 0
    superseded: int = 0  # documents replaced by a newer version of their file
    reactivated: int = 0  # older versions or missing documents whose file came back
    newly_missing: int = 0  # documents whose every location is now gone
    paused_folders: int = 0  # folders where too many files vanished at once, so none was judged
    # Documents replaced in this sync: their vectors still have to be removed.
    superseded_ids: list[int] = field(default_factory=list)


def _key(root: str, relative: str) -> tuple[str, str]:
    """Compare locations the way the file system does (Windows ignores case)."""
    return os.path.normcase(root), os.path.normcase(relative)


class LocationTracker:
    """Records, during one scan, where each file was found, and settles the library afterwards."""

    def __init__(self, session: Session, roots: Sequence[Path]) -> None:
        self._session = session
        # The deepest folder first, so a file belongs to the most specific allowed folder.
        self._roots = sorted(set(roots), key=lambda root: len(root.parts), reverse=True)
        self._rows: dict[tuple[str, str], DocumentSource] = {
            _key(row.source_root, row.source_path): row
            for row in session.scalars(select(DocumentSource))
        }
        self._seen: set[tuple[str, str]] = set()
        self._replaced: dict[int, None] = {}  # old document ids, in order, without repeats
        self._now = datetime.now(UTC)
        self.report = SyncReport()

    def _locate(self, path: Path) -> tuple[str, str] | None:
        for root in self._roots:
            if path.is_relative_to(root):
                return str(root), path.relative_to(root).as_posix()
        return None

    def saw(self, path: Path) -> None:
        """The file exists, whatever becomes of its content (it may be empty or unreadable)."""
        located = self._locate(path)
        if located is not None and _key(*located) in self._rows:
            self._seen.add(_key(*located))

    def attach(self, path: Path, document: Document) -> None:
        """The file at `path` holds the content of `document`."""
        located = self._locate(path)
        if located is None:
            return
        key = _key(*located)
        row = self._rows.get(key)
        if row is None:
            row = DocumentSource(
                document_id=document.id,
                source_root=located[0],
                source_path=located[1],
                first_seen_at=self._now,
                last_seen_at=self._now,
                status=LOCATION_PRESENT,
                misses=0,
            )
            self._session.add(row)
            self._rows[key] = row
            self.report.locations_added += 1
        else:
            if row.document_id != document.id:
                # An edit: this place now holds different content.
                self._replaced[row.document_id] = None
                if document.supersedes_id is None:
                    document.supersedes_id = row.document_id
                row.document_id = document.id
            row.status = LOCATION_PRESENT
            row.misses = 0
            row.missing_since = None
            row.last_seen_at = self._now
        self._seen.add(key)
        if document.status != DOC_ACTIVE:
            # An old version came back (the file was changed back) or a missing file returned.
            document.status = DOC_ACTIVE
            self.report.reactivated += 1

    def _backfill_legacy(self, legacy_by_document: Mapping[int, str]) -> None:
        """Give documents read before locations existed a location from the old bookkeeping."""
        has_location = {row.document_id for row in self._rows.values()}
        for document_id, text in legacy_by_document.items():
            if document_id in has_location:
                continue
            located = self._locate(Path(text))
            if located is None:
                continue  # its folder is no longer allowed: the folder removal handles those
            row = DocumentSource(
                document_id=document_id,
                source_root=located[0],
                source_path=located[1],
                first_seen_at=self._now,
                last_seen_at=self._now,
                status=LOCATION_PRESENT,  # set now: column defaults only apply when saved
                misses=0,
            )
            self._session.add(row)
            self._rows[_key(*located)] = row

    def finish(
        self, usable_roots: Collection[str], legacy_by_document: Mapping[int, str] | None = None
    ) -> SyncReport:
        """Judge what was not found, retire replaced versions and save.

        `usable_roots` are the folders that exist and were really scanned: a folder that could not
        be read is not judged.
        """
        if legacy_by_document:
            self._backfill_legacy(legacy_by_document)

        usable = {os.path.normcase(root) for root in usable_roots}
        by_root: dict[str, list[tuple[tuple[str, str], DocumentSource]]] = defaultdict(list)
        for key, row in self._rows.items():
            by_root[key[0]].append((key, row))

        judged_documents: set[int] = set()
        for root_key, entries in by_root.items():
            if root_key not in usable:
                continue
            present = [(k, r) for k, r in entries if r.status == LOCATION_PRESENT]
            unseen = [(k, r) for k, r in present if k not in self._seen]
            if len(unseen) >= MASS_DISAPPEARANCE_MIN and len(
                unseen
            ) > MASS_DISAPPEARANCE_SHARE * len(present):
                self.report.paused_folders += 1
                logger.warning("many files vanished from one folder at once; none was judged")
                continue
            for _location, row in unseen:
                row.misses += 1
                if row.misses >= MISSES_BEFORE_MISSING:
                    row.status = LOCATION_MISSING
                    row.missing_since = self._now
                    judged_documents.add(row.document_id)

        present_by_document: Counter[int] = Counter(
            row.document_id for row in self._rows.values() if row.status == LOCATION_PRESENT
        )
        for document_id in judged_documents:
            document = self._session.get(Document, document_id)
            if (
                document is not None
                and document.status == DOC_ACTIVE
                and present_by_document[document_id] == 0
            ):
                document.status = DOC_MISSING
                self.report.newly_missing += 1
        for document_id in self._replaced:
            document = self._session.get(Document, document_id)
            if (
                document is not None
                and document.status == DOC_ACTIVE
                and present_by_document[document_id] == 0
            ):
                document.status = DOC_SUPERSEDED
                self.report.superseded += 1
                self.report.superseded_ids.append(document_id)

        self._session.commit()
        return self.report


def legacy_locations(session: Session, sources_by_hash: Mapping[str, str]) -> dict[int, str]:
    """Map document id -> the path library.json remembered for it, for documents still active."""
    if not sources_by_hash:
        return {}
    rows = session.execute(
        select(Document.id, Document.content_hash).where(Document.status == DOC_ACTIVE)
    ).all()
    return {
        document_id: sources_by_hash[content_hash]
        for document_id, content_hash in rows
        if content_hash in sources_by_hash
    }
