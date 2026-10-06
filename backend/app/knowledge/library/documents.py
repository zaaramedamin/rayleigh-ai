"""Listing and removing documents."""

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.assistant.conversations import forget_document_passages
from app.knowledge.library.state import UpdateState, load_state
from app.storage.files import UnsafePathError, resolve_inside
from app.storage.models import (
    DOC_ACTIVE,
    DOC_SUPERSEDED,
    LOCATION_PRESENT,
    Chunk,
    Document,
    DocumentSource,
)
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    name: str
    size_bytes: int
    media_type: str
    created_at: datetime
    chunks: int
    searchable: bool
    source: str | None
    status: str
    kind: str


@dataclass
class Removed:
    """What removing documents took away, as counts."""

    documents: int = 0
    chunks: int = 0
    stored_files: int = 0

    def add(self, other: "Removed") -> None:
        self.documents += other.documents
        self.chunks += other.chunks
        self.stored_files += other.stored_files


def _native(source_root: str, source_path: str) -> str:
    return str(Path(source_root, *source_path.split("/")))


def _best_location(rows: Sequence[DocumentSource]) -> str | None:
    """Where to say a document came from: a place it is still found, else the last one."""
    if not rows:
        return None
    present = [row for row in rows if row.status == LOCATION_PRESENT]
    pick = max(present or rows, key=lambda row: row.last_seen_at)
    return _native(pick.source_root, pick.source_path)


def _locations_by_document(session: Session) -> dict[int, list[DocumentSource]]:
    found: dict[int, list[DocumentSource]] = {}
    for row in session.scalars(select(DocumentSource)):
        found.setdefault(row.document_id, []).append(row)
    return found


def list_documents(
    session: Session, data_dir: Path, embedding_model: str, *, include_history: bool = False
) -> list[DocumentInfo]:
    """Every document except replaced versions: chunk count, searchable, status, and its source.

    A document read before locations were recorded falls back to the path library.json kept.
    """
    counts = {
        row[0]: row[1]
        for row in session.execute(
            select(Chunk.document_id, func.count()).group_by(Chunk.document_id)
        )
    }
    legacy = load_state(data_dir).sources
    locations = _locations_by_document(session)
    query = select(Document)
    if not include_history:
        query = query.where(Document.status != DOC_SUPERSEDED)
    documents = session.scalars(query).all()
    infos = [
        DocumentInfo(
            id=d.id,
            name=d.original_filename,
            size_bytes=d.size_bytes,
            media_type=d.media_type,
            created_at=d.created_at,
            chunks=counts.get(d.id, 0),
            searchable=d.status == DOC_ACTIVE and d.indexed_model == embedding_model,
            source=_best_location(locations.get(d.id, [])) or legacy.get(d.content_hash),
            status=d.status,
            kind=d.kind,
        )
        for d in documents
    ]
    return sorted(infos, key=lambda info: info.name.lower())


@dataclass(frozen=True)
class LocationInfo:
    folder: str
    path: str  # inside the folder, with forward slashes
    status: str
    last_seen_at: datetime
    misses: int


@dataclass(frozen=True)
class VersionInfo:
    id: int
    created_at: datetime
    size_bytes: int
    chunks: int


@dataclass(frozen=True)
class DocumentDetails:
    info: DocumentInfo
    supersedes_id: int | None
    locations: list[LocationInfo]
    older_versions: list[VersionInfo]
    indexed_chunks: int


def document_details(
    session: Session, data_dir: Path, document_id: int, embedding_model: str
) -> DocumentDetails | None:
    """Everything the library knows about one document: where it was found, and its history."""
    document = session.get(Document, document_id)
    if document is None:
        return None
    infos = [
        info
        for info in list_documents(session, data_dir, embedding_model, include_history=True)
        if info.id == document_id
    ]
    if not infos:
        return None
    locations = sorted(
        (
            LocationInfo(
                folder=row.source_root,
                path=row.source_path,
                status=row.status,
                last_seen_at=row.last_seen_at,
                misses=row.misses,
            )
            for row in session.scalars(
                select(DocumentSource).where(DocumentSource.document_id == document_id)
            )
        ),
        key=lambda location: (location.folder.lower(), location.path.lower()),
    )
    older: list[VersionInfo] = []
    seen = {document_id}
    previous_id = document.supersedes_id
    while previous_id is not None and previous_id not in seen:
        seen.add(previous_id)
        previous = session.get(Document, previous_id)
        if previous is None:
            break
        chunk_count = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == previous.id)
        )
        older.append(
            VersionInfo(previous.id, previous.created_at, previous.size_bytes, chunk_count or 0)
        )
        previous_id = previous.supersedes_id
    indexed = session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.document_id == document_id, Chunk.indexed_model == embedding_model)
    )
    return DocumentDetails(infos[0], document.supersedes_id, locations, older, indexed or 0)


def _under(path: Path, folder: Path) -> bool:
    return path == folder or path.is_relative_to(folder)


def _norm(path: Path) -> Path:
    return Path(os.path.normcase(path))


def documents_per_folder(
    session: Session, data_dir: Path, folders: Sequence[Path]
) -> dict[Path, int]:
    """How many documents (not counting replaced versions) come from each folder."""
    statuses: Mapping[int, str] = dict(session.execute(select(Document.id, Document.status)).all())
    hashes = dict(session.execute(select(Document.id, Document.content_hash)).all())
    locations = _locations_by_document(session)
    legacy = load_state(data_dir).sources
    result: dict[Path, int] = {}
    for folder in folders:
        target = _norm(folder.resolve())
        ids: set[int] = set()
        for document_id, rows in locations.items():
            if statuses.get(document_id) == DOC_SUPERSEDED:
                continue
            if any(_under(_norm(Path(row.source_root)), target) for row in rows):
                ids.add(document_id)
        for document_id, content_hash in hashes.items():
            if document_id in locations or statuses.get(document_id) == DOC_SUPERSEDED:
                continue
            source = legacy.get(content_hash)
            if source is not None and _under(_norm(Path(source)), target):
                ids.add(document_id)
        result[folder] = len(ids)
    return result


def documents_from_folder(session: Session, data_dir: Path, folder: Path) -> list[Document]:
    """Documents that were found inside `folder`. Removing one also removes its older versions."""
    target = _norm(folder.resolve())
    legacy = load_state(data_dir).sources
    locations = _locations_by_document(session)
    ids = {
        document_id
        for document_id, rows in locations.items()
        if any(_under(_norm(Path(row.source_root)), target) for row in rows)
    }
    legacy_hashes = {
        content_hash
        for content_hash, source in legacy.items()
        if _under(_norm(Path(source)), target)
    }
    if legacy_hashes:
        query = select(Document.id).where(Document.content_hash.in_(legacy_hashes))
        if locations:
            query = query.where(Document.id.not_in(list(locations)))
        ids.update(session.scalars(query).all())
    found = (session.get(Document, document_id) for document_id in sorted(ids))
    return [document for document in found if document is not None]


def _remove_one(
    session: Session, data_dir: Path, document: Document, store: VectorStore | None
) -> Removed:
    document_id, stored_path = document.id, document.stored_path
    if store is not None:
        store.delete_document(document_id)
    chunk_count = (
        session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        or 0
    )
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    session.execute(delete(DocumentSource).where(DocumentSource.document_id == document_id))
    session.delete(document)
    # What a stored conversation kept of this document goes with it.
    forget_document_passages(session, [document_id])
    session.commit()

    removed_file = 0
    try:
        target = resolve_inside(data_dir, stored_path)
        if target.exists():
            target.unlink()
            removed_file = 1
    except (OSError, UnsafePathError):
        logger.warning("stored copy of a removed document could not be deleted")
    return Removed(documents=1, chunks=chunk_count, stored_files=removed_file)


def delete_document(
    session: Session,
    data_dir: Path,
    document: Document,
    *,
    exclude: bool,
    store: VectorStore | None = None,
) -> Removed:
    """Remove a document, its chunks, its vectors (if a store is given) and its stored copy.

    Older versions that this document replaced go with it: removing a note must not leave its
    earlier text behind. The original file in the folder you chose is never touched. With
    `exclude`, later scans skip that content, so removing a document from the library really keeps
    it out.
    """
    content_hash = document.content_hash
    older: list[Document] = []
    current = document.supersedes_id
    while current is not None:
        previous = session.get(Document, current)
        if previous is None or previous.status != DOC_SUPERSEDED or previous in older:
            break
        older.append(previous)
        current = previous.supersedes_id

    removed = _remove_one(session, data_dir, document, store)
    for previous in older:
        removed.add(_remove_one(session, data_dir, previous, store))

    with UpdateState(data_dir) as state:
        state.sources.pop(content_hash, None)
        if exclude and content_hash not in state.excluded:
            state.excluded.append(content_hash)
    return removed


def clear_exclusions(data_dir: Path) -> int:
    """Forget which documents were removed, so the next scan may add them again."""
    with UpdateState(data_dir) as state:
        count = len(state.excluded)
        state.excluded = []
    return count


def clear_legacy_sources(data_dir: Path) -> None:
    """Forget the paths library.json kept before locations lived in the database.

    They are plain text, so they are dropped once a sync has moved them into the database, where
    they are encrypted with the rest of the library.
    """
    with UpdateState(data_dir) as state:
        state.sources = {}
