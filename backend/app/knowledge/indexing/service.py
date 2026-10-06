"""Embed chunks and store their vectors, so documents become searchable."""

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import ColumnElement, and_, func, or_, select, update
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.storage.models import DOC_ACTIVE, DOC_MISSING, DOC_SUPERSEDED, Chunk, Document
from app.storage.vector_store import VectorPoint, VectorStore

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class IndexProgress:
    """How far an index run has come. Counts and times only."""

    documents_done: int
    documents_total: int
    chunks_done: int
    chunks_total: int
    seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.chunks_done / self.seconds if self.seconds > 0 else 0.0

    @property
    def seconds_left(self) -> float | None:
        """A rough estimate, from the speed so far. None until there is a speed."""
        rate = self.chunks_per_second
        return (self.chunks_total - self.chunks_done) / rate if rate > 0 else None


ProgressCallback = Callable[[IndexProgress], None]


@dataclass
class IndexSummary:
    documents_indexed: int = 0
    chunks_embedded: int = 0
    # Documents that are no longer current (replaced, or their file is gone) and lost their vectors.
    documents_purged: int = 0


def text_hash(text: str) -> str:
    """Short fingerprint of a chunk's text, stored next to its vector to detect stale vectors."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def embedding_input(chunk: Chunk) -> str:
    """Text given to the model. The heading path adds context to chunks deep inside a section."""
    return f"{chunk.heading_path}\n\n{chunk.text}" if chunk.heading_path else chunk.text


def _pending(model_name: str) -> ColumnElement[bool]:
    # Only the current version of a file is searched, so only those documents get vectors.
    return and_(
        Document.status == DOC_ACTIVE,
        or_(Document.indexed_model.is_(None), Document.indexed_model != model_name),
    )


def _stale() -> ColumnElement[bool]:
    return and_(Document.status != DOC_ACTIVE, Document.indexed_model.is_not(None))


def count_pending(session: Session, model_name: str) -> int:
    """Current documents whose vectors are missing, stale, or were made by a different model."""
    query = select(func.count()).select_from(Document).where(_pending(model_name))
    return session.scalar(query) or 0


@dataclass(frozen=True)
class LibraryCounts:
    """How many documents the library holds, by what becomes of them. Counts only."""

    active: int  # current versions of files that are still found: these are searched
    missing: int  # their file was not found in two updates in a row
    superseded: int  # older versions, kept as history
    chunks: int  # of active documents
    pending: int  # active documents that are not searchable yet

    @property
    def searchable(self) -> int:
        return self.active - self.pending


def library_counts(session: Session, model_name: str) -> LibraryCounts:
    by_status = dict(
        session.execute(select(Document.status, func.count()).group_by(Document.status)).all()
    )
    chunks = (
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == DOC_ACTIVE)
        )
        or 0
    )
    return LibraryCounts(
        active=by_status.get(DOC_ACTIVE, 0),
        missing=by_status.get(DOC_MISSING, 0),
        superseded=by_status.get(DOC_SUPERSEDED, 0),
        chunks=chunks,
        pending=count_pending(session, model_name),
    )


def count_stale_vectors(session: Session) -> int:
    """Documents that are no longer current but may still have vectors in the store."""
    return session.scalar(select(func.count()).select_from(Document).where(_stale())) or 0


def purge_inactive_vectors(session: Session, store: VectorStore) -> int:
    """Remove the vectors of documents that were replaced or whose file is gone.

    Their text stays in the database (it is history, or waiting for `prune`); only the search
    index forgets them. Returns the number of documents purged.
    """
    ids = session.scalars(select(Document.id).where(_stale()).order_by(Document.id)).all()
    for document_id in ids:
        store.delete_document(document_id)
    if ids:
        session.execute(update(Document).where(Document.id.in_(ids)).values(indexed_model=None))
        session.execute(update(Chunk).where(Chunk.document_id.in_(ids)).values(indexed_model=None))
        session.commit()
    return len(ids)


def count_pending_chunks(session: Session, model_name: str) -> int:
    """Chunks of current documents that still need a vector, for showing progress."""
    query = (
        select(func.count())
        .select_from(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            _pending(model_name),
            or_(Chunk.indexed_model.is_(None), Chunk.indexed_model != model_name),
        )
    )
    return session.scalar(query) or 0


def count_chunks_to_embed(session: Session, model_name: str, document_ids: Sequence[int]) -> int:
    """Chunks of these documents that have no vector from `model_name` yet."""
    total = 0
    for start in range(0, len(document_ids), 500):  # stay below SQLite's limit on parameters
        total += (
            session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(
                    Chunk.document_id.in_(list(document_ids[start : start + 500])),
                    or_(Chunk.indexed_model.is_(None), Chunk.indexed_model != model_name),
                )
            )
            or 0
        )
    return total


def index_pending(
    session: Session,
    embedder: EmbeddingProvider,
    store: VectorStore,
    *,
    only: Sequence[int] | None = None,
    limit: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> IndexSummary:
    """Embed pending documents (all of them, or at most `limit`, or just those in `only`).

    Progress is saved after every batch of chunks, so an interrupted run (a crash, a closed
    window, an embedder error) continues at the first chunk without a vector instead of starting
    the document again. A vector is stored before its chunk is marked, and a vector's id depends
    only on its chunk, so repeating a batch never duplicates anything. A document counts as
    indexed only after all its chunks are.
    """
    summary = IndexSummary()
    summary.documents_purged = purge_inactive_vectors(session, store)
    model = embedder.model_name
    query = select(Document.id).where(_pending(model)).order_by(Document.id)
    if only is not None:
        query = query.where(Document.id.in_(list(only)))
    if limit is not None:
        query = query.limit(limit)
    pending_ids = session.scalars(query).all()

    started = time.monotonic()
    chunks_total = count_chunks_to_embed(session, model, pending_ids)

    def report() -> None:
        if on_progress is not None:
            on_progress(
                IndexProgress(
                    documents_done=summary.documents_indexed,
                    documents_total=len(pending_ids),
                    chunks_done=summary.chunks_embedded,
                    chunks_total=chunks_total,
                    seconds=time.monotonic() - started,
                )
            )

    for document_id in pending_ids:
        document = session.get_one(Document, document_id)
        chunks = session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        ).all()
        todo = [chunk for chunk in chunks if chunk.indexed_model != model]
        file_type = Path(document.original_filename).suffix.lower()

        if len(todo) == len(chunks):
            store.delete_document(document_id)  # nothing of it is indexed: clear any leftovers
        for start in range(0, len(todo), EMBED_BATCH_SIZE):
            batch = todo[start : start + EMBED_BATCH_SIZE]
            vectors = embedder.embed_documents([embedding_input(chunk) for chunk in batch])
            store.upsert(
                [
                    VectorPoint(
                        document_id=document_id,
                        chunk_index=chunk.chunk_index,
                        vector=vector,
                        text_hash=text_hash(chunk.text),
                        file_type=file_type,
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
            )
            for chunk in batch:
                chunk.indexed_model = model
            session.commit()
            summary.chunks_embedded += len(batch)
            report()

        document.indexed_model = model
        session.commit()
        summary.documents_indexed += 1
        report()

    logger.info(
        "index finished documents=%d chunks=%d purged=%d",
        summary.documents_indexed,
        summary.chunks_embedded,
        summary.documents_purged,
    )
    return summary


def reset_index(session: Session, store: VectorStore) -> None:
    """Drop all vectors, so the next index_pending re-embeds every document."""
    store.reset()
    session.execute(update(Document).values(indexed_model=None))
    session.execute(update(Chunk).values(indexed_model=None))
    session.commit()
