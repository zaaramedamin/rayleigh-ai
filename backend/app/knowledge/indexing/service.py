"""Embed chunks and store their vectors, so documents become searchable."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import ColumnElement, func, or_, select, update
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.storage.models import Chunk, Document
from app.storage.vector_store import VectorPoint, VectorStore

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


@dataclass
class IndexSummary:
    documents_indexed: int = 0
    chunks_embedded: int = 0


def text_hash(text: str) -> str:
    """Short fingerprint of a chunk's text, stored next to its vector to detect stale vectors."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def embedding_input(chunk: Chunk) -> str:
    """Text given to the model. The heading path adds context to chunks deep inside a section."""
    return f"{chunk.heading_path}\n\n{chunk.text}" if chunk.heading_path else chunk.text


def _pending(model_name: str) -> ColumnElement[bool]:
    return or_(Document.indexed_model.is_(None), Document.indexed_model != model_name)


def count_pending(session: Session, model_name: str) -> int:
    """Documents whose vectors are missing, stale, or were made by a different model."""
    query = select(func.count()).select_from(Document).where(_pending(model_name))
    return session.scalar(query) or 0


def index_pending(
    session: Session, embedder: EmbeddingProvider, store: VectorStore
) -> IndexSummary:
    """Embed every pending document and replace its vectors in the store.

    A document is marked as indexed only after all its vectors are stored, so an interrupted
    run is simply redone next time.
    """
    summary = IndexSummary()
    pending_ids = session.scalars(
        select(Document.id).where(_pending(embedder.model_name)).order_by(Document.id)
    ).all()

    for document_id in pending_ids:
        document = session.get_one(Document, document_id)
        chunks = session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        ).all()
        file_type = Path(document.original_filename).suffix.lower()

        store.delete_document(document_id)
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
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

        document.indexed_model = embedder.model_name
        session.commit()
        summary.documents_indexed += 1
        summary.chunks_embedded += len(chunks)

    logger.info(
        "index finished documents=%d chunks=%d",
        summary.documents_indexed,
        summary.chunks_embedded,
    )
    return summary


def reset_index(session: Session, store: VectorStore) -> None:
    """Drop all vectors, so the next index_pending re-embeds every document."""
    store.reset()
    session.execute(update(Document).values(indexed_model=None))
    session.commit()
