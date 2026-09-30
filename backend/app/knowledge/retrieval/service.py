"""Find the stored chunks most relevant to a query."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.core.config import MAX_TOP_K
from app.knowledge.indexing.service import text_hash
from app.storage.models import Chunk, Document
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result. Text and provenance come from the database, never from a model."""

    citation_id: str
    document_id: int
    chunk_index: int
    score: float  # cosine similarity, higher is more relevant
    source: str  # original file name, for display
    heading_path: str
    start_line: int
    end_line: int
    text: str


def retrieve(
    session: Session,
    embedder: EmbeddingProvider,
    store: VectorStore,
    query: str,
    *,
    top_k: int,
    document_ids: Sequence[int] | None = None,
    file_types: Sequence[str] | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Return up to `top_k` chunks most similar to `query`, best first.

    Vectors whose chunk no longer exists, or whose text changed since it was embedded, are
    skipped rather than returned with the wrong text.
    """
    query = query.strip()
    if not query:
        raise ValueError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

    hits = store.search(
        embedder.embed_query(query),
        top_k=top_k,
        document_ids=document_ids,
        file_types=file_types,
    )
    if min_score is not None:
        hits = [hit for hit in hits if hit.score >= min_score]
    if not hits:
        logger.info("search finished results=0")
        return []

    rows = session.execute(
        select(Chunk, Document.original_filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            or_(
                *(
                    and_(Chunk.document_id == hit.document_id, Chunk.chunk_index == hit.chunk_index)
                    for hit in hits
                )
            )
        )
    ).all()
    stored = {(chunk.document_id, chunk.chunk_index): (chunk, name) for chunk, name in rows}

    results: list[RetrievedChunk] = []
    stale = 0
    for hit in hits:
        found = stored.get((hit.document_id, hit.chunk_index))
        if found is None or text_hash(found[0].text) != hit.text_hash:
            stale += 1
            continue
        chunk, source = found
        results.append(
            RetrievedChunk(
                citation_id=chunk.citation_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                score=hit.score,
                source=source,
                heading_path=chunk.heading_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
            )
        )

    if stale:
        logger.warning("search skipped stale vectors count=%d (run `python -m app index`)", stale)
    logger.info("search finished results=%d", len(results))  # never log the query itself
    return results
