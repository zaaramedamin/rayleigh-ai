"""Find the stored chunks most relevant to a query.

Three ways to search, chosen with `mode`:

- ``vector``   by meaning: the chunks whose embedding is closest to the question's;
- ``keyword``  by words: the chunks that contain the question's words (exact codes, names, numbers);
- ``hybrid``   both, merged by reciprocal rank fusion.

Whatever the mode, every result carries its **cosine similarity** to the question as `score`, and
that is the number the relevance gate uses (`ANSWER_MIN_SCORE`). Fused and keyword scores are not
similarities, so they only decide the order, never whether a note is relevant enough to answer from.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.core.config import MAX_TOP_K
from app.knowledge.indexing.service import text_hash
from app.knowledge.retrieval.keyword import KeywordHit, KeywordSearchUnavailable, keyword_search
from app.storage.models import DOC_ACTIVE, Chunk, Document
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000

SearchMode = Literal["vector", "keyword", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("vector", "keyword", "hybrid")

# Reciprocal rank fusion: each ranking gives a chunk 1 / (RRF_K + rank). 60 is the usual value; it
# keeps one first place from drowning out agreement between the two rankings.
RRF_K = 60
# A keyword match counts in proportion to the share of the question's words the chunk holds, and
# one that holds fewer than this share is ignored: matching a single common word of a longer
# question says almost nothing, while matching its exact code says a lot.
MIN_KEYWORD_COVERAGE = 0.5
# Merging needs more candidates than the final list: a chunk ranked 8th by both rankings can beat
# one ranked 1st by only one.
CANDIDATE_FACTOR = 4
MIN_CANDIDATES = 20
MAX_CANDIDATES = 200

Key = tuple[int, int]  # (document_id, chunk_index)


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result. Text and provenance come from the database, never from a model."""

    citation_id: str
    document_id: int
    chunk_index: int
    score: float  # cosine similarity to the question, higher is more relevant; the gate uses it
    source: str  # original file name, for display
    heading_path: str
    start_line: int
    end_line: int
    text: str
    # For formats with pages (PDF): the pages the chunk comes from, 1-based.
    start_page: int | None = None
    end_page: int | None = None
    # How well the words of the question match (BM25), when the chunk was found by keyword.
    keyword_score: float | None = None


def describe_location(
    start_line: int, end_line: int, start_page: int | None, end_page: int | None
) -> str:
    """Where in the file a chunk comes from: its pages if it has them, else its lines."""
    if start_page is not None:
        if end_page is None or end_page == start_page:
            return f"page {start_page}"
        return f"pages {start_page}-{end_page}"
    return f"lines {start_line}-{end_line}"


def weighted_rank_fusion(rankings: Sequence[Sequence[tuple[Key, float]]]) -> list[Key]:
    """Merge rankings (best first) into one. Each ranking lists (chunk, weight): a chunk gets
    weight / (RRF_K + rank) from every ranking that has it. A chunk near the top of several
    beats one that is first in only one. Ties keep the order of the first ranking that
    mentions the chunk."""
    scores: dict[Key, float] = defaultdict(float)
    first_seen: dict[Key, int] = {}
    for ranking in rankings:
        for rank, (key, weight) in enumerate(ranking, start=1):
            scores[key] += weight / (RRF_K + rank)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores, key=lambda key: (-scores[key], first_seen[key]))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Key]]) -> list[Key]:
    """Merge plain rankings (best first) with equal weight."""
    return weighted_rank_fusion([[(key, 1.0) for key in ranking] for ranking in rankings])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


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
    mode: SearchMode = "vector",
) -> list[RetrievedChunk]:
    """Return up to `top_k` chunks most relevant to `query`, best first.

    Vectors whose chunk no longer exists, or whose text changed since it was embedded, are
    skipped rather than returned with the wrong text. In `hybrid` mode a SQLite without FTS5 falls
    back to meaning search; in `keyword` mode it raises KeywordSearchUnavailable.
    """
    query = query.strip()
    if not query:
        raise ValueError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}")

    candidates = (
        top_k
        if mode == "vector"
        else min(MAX_CANDIDATES, max(top_k * CANDIDATE_FACTOR, MIN_CANDIDATES))
    )
    query_vector = embedder.embed_query(query)

    vector_hits = []
    if mode != "keyword":
        # Replaced versions and files that are gone are never searched. Their vectors are told to
        # the store as excluded, so they cannot crowd real matches out of the top results.
        inactive = session.scalars(select(Document.id).where(Document.status != DOC_ACTIVE)).all()
        vector_hits = store.search(
            query_vector,
            top_k=candidates,
            document_ids=document_ids,
            file_types=file_types,
            exclude_document_ids=inactive,
        )
    keyword_hits: list[KeywordHit] = []
    if mode != "vector":
        try:
            keyword_hits = keyword_search(
                session, query, top_k=candidates, document_ids=document_ids, file_types=file_types
            )
        except KeywordSearchUnavailable:
            if mode == "keyword":
                raise
            logger.warning("keyword search is not available here; using meaning search only")

    vector_order = [(hit.document_id, hit.chunk_index) for hit in vector_hits]
    keyword_order = [(hit.document_id, hit.chunk_index) for hit in keyword_hits]
    if mode == "vector":
        order = vector_order
    elif mode == "keyword":
        order = keyword_order
    else:
        strong = [h for h in keyword_hits if h.coverage >= MIN_KEYWORD_COVERAGE]
        order = weighted_rank_fusion(
            [
                [(key, 1.0) for key in vector_order],
                [((h.document_id, h.chunk_index), h.coverage) for h in strong],
            ]
        )
    order = order[:top_k]
    if not order:
        logger.info("search finished results=0")
        return []

    by_vector = {(hit.document_id, hit.chunk_index): hit for hit in vector_hits}
    by_keyword = {(hit.document_id, hit.chunk_index): hit.score for hit in keyword_hits}
    # A chunk found only by its words has no similarity yet: ask the store for it.
    missing = [key for key in order if key not in by_vector]
    similarity = store.similarities(query_vector, missing) if missing else {}

    rows = session.execute(
        select(Chunk, Document.original_filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Document.status == DOC_ACTIVE,
            Chunk.indexed_model == embedder.model_name,  # only what is searchable
            or_(*(and_(Chunk.document_id == d, Chunk.chunk_index == c) for d, c in order)),
        )
    ).all()
    stored = {(chunk.document_id, chunk.chunk_index): (chunk, name) for chunk, name in rows}

    results: list[RetrievedChunk] = []
    stale = 0
    for key in order:
        found = stored.get(key)
        hit = by_vector.get(key)
        if found is None or (hit is not None and text_hash(found[0].text) != hit.text_hash):
            stale += 1
            continue
        score = hit.score if hit is not None else similarity.get(key, 0.0)
        if min_score is not None and score < min_score:
            continue
        chunk, source = found
        results.append(
            RetrievedChunk(
                citation_id=chunk.citation_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                score=score,
                source=source,
                heading_path=chunk.heading_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                keyword_score=by_keyword.get(key),
            )
        )

    if stale:
        logger.warning("search skipped stale vectors count=%d (run `python -m app index`)", stale)
    logger.info("search finished results=%d mode=%s", len(results), mode)  # never the query itself
    return results
