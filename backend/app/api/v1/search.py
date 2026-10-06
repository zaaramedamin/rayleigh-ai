from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import EmbedderDep, SessionDep, SettingsDep, VectorStoreDep
from app.core.config import MAX_TOP_K
from app.knowledge.retrieval.keyword import KeywordSearchUnavailable
from app.knowledge.retrieval.service import MAX_QUERY_CHARS, retrieve

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS, description="What to look for, in plain words."
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="Number of results (default: RETRIEVAL_TOP_K).",
    )
    document_ids: list[int] | None = Field(
        default=None, max_length=100, description="Only search these documents."
    )
    file_types: list[Annotated[str, Field(max_length=16)]] | None = Field(
        default=None, max_length=20, description='Only search these file types, e.g. [".md"].'
    )
    mode: Literal["vector", "keyword", "hybrid"] | None = Field(
        default=None,
        description="By meaning, by the words in the query, or both (default: SEARCH_MODE).",
    )


class SearchResult(BaseModel):
    citation_id: str
    document_id: int
    chunk_index: int
    score: float
    source: str
    heading_path: str
    start_line: int
    end_line: int
    text: str
    start_page: int | None = Field(default=None, description="First page, for formats with pages.")
    end_page: int | None = None
    keyword_score: float | None = Field(
        default=None, description="How well the words match, when found by keyword (BM25)."
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]


@router.post("/search")
def search(
    request: SearchRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: VectorStoreDep,
    settings: SettingsDep,
) -> SearchResponse:
    """Semantic search over the indexed notes: the most similar chunks, with their sources."""
    try:
        results = retrieve(
            session,
            embedder,
            store,
            request.query,
            top_k=settings.retrieval_top_k if request.top_k is None else request.top_k,
            document_ids=request.document_ids,
            file_types=request.file_types,
            mode=request.mode or settings.search_mode,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except KeywordSearchUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return SearchResponse(results=[SearchResult(**asdict(result)) for result in results])
