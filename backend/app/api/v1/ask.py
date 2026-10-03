from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.ai.llm.base import (
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.api.deps import EmbedderDep, LLMDep, SessionDep, SettingsDep, locked_vector_store
from app.core.config import MAX_TOP_K
from app.knowledge.answering.service import compose_answer
from app.knowledge.retrieval.service import MAX_QUERY_CHARS, retrieve

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="Notes to consider (default: RETRIEVAL_TOP_K).",
    )
    document_ids: list[int] | None = Field(default=None, max_length=100)
    file_types: list[Annotated[str, Field(max_length=16)]] | None = Field(
        default=None, max_length=20
    )


class AskSource(BaseModel):
    marker: int = Field(description='The number used in the answer text, e.g. 1 for "[1]".')
    citation_id: str
    document_id: int
    chunk_index: int
    source: str
    heading_path: str
    start_line: int
    end_line: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True only when the answer is backed by cited notes.")
    reason: Literal["answered", "no_relevant_notes", "model_declined", "no_valid_citation"]
    sources: list[AskSource]
    notes_considered: int


@router.post("/ask")
def ask(
    request: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> AskResponse:
    """Answer a question from the notes, with citations, or say there is not enough information."""
    try:
        with locked_vector_store(settings, embedder) as store:
            retrieved = retrieve(
                session,
                embedder,
                store,
                request.question,
                top_k=settings.retrieval_top_k if request.top_k is None else request.top_k,
                document_ids=request.document_ids,
                file_types=request.file_types,
            )
        # The search index is released here; the model can take a while.
        answer = compose_answer(llm, request.question, retrieved, settings.answer_min_score)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(exc)) from exc
    except (LLMUnavailableError, LLMModelNotFoundError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        reason=answer.reason,
        sources=[AskSource(**vars(source)) for source in answer.sources],
        notes_considered=answer.notes_considered,
    )
