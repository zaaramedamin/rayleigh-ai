import json
import logging
from collections.abc import AsyncIterator, Generator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.llm.base import (
    ChatMessage,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.api.deps import EmbedderDep, LLMDep, SessionDep, SettingsDep, locked_vector_store
from app.api.v1.chat import MAX_HISTORY_TURNS, ChatTurn
from app.core.config import MAX_TOP_K, Settings
from app.knowledge.answering.rewrite import Rewrite, standalone_question
from app.knowledge.answering.service import (
    Answer,
    Finished,
    Token,
    compose_answer,
    stream_answer,
)
from app.knowledge.retrieval.keyword import KeywordSearchUnavailable
from app.knowledge.retrieval.service import MAX_QUERY_CHARS, RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

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
    mode: Literal["vector", "keyword", "hybrid"] | None = Field(
        default=None, description="How notes are found (default: SEARCH_MODE)."
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Earlier turns of this conversation, oldest first. With them, a follow-up "
        'such as "and how much was it?" is first rewritten into a question that stands alone.',
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
    start_page: int | None = Field(default=None, description="First page, for formats with pages.")
    end_page: int | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True only when the answer is backed by cited notes.")
    reason: Literal["answered", "no_relevant_notes", "model_declined", "no_valid_citation"]
    sources: list[AskSource]
    notes_considered: int
    searched_for: str | None = Field(
        default=None,
        description="The standalone question the notes were searched for, when a follow-up "
        "was rewritten. Null when the question was used as typed.",
    )


def llm_status(exc: LLMError) -> int:
    """The HTTP status for a model failure: slow, not there, or something else."""
    if isinstance(exc, LLMTimeoutError):
        return status.HTTP_504_GATEWAY_TIMEOUT
    if isinstance(exc, LLMUnavailableError | LLMModelNotFoundError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_502_BAD_GATEWAY


def _search(
    request: AskRequest,
    session: Session,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    settings: Settings,
) -> tuple[Rewrite, list[RetrievedChunk]]:
    """Rewrite a follow-up, then find the notes for it."""
    # A follow-up is rewritten first, before the search index is opened: the model can be slow.
    rewrite = standalone_question(
        llm, request.question, [ChatMessage(t.role, t.content) for t in request.history]
    )
    with locked_vector_store(settings, embedder) as store:
        retrieved = retrieve(
            session,
            embedder,
            store,
            rewrite.text,
            top_k=settings.retrieval_top_k if request.top_k is None else request.top_k,
            document_ids=request.document_ids,
            file_types=request.file_types,
            mode=request.mode or settings.search_mode,
        )
    # The search index is released here; the model can take a while.
    return rewrite, retrieved


def _response(answer: Answer, rewrite: Rewrite) -> AskResponse:
    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        reason=answer.reason,
        sources=[AskSource(**vars(source)) for source in answer.sources],
        notes_considered=answer.notes_considered,
        searched_for=rewrite.text if rewrite.changed else None,
    )


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
        rewrite, retrieved = _search(request, session, embedder, llm, settings)
        answer = compose_answer(llm, rewrite.text, retrieved, settings.answer_min_score)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except KeywordSearchUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(llm_status(exc), str(exc)) from exc
    return _response(answer, rewrite)


def _event(name: str, data: dict[str, Any]) -> str:
    """One server-sent event. The data is a single line of JSON."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _events(
    stream: Generator[Token | Finished, None, None], rewrite: Rewrite
) -> AsyncIterator[str]:
    """The answer as events. Closing this (the browser went away) stops the model."""
    try:
        if rewrite.changed:
            yield _event("searching", {"searched_for": rewrite.text})
        # The model is slow and blocking: each piece is waited for off the event loop.
        while (item := await run_in_threadpool(next, stream, None)) is not None:
            if isinstance(item, Token):
                yield _event("token", {"text": item.text})
            else:
                yield _event("done", _response(item.answer, rewrite).model_dump())
    except LLMError as exc:
        yield _event("error", {"status": llm_status(exc), "detail": str(exc)})
    except Exception as exc:
        # What went wrong can hold pieces of a note or a question: only its kind is logged.
        logger.error("streamed answer failed error=%s", type(exc).__name__)
        yield _event(
            "error",
            {"status": 500, "detail": "Something went wrong on this computer. See the server log."},
        )
    finally:
        stream.close()


@router.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """The same answer as `/ask`, sent as server-sent events while the model writes it.

    Events: `searching` (the standalone question, when a follow-up was rewritten), `token` (text
    so far, **unverified**: citations are only checked once the reply is complete), then exactly
    one of `done` (the checked answer, shaped like the `/ask` response, which replaces the streamed
    text) or `error` (`status` and `detail`, as the HTTP error `/ask` would have returned). Closing
    the connection stops the model. Problems found before anything was sent, such as an invalid
    question, are ordinary HTTP errors.
    """
    try:
        rewrite, retrieved = _search(request, session, embedder, llm, settings)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except KeywordSearchUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    stream = stream_answer(llm, rewrite.text, retrieved, settings.answer_min_score)
    return StreamingResponse(
        _events(stream, rewrite),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )
