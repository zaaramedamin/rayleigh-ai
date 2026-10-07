from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.ai.llm.base import LLMError
from app.api.deps import EmbedderDep, LLMDep, SessionDep, SettingsDep, locked_vector_store
from app.api.v1.ask import AskSource, llm_status
from app.core.config import MAX_TOP_K
from app.knowledge.answering.tasks import (
    MAX_COMPARED,
    ComparedDocument,
    compare,
    extract,
    load_document_text,
    summarize,
)
from app.knowledge.retrieval.keyword import KeywordSearchUnavailable
from app.knowledge.retrieval.service import MAX_QUERY_CHARS, retrieve

router = APIRouter(prefix="/tasks", tags=["tasks"])

# An extraction reads more notes than an answer does, because a table has many rows.
DEFAULT_EXTRACT_NOTES = 8


class SummarizeRequest(BaseModel):
    document_id: int


class SummaryOut(BaseModel):
    text: str
    document_id: int
    name: str
    parts: int = Field(description="How many parts the document was cut into.")
    covered_parts: int = Field(description="How many of them were summarized.")
    truncated: bool = Field(description="True when the document was longer than one task reads.")


class CompareRequest(BaseModel):
    document_ids: list[int] = Field(min_length=2, max_length=MAX_COMPARED)


class ComparedDocumentOut(BaseModel):
    marker: int = Field(description='The number the text uses for it, e.g. 1 for "[1]".')
    document_id: int
    name: str


class ComparisonOut(BaseModel):
    text: str
    grounded: bool = Field(description="True only when the text cites at least one document.")
    reason: Literal["compared", "nothing_to_compare", "no_valid_citation"]
    documents: list[ComparedDocumentOut] = Field(description="Every document compared.")
    sources: list[ComparedDocumentOut] = Field(description="The ones the text cites.")
    truncated_documents: list[int] = Field(description="Documents that were only partly read.")


class ExtractRequest(BaseModel):
    request: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS, description="Which facts are wanted."
    )
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K, description="Notes to search.")
    document_ids: list[int] | None = Field(default=None, max_length=100)
    file_types: list[Annotated[str, Field(max_length=16)]] | None = Field(
        default=None, max_length=20
    )
    mode: Literal["vector", "keyword", "hybrid"] | None = Field(
        default=None, description="How notes are found (default: SEARCH_MODE)."
    )


class ExtractedRowOut(BaseModel):
    item: str
    value: str
    marker: int = Field(description="The number of the source note, as in `sources`.")


class ExtractionOut(BaseModel):
    rows: list[ExtractedRowOut]
    sources: list[AskSource]
    reason: Literal["extracted", "no_relevant_notes", "nothing_found", "unreadable", "no_valid_row"]
    notes_considered: int
    dropped: int = Field(description="Rows the model wrote that were refused.")


def _documents(entries: tuple[ComparedDocument, ...]) -> list[ComparedDocumentOut]:
    return [
        ComparedDocumentOut(marker=e.marker, document_id=e.document_id, name=e.name)
        for e in entries
    ]


@router.post("/summarize")
def summarize_document(body: SummarizeRequest, session: SessionDep, llm: LLMDep) -> SummaryOut:
    """A short summary of one current document, written from its own text.

    A long document is read in parts and the result says when it was cut short. Works whatever
    state the search index is in.
    """
    try:
        summary = summarize(llm, load_document_text(session, body.document_id))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(llm_status(exc), str(exc)) from exc
    return SummaryOut(
        text=summary.text,
        document_id=summary.document_id,
        name=summary.name,
        parts=summary.parts,
        covered_parts=summary.covered_parts,
        truncated=summary.truncated,
    )


@router.post("/compare")
def compare_documents(body: CompareRequest, session: SessionDep, llm: LLMDep) -> ComparisonOut:
    """What two to four current documents have in common and where they differ, with citations.

    As for an answer, the text is only returned if it cites a document by a number that exists.
    """
    try:
        documents = [load_document_text(session, number) for number in body.document_ids]
        comparison = compare(llm, documents)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(llm_status(exc), str(exc)) from exc
    return ComparisonOut(
        text=comparison.text,
        grounded=comparison.grounded,
        reason=comparison.reason,
        documents=_documents(comparison.documents),
        sources=_documents(comparison.sources),
        truncated_documents=list(comparison.truncated),
    )


@router.post("/extract")
def extract_facts(
    body: ExtractRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> ExtractionOut:
    """A table of the facts the notes hold that match the request, each with its source note."""
    try:
        with locked_vector_store(settings, embedder) as store:
            found = retrieve(
                session,
                embedder,
                store,
                body.request,
                top_k=body.top_k or max(settings.retrieval_top_k, DEFAULT_EXTRACT_NOTES),
                document_ids=body.document_ids,
                file_types=body.file_types,
                mode=body.mode or settings.search_mode,
            )
        # The search index is released here; the model can take a while.
        result = extract(llm, body.request, found, settings.answer_min_score)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except KeywordSearchUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(llm_status(exc), str(exc)) from exc
    return ExtractionOut(
        rows=[ExtractedRowOut(item=r.item, value=r.value, marker=r.marker) for r in result.rows],
        sources=[AskSource(**vars(source)) for source in result.sources],
        reason=result.reason,
        notes_considered=result.notes_considered,
        dropped=result.dropped,
    )
