import json
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import SessionDep
from app.assistant.feedback import (
    MAX_ANSWER_CHARS,
    MAX_FEEDBACK,
    MAX_NOTE_CHARS,
    MAX_QUESTION_CHARS,
    FeedbackFullError,
    add_feedback,
    change_kind,
    count_by_kind,
    count_feedback,
    delete_all_feedback,
    delete_feedback,
    get_feedback,
    list_feedback,
)
from app.storage.models import Feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])

Kind = Literal["helpful", "not_helpful", "wrong_source", "missing_info"]


class FeedbackIn(BaseModel):
    kind: Kind
    mode: Literal["notes", "general"] = Field(description="Which kind of answer is marked.")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    details: dict[str, Any] | None = Field(
        default=None, description="What the interface knew: the sources, why it was refused."
    )


class FeedbackChange(BaseModel):
    kind: Kind
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class FeedbackOut(BaseModel):
    id: int
    kind: Kind
    mode: Literal["notes", "general"]
    question: str
    answer: str
    note: str | None
    details: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class FeedbackList(BaseModel):
    marks: list[FeedbackOut]
    total: int = Field(description="All marks that match, before paging.")
    counts: dict[str, int] = Field(description="Marks of each kind, over everything kept.")
    limit: int = Field(description="The most marks that can be kept.")


class ClearedFeedback(BaseModel):
    deleted: int


def _out(mark: Feedback) -> FeedbackOut:
    details: dict[str, Any] | None = None
    if mark.details:
        try:
            decoded = json.loads(mark.details)
        except ValueError:
            decoded = None
        details = decoded if isinstance(decoded, dict) else None
    return FeedbackOut(
        id=mark.id,
        kind=mark.kind,  # type: ignore[arg-type]  # only the service writes it, from the same list
        mode="notes" if mark.mode == "notes" else "general",
        question=mark.question,
        answer=mark.answer,
        note=mark.note,
        details=details,
        created_at=mark.created_at,
        updated_at=mark.updated_at,
    )


def _existing(session: Session, feedback_id: int) -> Feedback:
    mark = get_feedback(session, feedback_id)
    if mark is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That mark does not exist.")
    return mark


@router.get("")
def get_marks(
    session: SessionDep,
    kinds: Annotated[list[Kind] | None, Query(description="Only marks of these kinds.")] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> FeedbackList:
    """The owner's marks on answers, newest first."""
    counts = count_by_kind(session)
    total = sum(counts[k] for k in kinds) if kinds else count_feedback(session)
    return FeedbackList(
        marks=[_out(m) for m in list_feedback(session, kinds=kinds, limit=limit, offset=offset)],
        total=total,
        counts=counts,
        limit=MAX_FEEDBACK,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def post_mark(body: FeedbackIn, session: SessionDep) -> FeedbackOut:
    """Mark an answer. A thumbs-down, a wrong source or missing information can later become an
    evaluation question (`python -m app feedback export`)."""
    try:
        mark = add_feedback(
            session,
            kind=body.kind,
            mode=body.mode,
            question=body.question,
            answer=body.answer,
            note=body.note,
            details=body.details,
        )
    except FeedbackFullError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _out(mark)


@router.put("/{feedback_id}")
def put_mark(feedback_id: int, body: FeedbackChange, session: SessionDep) -> FeedbackOut:
    """Change what a mark says, for example from not helpful to wrong source."""
    mark = _existing(session, feedback_id)
    try:
        change_kind(session, mark, body.kind, body.note)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _out(mark)


@router.delete("/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_mark(feedback_id: int, session: SessionDep) -> None:
    """Take a mark back, for good."""
    if not delete_feedback(session, feedback_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That mark does not exist.")


@router.delete("")
def remove_all_marks(session: SessionDep) -> ClearedFeedback:
    """Delete every mark, for good."""
    return ClearedFeedback(deleted=delete_all_feedback(session))
