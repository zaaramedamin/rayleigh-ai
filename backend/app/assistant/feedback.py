"""The owner's marks on answers: helpful, not helpful, wrong source, missing information.

A mark is kept on this computer so that a failure can later become an evaluation question
(app.evaluation.candidates). It keeps the question, the answer that was given and what the answer
cited. Those are the owner's words and text built from their notes, so they are treated like note
text: stored in encrypted columns when the library is encrypted, never logged, and replaced by a
notice when a document they quoted is removed from the library.
"""

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.assistant.conversations import cited_documents
from app.storage.models import Feedback

KINDS = ("helpful", "not_helpful", "wrong_source", "missing_info")
# The marks that say something went wrong, and so can become evaluation questions.
FAILURES = ("not_helpful", "wrong_source", "missing_info")
MODES = ("notes", "general")

MAX_FEEDBACK = 5000
MAX_QUESTION_CHARS = 8000
MAX_ANSWER_CHARS = 16000
MAX_NOTE_CHARS = 500
MAX_DETAILS_CHARS = 60_000

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class FeedbackFullError(ValueError):
    """As many marks are kept as may be. Delete some first."""


def _details_json(details: dict[str, Any] | None) -> str | None:
    if details is None:
        return None
    try:
        text = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("the details of a mark must be plain JSON data") from exc
    if len(text) > MAX_DETAILS_CHARS:
        raise ValueError("the details of a mark are too large")
    return text


def _clean_note(note: str | None) -> str | None:
    cleaned = " ".join(_CONTROL.sub(" ", note or "").split())[:MAX_NOTE_CHARS].strip()
    return cleaned or None


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"a mark is one of: {', '.join(KINDS)}")


def count_feedback(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Feedback)) or 0


def add_feedback(
    session: Session,
    *,
    kind: str,
    mode: str,
    question: str,
    answer: str,
    note: str | None = None,
    details: dict[str, Any] | None = None,
) -> Feedback:
    """Keep a mark. Raises ValueError if it is not valid, FeedbackFullError at the limit."""
    _check_kind(kind)
    if mode not in MODES:
        raise ValueError(f"the mode of an answer is one of: {', '.join(MODES)}")
    asked, given = question.strip(), answer.strip()
    if not asked or not given:
        raise ValueError("a mark needs the question and the answer it is about")
    if len(asked) > MAX_QUESTION_CHARS or len(given) > MAX_ANSWER_CHARS:
        raise ValueError("the question or the answer is too long to keep")
    stored_details = _details_json(details)  # before anything is saved
    if count_feedback(session) >= MAX_FEEDBACK:
        raise FeedbackFullError(f"{MAX_FEEDBACK} marks are kept; delete some to add more")

    mark = Feedback(
        kind=kind,
        mode=mode,
        question=asked,
        answer=given,
        note=_clean_note(note),
        details=stored_details,
        cited_documents=cited_documents(details),
    )
    session.add(mark)
    session.commit()
    return mark


def get_feedback(session: Session, feedback_id: int) -> Feedback | None:
    return session.get(Feedback, feedback_id)


def change_kind(session: Session, mark: Feedback, kind: str, note: str | None = None) -> None:
    """Change what a mark says, for example from not helpful to wrong source."""
    _check_kind(kind)
    mark.kind = kind
    if note is not None:
        mark.note = _clean_note(note)
    mark.updated_at = datetime.now(UTC)
    session.commit()


def list_feedback(
    session: Session, *, kinds: Iterable[str] | None = None, limit: int = 100, offset: int = 0
) -> list[Feedback]:
    """Marks, the newest first, optionally only of some kinds."""
    query = select(Feedback)
    if kinds is not None:
        query = query.where(Feedback.kind.in_(list(kinds)))
    query = (
        query.order_by(Feedback.created_at.desc(), Feedback.id.desc()).limit(limit).offset(offset)
    )
    return list(session.scalars(query))


def count_by_kind(session: Session) -> dict[str, int]:
    """How many marks of each kind, with zero for a kind nobody used."""
    rows = session.execute(select(Feedback.kind, func.count()).group_by(Feedback.kind)).all()
    found = {kind: int(count) for kind, count in rows}
    return {kind: found.get(kind, 0) for kind in KINDS}


def delete_feedback(session: Session, feedback_id: int) -> bool:
    mark = session.get(Feedback, feedback_id)
    if mark is None:
        return False
    session.delete(mark)
    session.commit()
    return True


def delete_all_feedback(session: Session) -> int:
    count = count_feedback(session)
    session.execute(delete(Feedback))
    session.commit()
    session.expire_all()
    return count
