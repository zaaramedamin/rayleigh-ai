"""Turn the owner's marked failures into evaluation questions that wait for review.

A thumbs-down, a wrong source or missing information tells us where the assistant fell short. This
module writes each such mark, for an answer from the notes, as an entry shaped like the entries of
an evaluation set (`questions.json`), so that after a quick review it can be copied there. The
entry is deliberately *incomplete*: the evaluation needs to know which note should answer the
question and what a good answer says, and only the owner can say that. Until they fill it in the
entry would not pass the evaluation loader, so a candidate can never end up in a set by accident.

The file holds the owner's questions and answers, so it goes in the private evaluation folder
(`eval-private/`, never committed). Entries already in the file are kept as they are, because the
owner may have edited them; only marks that are not in it yet are added.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.assistant.feedback import FAILURES, list_feedback
from app.evaluation.dataset import BACKEND_DIR
from app.storage.models import Feedback

DEFAULT_FILE = BACKEND_DIR / "eval-private" / "feedback-candidates.json"
DESCRIPTION = (
    "Evaluation questions waiting for review, made from answers you marked. For each one: name the "
    "note that should answer it in expected_sources, say what a good answer contains in "
    "answer_contains, then copy the entry into the questions.json of your private evaluation set. "
    "If the notes really do not hold the answer, change type to unanswerable and delete both "
    "lists. "
    "The 'review' part is for you and is not read by the evaluation."
)
# Reasons an answer was refused: the assistant said it did not know.
_REFUSALS = ("no_relevant_notes", "model_declined", "no_valid_citation")

_HEADLINES = {
    "not_helpful": "marked NOT HELPFUL",
    "wrong_source": "marked WRONG SOURCE",
    "missing_info": "marked MISSING INFORMATION",
}


class CandidatesFileError(ValueError):
    """The candidates file exists but is not one this program wrote."""


@dataclass(frozen=True)
class ExportResult:
    added: int
    kept: int  # entries that were already in the file, left as they were
    skipped_general: int  # failures on general chat: the evaluation checks answers from notes
    path: Path


def _details(mark: Feedback) -> dict[str, Any]:
    try:
        decoded = json.loads(mark.details) if mark.details else {}
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _cited_files(details: dict[str, Any]) -> list[str]:
    sources = details.get("sources")
    if not isinstance(sources, list):
        return []
    names = [
        s["source"] for s in sources if isinstance(s, dict) and isinstance(s.get("source"), str)
    ]
    return list(dict.fromkeys(names))


def candidate_from(mark: Feedback) -> dict[str, Any]:
    """The reviewable evaluation entry for one marked failure of a notes answer."""
    details = _details(mark)
    cited = _cited_files(details)
    reason = details.get("reason") if isinstance(details.get("reason"), str) else None
    refused = reason in _REFUSALS
    todo = ["name the note that should answer it in expected_sources"]
    if mark.kind == "wrong_source":
        todo.append(
            f"the assistant cited {', '.join(cited) or 'a note'}, which you marked as wrong"
        )
    todo.append("say in answer_contains what a good answer must contain")
    if refused or mark.kind == "missing_info":
        todo.append(
            "if the notes really do not hold the answer, change type to unanswerable "
            "and delete expected_sources and answer_contains"
        )
    marked_on = (
        mark.created_at.strftime("%Y-%m-%d") if isinstance(mark.created_at, datetime) else ""
    )
    return {
        "id": f"feedback-{mark.id}",
        "type": "answerable",
        "group": "feedback",
        "question": mark.question,
        "expected_sources": [],
        "answer_contains": [],
        "note": f"{_HEADLINES.get(mark.kind, mark.kind)} on {marked_on}: review before use",
        "review": {
            "feedback_id": mark.id,
            "marked": mark.kind,
            "marked_on": marked_on,
            "answer_given": mark.answer,
            "answer_was_a_refusal": refused,
            "cited": cited,
            "searched_for": details.get("searched_for"),
            "owner_note": mark.note,
            "todo": todo,
        },
    }


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CandidatesFileError(f"{path} could not be read as JSON: {exc}") from exc
    entries = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise CandidatesFileError(f"{path} is not a candidates file (no 'candidates' list)")
    return entries


def _already(entries: list[dict[str, Any]]) -> set[int]:
    found: set[int] = set()
    for entry in entries:
        review = entry.get("review")
        number = review.get("feedback_id") if isinstance(review, dict) else None
        if isinstance(number, int):
            found.add(number)
    return found


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    """Replace the file in one step, so an interruption cannot leave half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {"description": DESCRIPTION, "candidates": entries}, indent=2, ensure_ascii=False
    )
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".candidates-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def export_candidates(session: Session, path: Path = DEFAULT_FILE) -> ExportResult:
    """Add every marked failure that is not in the file yet. Raises CandidatesFileError."""
    entries = _read(path)
    known = _already(entries)
    added = skipped = 0
    # Oldest first, so the file reads in the order things went wrong.
    for mark in reversed(list_feedback(session, kinds=FAILURES, limit=10_000)):
        if mark.id in known:
            continue
        if mark.mode != "notes":
            skipped += 1
            continue
        entries.append(candidate_from(mark))
        added += 1
    if added:
        _write(path, entries)
    return ExportResult(added=added, kept=len(entries) - added, skipped_general=skipped, path=path)
