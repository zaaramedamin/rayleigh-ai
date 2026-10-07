"""Marked failures become evaluation questions that wait for review."""

import json
import os
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.assistant.feedback import add_feedback, delete_feedback
from app.evaluation import candidates
from app.evaluation.candidates import (
    DEFAULT_FILE,
    CandidatesFileError,
    candidate_from,
    export_candidates,
)
from app.evaluation.dataset import BACKEND_DIR, DatasetError, load_dataset
from app.storage.models import Feedback


def mark(session: Session, kind: str = "not_helpful", **overrides: object) -> Feedback:
    values: dict[str, object] = {
        "kind": kind,
        "mode": "notes",
        "question": "How long do oats simmer?",
        "answer": "Rice needs eighteen minutes [1].",
        "details": {
            "sources": [{"document_id": 4, "source": "rice.txt"}],
            "reason": "answered",
            "searched_for": None,
        },
        **overrides,
    }
    return add_feedback(session, **values)  # type: ignore[arg-type]


def entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["candidates"]


# --- one entry -------------------------------------------------------------------------------


def test_a_wrong_source_becomes_an_entry_that_says_what_to_fill_in(session: Session) -> None:
    created = mark(session, "wrong_source", note="it used the rice note")

    entry = candidate_from(created)

    assert entry["id"] == f"feedback-{created.id}"
    assert (entry["type"], entry["group"]) == ("answerable", "feedback")
    assert entry["question"] == "How long do oats simmer?"
    assert (
        entry["expected_sources"] == [] and entry["answer_contains"] == []
    )  # only the owner knows
    review = entry["review"]
    assert review["marked"] == "wrong_source" and review["cited"] == ["rice.txt"]
    assert review["answer_given"] == "Rice needs eighteen minutes [1]."
    assert (
        review["owner_note"] == "it used the rice note" and review["answer_was_a_refusal"] is False
    )
    assert any("rice.txt" in item and "wrong" in item for item in review["todo"])
    assert "WRONG SOURCE" in entry["note"]


def test_an_answer_that_refused_comes_with_the_hint_that_the_notes_may_lack_it(
    session: Session,
) -> None:
    created = mark(
        session,
        "missing_info",
        answer="I don't have enough information in your notes to answer that.",
        details={"sources": [], "reason": "no_relevant_notes"},
    )

    review = candidate_from(created)["review"]

    assert review["answer_was_a_refusal"] is True and review["cited"] == []
    assert any("unanswerable" in item for item in review["todo"])


def test_an_entry_still_builds_when_the_details_are_missing_or_odd(session: Session) -> None:
    plain = mark(session, details=None)
    odd = mark(session, details={"sources": "not a list", "reason": 5})

    for created in (plain, odd):
        review = candidate_from(created)["review"]
        assert review["cited"] == [] and review["answer_was_a_refusal"] is False


def test_an_unreviewed_entry_cannot_be_loaded_as_part_of_an_evaluation_set(
    session: Session, tmp_path: Path
) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "oats.md").write_text("Simmer five minutes.", encoding="utf-8")
    entry = candidate_from(mark(session, "wrong_source"))

    def write(question: dict) -> None:
        (tmp_path / "questions.json").write_text(
            json.dumps({"questions": [question]}), encoding="utf-8"
        )

    write(entry)
    with pytest.raises(DatasetError, match="at least one expected source"):
        load_dataset(tmp_path)

    # After the owner names the note and what a good answer says, the same entry is valid.
    entry["expected_sources"] = [{"file": "oats.md"}]
    entry["answer_contains"] = [["five minutes"]]
    write(entry)
    assert load_dataset(tmp_path).questions[0].question == "How long do oats simmer?"


# --- the file --------------------------------------------------------------------------------


def test_failures_on_notes_answers_are_written_oldest_first_and_helpful_ones_are_not(
    session: Session, tmp_path: Path
) -> None:
    mark(session, "helpful", question="a thumbs-up")
    first = mark(session, "not_helpful", question="first failure")
    second = mark(session, "missing_info", question="second failure")
    target = tmp_path / "nested" / "private" / "candidates.json"

    result = export_candidates(session, target)

    assert (result.added, result.kept, result.skipped_general) == (2, 0, 0)
    assert [e["id"] for e in entries(target)] == [f"feedback-{first.id}", f"feedback-{second.id}"]
    assert "waiting for review" in json.loads(target.read_text(encoding="utf-8"))["description"]


def test_a_failure_on_general_chat_is_counted_but_not_written(
    session: Session, tmp_path: Path
) -> None:
    mark(session, "not_helpful", mode="general", question="chat question")
    mark(session, "not_helpful", question="notes question")
    target = tmp_path / "candidates.json"

    result = export_candidates(session, target)

    assert (result.added, result.skipped_general) == (1, 1)
    assert [e["question"] for e in entries(target)] == ["notes question"]


def test_running_it_again_adds_only_new_marks_and_keeps_what_the_owner_edited(
    session: Session, tmp_path: Path
) -> None:
    mark(session, "not_helpful", question="first")
    target = tmp_path / "candidates.json"
    export_candidates(session, target)
    reviewed = json.loads(target.read_text(encoding="utf-8"))
    reviewed["candidates"][0]["expected_sources"] = [{"file": "oats.md"}]  # the owner's edit
    target.write_text(json.dumps(reviewed), encoding="utf-8")
    mark(session, "wrong_source", question="second")

    again = export_candidates(session, target)

    assert (again.added, again.kept) == (1, 1)
    assert [e["question"] for e in entries(target)] == ["first", "second"]
    assert entries(target)[0]["expected_sources"] == [{"file": "oats.md"}]  # not overwritten
    assert export_candidates(session, target).added == 0  # nothing new, nothing written


def test_an_entry_stays_when_its_mark_is_deleted_because_the_owner_may_have_edited_it(
    session: Session, tmp_path: Path
) -> None:
    created = mark(session, question="kept in the file")
    target = tmp_path / "candidates.json"
    export_candidates(session, target)

    delete_feedback(session, created.id)
    result = export_candidates(session, target)

    assert (result.added, result.kept) == (0, 1)
    assert entries(target)[0]["question"] == "kept in the file"


def test_with_nothing_to_export_no_file_is_made(session: Session, tmp_path: Path) -> None:
    mark(session, "helpful")
    target = tmp_path / "private" / "candidates.json"

    result = export_candidates(session, target)

    assert result.added == 0 and not target.exists() and not target.parent.exists()


@pytest.mark.parametrize(
    "content", ["{ nope", "[1, 2]", '{"candidates": "x"}', '{"candidates": [1]}']
)
def test_a_file_that_is_not_ours_is_refused_and_left_alone(
    session: Session, tmp_path: Path, content: str
) -> None:
    mark(session)
    target = tmp_path / "candidates.json"
    target.write_text(content, encoding="utf-8")

    with pytest.raises(CandidatesFileError):
        export_candidates(session, target)

    assert target.read_text(encoding="utf-8") == content


def test_a_write_that_fails_leaves_the_old_file_and_no_leftovers(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark(session, question="first")
    target = tmp_path / "out" / "candidates.json"
    export_candidates(session, target)
    before = target.read_text(encoding="utf-8")
    mark(session, question="second")

    def broken(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(candidates.os, "replace", broken)

    with pytest.raises(OSError, match="disk full"):
        export_candidates(session, target)

    assert target.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in target.parent.iterdir()) == ["candidates.json"]  # no temp file


def test_the_default_file_is_in_the_private_folder_that_git_ignores() -> None:
    assert DEFAULT_FILE.parent.name == "eval-private" and DEFAULT_FILE.parent.parent == BACKEND_DIR
    ignore = (BACKEND_DIR.parent / ".gitignore").read_text(encoding="utf-8")
    assert "eval-private/" in ignore.splitlines()
    assert os.path.basename(DEFAULT_FILE) == "feedback-candidates.json"
