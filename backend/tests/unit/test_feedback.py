"""Marks on answers: the service behind /api/v1/feedback."""

import sqlite3
import time
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.assistant import feedback as service
from app.assistant.conversations import REMOVED_NOTICE, forget_document_passages
from app.assistant.feedback import (
    FAILURES,
    KINDS,
    FeedbackFullError,
    add_feedback,
    change_kind,
    count_by_kind,
    delete_all_feedback,
    delete_feedback,
    list_feedback,
)
from app.knowledge.library.documents import delete_document
from app.security.migrate import encrypt_library
from app.security.vault import TEXT_PREFIX
from app.storage.database import DB_FILENAME, create_db_engine
from app.storage.files import save_file
from app.storage.models import Feedback

SOURCES = {
    "sources": [{"document_id": 4, "source": "oats.md"}, {"document_id": 9}],
    "reason": "answered",
}


def mark(session: Session, kind: str = "not_helpful", **overrides: object) -> Feedback:
    values: dict[str, object] = {
        "kind": kind,
        "mode": "notes",
        "question": "How long do oats simmer?",
        "answer": "Five minutes [1].",
        **overrides,
    }
    return add_feedback(session, **values)  # type: ignore[arg-type]


def stored(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Feedback)) or 0


# --- adding ----------------------------------------------------------------------------------


def test_a_mark_keeps_what_is_needed_to_turn_a_failure_into_a_question(session: Session) -> None:
    created = mark(
        session, "wrong_source", question="  Which oats?  ", answer=" Rolled [1]. ", details=SOURCES
    )

    again = session.get(Feedback, created.id)
    assert again is not None
    assert (again.kind, again.mode) == ("wrong_source", "notes")
    assert (again.question, again.answer) == ("Which oats?", "Rolled [1].")
    assert '"oats.md"' in (again.details or "")
    assert again.cited_documents == ",4,9,"  # numbers only: which documents, never what they said
    assert again.created_at is not None and again.updated_at is not None


def test_every_kind_of_mark_is_accepted_and_failures_are_named(session: Session) -> None:
    for kind in KINDS:
        mark(session, kind)

    assert stored(session) == 4
    assert set(FAILURES) == set(KINDS) - {"helpful"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "great"},
        {"mode": "secret"},
        {"question": "  "},
        {"answer": ""},
        {"question": "q" * (service.MAX_QUESTION_CHARS + 1)},
        {"answer": "a" * (service.MAX_ANSWER_CHARS + 1)},
        {"details": {"x": object()}},
        {"details": {"big": "y" * (service.MAX_DETAILS_CHARS + 1)}},
    ],
)
def test_a_mark_that_is_not_valid_is_refused_and_nothing_is_saved(
    session: Session, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        mark(session, **overrides)

    assert stored(session) == 0


def test_the_note_is_one_short_clean_line_or_nothing(session: Session) -> None:
    assert mark(session, note="  it\x00 missed\nthe   garage ").note == "it missed the garage"
    assert mark(session, note="n" * 900).note == "n" * service.MAX_NOTE_CHARS
    assert mark(session, note="   \x00 ").note is None
    assert mark(session).note is None


def test_only_so_many_marks_are_kept(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "MAX_FEEDBACK", 2)
    mark(session)
    mark(session)

    with pytest.raises(FeedbackFullError, match="delete some"):
        mark(session)


# --- changing and reading --------------------------------------------------------------------


def test_a_mark_can_be_changed_and_its_time_moves(session: Session) -> None:
    created = mark(session, "not_helpful")
    first = created.updated_at
    time.sleep(0.01)

    change_kind(session, created, "wrong_source", note="it cited the rice note")

    assert (created.kind, created.note) == ("wrong_source", "it cited the rice note")
    assert created.updated_at > first
    change_kind(session, created, "missing_info")  # a note left out is kept as it was
    assert created.note == "it cited the rice note"


def test_changing_to_an_unknown_kind_changes_nothing(session: Session) -> None:
    created = mark(session, "helpful")

    with pytest.raises(ValueError, match="a mark is one of"):
        change_kind(session, created, "great")

    assert created.kind == "helpful"


def test_marks_are_listed_newest_first_and_can_be_filtered_and_paged(session: Session) -> None:
    first = mark(session, "helpful", question="first")
    second = mark(session, "not_helpful", question="second")
    third = mark(session, "wrong_source", question="third")

    assert [m.id for m in list_feedback(session)] == [third.id, second.id, first.id]
    assert [m.id for m in list_feedback(session, kinds=FAILURES)] == [third.id, second.id]
    assert [m.id for m in list_feedback(session, limit=1, offset=1)] == [second.id]


def test_the_counts_show_every_kind_even_when_nobody_used_it(session: Session) -> None:
    mark(session, "not_helpful")
    mark(session, "not_helpful")
    mark(session, "helpful")

    assert count_by_kind(session) == {
        "helpful": 1,
        "not_helpful": 2,
        "wrong_source": 0,
        "missing_info": 0,
    }


# --- deleting --------------------------------------------------------------------------------


def test_marks_can_be_deleted_one_at_a_time_or_all_at_once(session: Session) -> None:
    keep, drop = mark(session), mark(session)

    assert delete_feedback(session, drop.id) is True
    assert delete_feedback(session, drop.id) is False
    assert [m.id for m in list_feedback(session)] == [keep.id]
    mark(session)
    assert delete_all_feedback(session) == 2
    assert stored(session) == 0


def test_a_deleted_mark_leaves_no_trace_in_the_database_file(
    session: Session, data_dir: Path
) -> None:
    secret = "Quillon-Marmalade-4821"
    created = mark(
        session, question=f"What is {secret}?", answer=f"It is {secret} [1].", details=SOURCES
    )
    database_file = data_dir / DB_FILENAME
    assert secret.encode() in database_file.read_bytes(), (
        "the test needs the text to be there first"
    )

    delete_feedback(session, created.id)

    assert secret.encode() not in database_file.read_bytes()


def test_marks_survive_the_database_being_reopened(session: Session, data_dir: Path) -> None:
    created_id = mark(session, "missing_info", note="the notes should say").id
    session.close()

    engine = create_db_engine(data_dir)  # a new start of the program
    with Session(engine) as again:
        (found,) = list_feedback(again)
        assert (found.id, found.kind, found.note) == (
            created_id,
            "missing_info",
            "the notes should say",
        )
    engine.dispose()


# --- an encrypted library --------------------------------------------------------------------


def test_the_words_of_a_mark_are_encrypted_with_the_library_and_still_readable(
    migrated_data_dir: Path,
) -> None:
    secrets = {
        "question": "Walrus-Basement-9150",
        "answer": "Sentinel-Parrot-3318",
        "note": "Pebble-Harbor-7102",
        "details": "Heron-Lantern-3067",
    }
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        add_feedback(
            session,
            kind="wrong_source",
            mode="notes",
            question=secrets["question"],
            answer=secrets["answer"],
            note=secrets["note"],
            details={"sources": [{"document_id": 2, "source": secrets["details"]}]},
        )
    engine.dispose()

    encrypt_library(migrated_data_dir, "correct horse battery staple", backup_to=None)

    raw = b"".join(p.read_bytes() for p in migrated_data_dir.rglob("*") if p.is_file())
    assert [kind for kind, word in secrets.items() if word.encode() in raw] == []
    connection = sqlite3.connect(migrated_data_dir / DB_FILENAME)
    try:
        row = connection.execute(
            "SELECT kind, question, answer, note, details, cited_documents FROM feedback"
        ).fetchone()
    finally:
        connection.close()
    assert row[0] == "wrong_source" and row[5] == ",2,"  # structure stays readable, never words
    assert all(value.startswith(TEXT_PREFIX) for value in row[1:5])
    engine = create_db_engine(migrated_data_dir)  # unlocks through this Windows account
    with Session(engine) as session:
        (found,) = list_feedback(session)
        assert found.question == secrets["question"] and secrets["details"] in (found.details or "")
    engine.dispose()


# --- removing a document from the library ----------------------------------------------------


def cited(document_id: int) -> dict[str, object]:
    return {"sources": [{"document_id": document_id, "source": f"note-{document_id}.md"}]}


def test_marks_that_quoted_a_removed_document_lose_the_answer_but_keep_the_question(
    session: Session,
) -> None:
    gone = mark(session, answer="The garage code is 4821 [1].", note="wrong note", details=cited(5))
    other = mark(session, answer="Rice needs water [1].", details=cited(15))  # 15 is not 5
    untouched = mark(session, answer="Hello.", details=None)

    scrubbed = forget_document_passages(session, [5])
    session.commit()

    assert scrubbed == 1
    session.refresh(gone), session.refresh(other), session.refresh(untouched)
    assert gone.answer == REMOVED_NOTICE and gone.note is None
    assert gone.details == '{"removed": true}' and gone.cited_documents == ""
    assert gone.question == "How long do oats simmer?"  # what the owner asked is theirs
    assert (other.answer, untouched.answer) == ("Rice needs water [1].", "Hello.")


def test_removing_a_document_from_the_library_scrubs_the_marks_that_quoted_it(
    session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, b"The garage code is 4821.", "garage.md")
    created = mark(
        session,
        question="What is the garage code?",
        answer="The garage code is 4821 [1].",
        details=cited(document.id),
    )
    database_file = data_dir / DB_FILENAME
    assert b"4821" in database_file.read_bytes()

    delete_document(session, data_dir, document, exclude=False)

    session.refresh(created)
    assert created.answer == REMOVED_NOTICE
    assert b"4821" not in database_file.read_bytes()


def test_a_library_that_has_no_feedback_table_yet_is_not_a_problem(session: Session) -> None:
    session.execute(text("DROP TABLE feedback"))
    session.commit()

    assert forget_document_passages(session, [1, 2]) == 0
