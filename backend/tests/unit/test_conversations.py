"""Stored conversations: the service behind /api/v1/conversations."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.assistant import conversations as service
from app.assistant.conversations import (
    REMOVED_NOTICE,
    ConversationFullError,
    NewMessage,
    TooManyConversationsError,
    append_messages,
    create_conversation,
    delete_all_conversations,
    delete_conversation,
    forget_document_passages,
    list_conversations,
    messages_of,
    purge_older_than,
    rename_conversation,
    title_from,
)
from app.knowledge.library.documents import delete_document
from app.storage.database import DB_FILENAME, create_db_engine
from app.storage.files import save_file
from app.storage.models import Conversation, Message


def say(
    question: str = "How do I cook oats?", answer: str = "Simmer them [1]."
) -> list[NewMessage]:
    return [NewMessage("user", "notes", question), NewMessage("assistant", "notes", answer)]


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


# --- the basics -------------------------------------------------------------------------------


def test_a_conversation_keeps_its_messages_in_order(session: Session) -> None:
    conversation = create_conversation(session)

    append_messages(session, conversation, say("one", "first"))
    append_messages(session, conversation, say("two", "second"))

    stored = messages_of(session, conversation.id)
    assert [(m.position, m.role, m.content) for m in stored] == [
        (0, "user", "one"),
        (1, "assistant", "first"),
        (2, "user", "two"),
        (3, "assistant", "second"),
    ]


def test_an_empty_title_becomes_the_start_of_the_first_question(session: Session) -> None:
    conversation = create_conversation(session)

    append_messages(session, conversation, say("  How long do I\nsimmer   oats?  "))

    assert conversation.title == "How long do I simmer oats?"


def test_a_long_first_question_gives_a_short_title(session: Session) -> None:
    title = title_from("word " * 100)

    assert len(title) <= service.TITLE_FROM_MESSAGE_CHARS and title.endswith("...")


def test_a_title_given_by_the_owner_is_kept(session: Session) -> None:
    conversation = create_conversation(session, "  Oat\x00 plans  ")

    append_messages(session, conversation, say("something else"))

    assert conversation.title == "Oat plans"


def test_renaming_needs_a_real_title(session: Session) -> None:
    conversation = create_conversation(session, "old")

    rename_conversation(session, conversation, " new  name ")
    assert conversation.title == "new name"
    with pytest.raises(ValueError, match="needs a title"):
        rename_conversation(session, conversation, " \x00 ")
    assert conversation.title == "new name"


def test_the_list_shows_the_most_recently_used_first_with_message_counts(
    session: Session,
) -> None:
    first = create_conversation(session, "first")
    second = create_conversation(session, "second")
    append_messages(session, first, say())  # touched last
    append_messages(session, second, say()[:1])
    append_messages(session, first, say())

    listed = list_conversations(session)

    assert [(c.title, c.message_count) for c in listed] == [("first", 4), ("second", 1)]
    assert [c.title for c in list_conversations(session, limit=1, offset=1)] == ["second"]


def test_what_was_saved_is_still_there_after_the_database_is_reopened(
    session: Session, data_dir: Path
) -> None:
    conversation = create_conversation(session)
    append_messages(
        session,
        conversation,
        [
            NewMessage(
                "assistant", "notes", "Simmer [1].", {"sources": [{"document_id": 3}], "x": "é"}
            )
        ],
    )
    conversation_id = conversation.id
    session.close()

    engine = create_db_engine(data_dir)  # a new start of the program
    with Session(engine) as again:
        (message,) = messages_of(again, conversation_id)
        assert message.content == "Simmer [1]."
        assert '"é"' in (message.payload or "")
    engine.dispose()


# --- limits and refusals ----------------------------------------------------------------------


def test_a_bad_message_in_a_batch_saves_nothing(session: Session) -> None:
    conversation = create_conversation(session)

    with pytest.raises(ValueError, match="empty"):
        append_messages(
            session,
            conversation,
            [NewMessage("user", "general", "fine"), NewMessage("user", "general", "  ")],
        )

    assert messages_of(session, conversation.id) == []
    assert conversation.title == ""


def test_a_message_that_is_too_long_is_refused(session: Session) -> None:
    conversation = create_conversation(session)

    with pytest.raises(ValueError, match="longer than"):
        append_messages(
            session,
            conversation,
            [NewMessage("user", "general", "x" * (service.MAX_CONTENT_CHARS + 1))],
        )


@pytest.mark.parametrize(
    "payload",
    [{"x": object()}, {"x": {1, 2}}, {"big": "y" * (service.MAX_PAYLOAD_CHARS + 1)}],
)
def test_details_that_are_not_plain_or_are_too_large_are_refused(
    session: Session, payload: dict
) -> None:
    conversation = create_conversation(session)

    with pytest.raises(ValueError, match="details"):
        append_messages(session, conversation, [NewMessage("assistant", "notes", "a", payload)])

    assert messages_of(session, conversation.id) == []


def test_a_full_conversation_takes_no_more_and_nothing_of_the_batch(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "MAX_MESSAGES", 3)
    conversation = create_conversation(session)
    append_messages(session, conversation, say())

    with pytest.raises(ConversationFullError, match="start a new one"):
        append_messages(session, conversation, say())

    assert len(messages_of(session, conversation.id)) == 2  # not even the first of the batch


def test_only_so_many_conversations_are_kept(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "MAX_CONVERSATIONS", 2)
    create_conversation(session)
    create_conversation(session)

    with pytest.raises(TooManyConversationsError, match="delete some"):
        create_conversation(session)


# --- deleting ---------------------------------------------------------------------------------


def test_deleting_a_conversation_deletes_its_messages_and_only_its_own(session: Session) -> None:
    keep, drop = create_conversation(session, "keep"), create_conversation(session, "drop")
    append_messages(session, keep, say())
    append_messages(session, drop, say())

    assert delete_conversation(session, drop.id) is True

    assert [c.title for c in list_conversations(session)] == ["keep"]
    assert count(session, Message) == 2
    assert delete_conversation(session, drop.id) is False  # already gone


def test_delete_all_empties_both_tables(session: Session) -> None:
    for _ in range(3):
        append_messages(session, create_conversation(session), say())

    assert delete_all_conversations(session) == 3

    assert (count(session, Conversation), count(session, Message)) == (0, 0)


def test_a_deleted_conversation_leaves_no_trace_in_the_database_file(
    session: Session, data_dir: Path
) -> None:
    secret = "Quillon-Marmalade-4821"
    conversation = create_conversation(session)
    append_messages(
        session,
        conversation,
        [
            NewMessage("user", "general", f"My secret is {secret}"),
            NewMessage(
                "assistant", "notes", "ok", {"sources": [{"document_id": 1, "text": secret}]}
            ),
        ],
    )
    database_file = data_dir / DB_FILENAME
    assert secret.encode() in database_file.read_bytes(), (
        "the test needs the text to be there first"
    )

    delete_conversation(session, conversation.id)

    assert secret.encode() not in database_file.read_bytes()


# --- retention --------------------------------------------------------------------------------


def test_conversations_older_than_the_retention_period_are_deleted(session: Session) -> None:
    old, recent = create_conversation(session, "old"), create_conversation(session, "recent")
    append_messages(session, old, say())
    append_messages(session, recent, say())
    old.updated_at = datetime.now(UTC) - timedelta(days=40)
    session.commit()

    assert purge_older_than(session, 30) == 1

    assert [c.title for c in list_conversations(session)] == ["recent"]
    assert count(session, Message) == 2


def test_no_retention_period_keeps_everything(session: Session) -> None:
    old = create_conversation(session, "old")
    old.updated_at = datetime.now(UTC) - timedelta(days=4000)
    session.commit()

    assert purge_older_than(session, 0) == 0

    assert count(session, Conversation) == 1


# --- removing a document from the library -----------------------------------------------------


def cited(document_id: int) -> NewMessage:
    return NewMessage(
        "assistant",
        "notes",
        f"It says so [1]. (document {document_id})",
        {"grounded": True, "sources": [{"document_id": document_id, "text": "the note"}]},
    )


def test_an_answer_that_quoted_a_removed_document_is_replaced_by_a_notice(
    session: Session,
) -> None:
    conversation = create_conversation(session)
    append_messages(
        session, conversation, [NewMessage("user", "notes", "q"), cited(5), cited(15), cited(6)]
    )

    scrubbed = forget_document_passages(session, [5])
    session.commit()

    texts = {m.content for m in messages_of(session, conversation.id)}
    assert scrubbed == 1
    assert REMOVED_NOTICE in texts
    assert "It says so [1]. (document 15)" in texts  # document 15 is not document 5
    assert "It says so [1]. (document 6)" in texts
    gone = next(m for m in messages_of(session, conversation.id) if m.content == REMOVED_NOTICE)
    assert gone.payload == '{"removed": true}' and gone.cited_documents == ""


def test_removing_a_document_from_the_library_scrubs_the_conversations_that_quoted_it(
    session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, b"The garage code is 4821.", "garage.md")
    conversation = create_conversation(session)
    append_messages(
        session,
        conversation,
        [
            NewMessage("user", "notes", "What is the garage code?"),
            NewMessage(
                "assistant",
                "notes",
                "The garage code is 4821 [1].",
                {"sources": [{"document_id": document.id, "text": "The garage code is 4821."}]},
            ),
        ],
    )
    database_file = data_dir / DB_FILENAME
    assert b"4821" in database_file.read_bytes()

    delete_document(session, data_dir, document, exclude=False)

    answer = messages_of(session, conversation.id)[1]
    assert answer.content == REMOVED_NOTICE and "4821" not in (answer.payload or "")
    assert b"4821" not in database_file.read_bytes()
    # The question the owner asked is theirs and stays.
    assert messages_of(session, conversation.id)[0].content == "What is the garage code?"


def test_answers_that_quoted_nothing_are_not_touched(session: Session) -> None:
    conversation = create_conversation(session)
    append_messages(
        session, conversation, [NewMessage("assistant", "general", "Hello!", {"model": "m"})]
    )

    assert forget_document_passages(session, [1, 2, 3]) == 0

    assert messages_of(session, conversation.id)[0].content == "Hello!"


def test_sources_with_odd_document_numbers_are_ignored_when_recording_citations(
    session: Session,
) -> None:
    conversation = create_conversation(session)
    payload = {
        "sources": [{"document_id": True}, {"document_id": "7"}, {"document_id": 7}, "x", {}]
    }

    (message,) = append_messages(
        session, conversation, [NewMessage("assistant", "notes", "a", payload)]
    )

    assert message.cited_documents == ",7,"
