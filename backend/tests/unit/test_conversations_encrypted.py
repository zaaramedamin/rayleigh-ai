"""Stored conversations in an encrypted library."""

import sqlite3
from pathlib import Path

from sqlalchemy.orm import Session

from app.assistant.conversations import (
    NewMessage,
    append_messages,
    create_conversation,
    list_conversations,
    messages_of,
)
from app.security.migrate import encrypt_library
from app.security.vault import TEXT_PREFIX
from app.storage.database import DB_FILENAME, create_db_engine

PASSPHRASE = "correct horse battery staple"
TITLE = "Garage-Plans-Zephyr"
SECRET = "ZEBRA-GARAGE-4821"
SOURCE_TEXT = "Hidden-Source-Passage-Quokka"


def make_conversation(folder: Path) -> None:
    engine = create_db_engine(folder)
    with Session(engine) as session:
        conversation = create_conversation(session, TITLE)
        append_messages(
            session,
            conversation,
            [
                NewMessage("user", "general", f"My code is {SECRET}"),
                NewMessage(
                    "assistant",
                    "notes",
                    f"The code is {SECRET} [1].",
                    {"sources": [{"document_id": 2, "text": SOURCE_TEXT}]},
                ),
            ],
        )
    engine.dispose()


def readable_in(folder: Path, needles: list[str]) -> list[str]:
    return [
        path.name
        for path in folder.rglob("*")
        if path.is_file() and any(n.encode() in path.read_bytes() for n in needles)
    ]


def test_the_conversation_is_encrypted_with_the_library_and_still_readable(
    migrated_data_dir: Path,
) -> None:
    make_conversation(migrated_data_dir)
    assert readable_in(migrated_data_dir, [TITLE, SECRET, SOURCE_TEXT]), "needs plaintext first"

    encrypt_library(migrated_data_dir, PASSPHRASE, backup_to=None)

    assert readable_in(migrated_data_dir, [TITLE, SECRET, SOURCE_TEXT]) == []
    connection = sqlite3.connect(migrated_data_dir / DB_FILENAME)
    try:
        for table, column in (
            ("conversations", "title"),
            ("messages", "content"),
            ("messages", "payload"),
        ):
            rows = connection.execute(f"SELECT {column} FROM {table}")
            values = [r[0] for r in rows if r[0] is not None]  # a question has no details
            assert values and all(v.startswith(TEXT_PREFIX) for v in values), (table, column)
        # What stays readable is structure, never words: numbers and which document was quoted.
        assert connection.execute("SELECT cited_documents FROM messages").fetchall() == [
            ("",),
            (",2,",),
        ]
    finally:
        connection.close()

    engine = create_db_engine(migrated_data_dir)  # unlocks through this Windows account
    with Session(engine) as session:
        (summary,) = list_conversations(session)
        assert (summary.title, summary.message_count) == (TITLE, 2)
        user, answer = messages_of(session, summary.id)
        assert SECRET in user.content and SOURCE_TEXT in (answer.payload or "")
        # A conversation written after encryption is encrypted too.
        later = create_conversation(session, "Later-Title-Marmot")
        append_messages(session, later, [NewMessage("user", "general", "Later-Words-Ibex")])
    engine.dispose()
    assert readable_in(migrated_data_dir, ["Later-Title-Marmot", "Later-Words-Ibex"]) == []
