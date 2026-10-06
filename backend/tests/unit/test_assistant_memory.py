import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.assistant.identity import (
    DEFAULT_ROLE,
    MAX_ADDRESS_CHARS,
    MAX_NAME_CHARS,
    MAX_ROLE_CHARS,
    Identity,
    load_identity,
    save_identity,
)
from app.assistant.memory import (
    MAX_MEMORIES,
    MAX_MEMORY_CHARS,
    MemoryFullError,
    add_memory,
    clear_memories,
    delete_memory,
    list_memories,
    memories_for_prompt,
)
from app.security.migrate import encrypt_library
from app.storage.database import DB_FILENAME, create_db_engine

# --- memory -----------------------------------------------------------------------------------


def test_a_memory_is_kept_and_listed_oldest_first(session: Session) -> None:
    add_memory(session, "I live in Lyon", "owner")
    add_memory(session, "My sister is called Mia", "assistant")

    memories = list_memories(session)

    assert [(m.text, m.origin) for m in memories] == [
        ("I live in Lyon", "owner"),
        ("My sister is called Mia", "assistant"),
    ]
    assert all(m.created_at is not None for m in memories)


def test_the_same_fact_is_not_saved_twice(session: Session) -> None:
    first, created = add_memory(session, "I prefer short answers.", "owner")
    again, created_again = add_memory(session, "  i prefer SHORT answers  ", "assistant")

    assert (created, created_again) == (True, False)
    assert again.id == first.id
    assert len(list_memories(session)) == 1


def test_a_memory_is_one_bounded_plain_line(session: Session) -> None:
    memory, _ = add_memory(session, "line one\nline two\x00\t" + "x" * 1000, "assistant")

    assert "\n" not in memory.text and "\x00" not in memory.text
    assert memory.text.startswith("line one line two x")
    assert len(memory.text) == MAX_MEMORY_CHARS


@pytest.mark.parametrize("text", ["", "   ", "\n\t\x00"])
def test_nothing_is_not_a_memory(session: Session, text: str) -> None:
    with pytest.raises(ValueError, match="nothing to remember"):
        add_memory(session, text, "owner")
    assert list_memories(session) == []


def test_a_full_memory_refuses_more_until_something_is_deleted(session: Session) -> None:
    for number in range(MAX_MEMORIES):
        add_memory(session, f"fact number {number}", "owner")

    with pytest.raises(MemoryFullError, match="full"):
        add_memory(session, "one more", "assistant")

    assert delete_memory(session, list_memories(session)[0].id) is True
    assert add_memory(session, "one more", "assistant")[1] is True


def test_deleting_and_clearing(session: Session) -> None:
    kept, _ = add_memory(session, "keep me", "owner")
    gone, _ = add_memory(session, "delete me", "owner")

    assert delete_memory(session, gone.id) is True
    assert delete_memory(session, gone.id) is False  # already gone
    assert [m.id for m in list_memories(session)] == [kept.id]

    assert clear_memories(session) == 1
    assert list_memories(session) == []
    assert clear_memories(session) == 0


def test_the_prompt_gets_the_newest_memories_that_fit_oldest_first() -> None:
    texts = ["a" * 40, "b" * 40, "c" * 40]

    assert memories_for_prompt(texts, budget=90) == texts[1:]
    assert memories_for_prompt(texts, budget=1000) == texts
    assert memories_for_prompt(texts, budget=10) == []


# --- identity ---------------------------------------------------------------------------------


def test_a_new_library_has_the_default_identity(session: Session) -> None:
    identity = load_identity(session)

    assert identity == Identity()
    assert (identity.name, identity.address, identity.role) == ("Reyleight", "sir", DEFAULT_ROLE)
    assert identity.use_profile is True and identity.use_memory is True


def test_the_identity_survives_a_round_trip(session: Session) -> None:
    wanted = Identity(
        name="Jarvis", address="boss", role="Run my lab.", use_profile=False, use_memory=False
    )

    assert save_identity(session, wanted) == wanted
    assert load_identity(session) == wanted

    # Saving again updates the same rows instead of adding more.
    save_identity(session, Identity(name="Friday"))
    assert load_identity(session).name == "Friday"


def test_the_identity_is_cleaned_and_bounded(session: Session) -> None:
    saved = save_identity(
        session,
        Identity(
            name="  Jar\x00vis\n" + "x" * 100,
            address="s" * 100,
            role="Line one.\r\nLine two.\x07" + "y" * 5000,
        ),
    )

    assert saved.name.startswith("Jarvis x") and len(saved.name) == MAX_NAME_CHARS
    assert len(saved.address) == MAX_ADDRESS_CHARS
    assert saved.role.startswith("Line one.\nLine two.y") and len(saved.role) == MAX_ROLE_CHARS


def test_an_empty_name_or_role_goes_back_to_the_default_and_no_address_is_allowed(
    session: Session,
) -> None:
    saved = save_identity(session, Identity(name="  ", address="", role="\n"))

    assert (saved.name, saved.address, saved.role) == ("Reyleight", "", DEFAULT_ROLE)


# --- encryption at rest -----------------------------------------------------------------------


def test_memories_and_the_role_are_encrypted_with_the_library(migrated_data_dir: Path) -> None:
    secret_fact, secret_role = "My safe code is ZEBRA-4821", "Guard the PHOENIX project."
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        add_memory(session, secret_fact, "owner")
        save_identity(session, Identity(role=secret_role))
    engine.dispose()
    db_path = migrated_data_dir / DB_FILENAME
    assert b"ZEBRA-4821" in db_path.read_bytes()

    encrypt_library(migrated_data_dir, "correct horse battery staple", backup_to=None)

    raw = db_path.read_bytes()
    assert b"ZEBRA-4821" not in raw and b"PHOENIX" not in raw
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        assert [m.text for m in list_memories(session)] == [secret_fact]
        assert load_identity(session).role == secret_role
        # What is saved after encryption is encrypted as well, and still found as a duplicate.
        add_memory(session, "My new code is OTTER-77", "assistant")
        assert add_memory(session, secret_fact, "assistant")[1] is False
    engine.dispose()
    assert b"OTTER-77" not in db_path.read_bytes()
    connection = sqlite3.connect(db_path)
    try:
        keys = [row[0] for row in connection.execute("SELECT key FROM assistant_settings")]
    finally:
        connection.close()
    assert "role" in keys  # setting names stay readable; their values do not
