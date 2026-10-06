from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _config(db_path: Path) -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return config


def _tables(db_path: Path) -> list[str]:
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        return inspect(engine).get_table_names()
    finally:
        engine.dispose()


def _columns(db_path: Path, table: str) -> set[str]:
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        return {column["name"] for column in inspect(engine).get_columns(table)}
    finally:
        engine.dispose()


def test_upgrade_creates_tables_and_downgrade_removes_them(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    config = _config(db_path)

    command.upgrade(config, "head")
    assert {"documents", "chunks"} <= set(_tables(db_path))
    assert "indexed_model" in _columns(db_path, "documents")

    command.downgrade(config, "0002")
    assert "indexed_model" not in _columns(db_path, "documents")

    command.downgrade(config, "0001")
    assert "chunks" not in _tables(db_path)
    assert "documents" in _tables(db_path)

    command.downgrade(config, "base")
    assert "documents" not in _tables(db_path)


def test_the_assistant_tables_are_created_and_removed(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    config = _config(db_path)

    command.upgrade(config, "head")
    assert {"assistant_memories", "assistant_settings"} <= set(_tables(db_path))
    assert _columns(db_path, "assistant_memories") == {"id", "text", "origin", "created_at"}
    assert _columns(db_path, "assistant_settings") == {"id", "key", "value"}

    command.downgrade(config, "0003")
    assert not {"assistant_memories", "assistant_settings"} & set(_tables(db_path))
    assert {"documents", "chunks"} <= set(_tables(db_path))


def test_migration_matches_models(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    config = _config(db_path)

    command.upgrade(config, "head")

    # Raises if the models have drifted from the migrations.
    command.check(config)


def test_the_tracking_migration_keeps_documents_marks_the_profile_and_round_trips(
    tmp_path: Path,
) -> None:
    import sqlite3

    db_path = tmp_path / "test.db"
    config = _config(db_path)
    command.upgrade(config, "0004")
    connection = sqlite3.connect(db_path)
    for number, name in enumerate(["My profile.md", "notes.md"]):
        connection.execute(
            "INSERT INTO documents (original_filename, content_hash, stored_path, size_bytes, "
            "media_type, created_at) VALUES (?, ?, ?, 1, 'text/markdown', '2026-10-05 00:00:00')",
            (name, f"hash{number}", f"files/ha/hash{number}"),
        )
    connection.commit()
    connection.close()

    command.upgrade(config, "head")

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT original_filename, status, kind, supersedes_id FROM documents ORDER BY id"
    ).fetchall()
    connection.close()
    assert rows == [
        ("My profile.md", "active", "profile", None),
        ("notes.md", "active", "note", None),
    ]
    assert _columns(db_path, "document_sources") == {
        "id",
        "document_id",
        "source_root",
        "source_path",
        "first_seen_at",
        "last_seen_at",
        "status",
        "misses",
        "missing_since",
    }

    command.downgrade(config, "0004")
    assert "document_sources" not in _tables(db_path)
    assert not {"status", "kind", "supersedes_id"} & _columns(db_path, "documents")
    command.upgrade(config, "head")
    assert "document_sources" in _tables(db_path)


def test_existing_chunks_take_the_model_of_their_document_and_the_migration_round_trips(
    tmp_path: Path,
) -> None:
    import sqlite3

    db_path = tmp_path / "test.db"
    config = _config(db_path)
    command.upgrade(config, "0005")
    connection = sqlite3.connect(db_path)
    for number, model in enumerate(["model-a", None]):
        connection.execute(
            "INSERT INTO documents (original_filename, content_hash, stored_path, size_bytes, "
            "media_type, created_at, indexed_model) "
            "VALUES (?, ?, ?, 1, 'text/plain', '2026-10-05 00:00:00', ?)",
            (f"n{number}.txt", f"hash{number}", f"files/ha/hash{number}", model),
        )
        connection.execute(
            "INSERT INTO chunks (document_id, chunk_index, text, heading_path, start_line, "
            "end_line, char_count) VALUES (?, 0, 'text', '', 1, 1, 4)",
            (number + 1,),
        )
    connection.commit()
    connection.close()

    command.upgrade(config, "head")

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT document_id, indexed_model FROM chunks ORDER BY id"
    ).fetchall()
    connection.close()
    assert rows == [(1, "model-a"), (2, None)]  # a document that was not indexed stays pending

    command.downgrade(config, "0005")
    assert "indexed_model" not in _columns(db_path, "chunks")
    command.upgrade(config, "head")
    assert "indexed_model" in _columns(db_path, "chunks")


def test_existing_chunks_get_no_page_and_the_page_migration_round_trips(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "test.db"
    config = _config(db_path)
    command.upgrade(config, "0006")
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO documents (original_filename, content_hash, stored_path, size_bytes, "
        "media_type, created_at) VALUES ('a.md', 'h', 'files/h', 1, 'text/markdown', "
        "'2026-10-05 00:00:00')"
    )
    connection.execute(
        "INSERT INTO chunks (document_id, chunk_index, text, heading_path, start_line, "
        "end_line, char_count) VALUES (1, 0, 'text', '', 1, 1, 4)"
    )
    connection.commit()
    connection.close()

    command.upgrade(config, "0007")

    connection = sqlite3.connect(db_path)
    row = connection.execute("SELECT start_page, end_page, text FROM chunks").fetchone()
    connection.close()
    assert row == (None, None, "text")  # a file without pages has none

    command.downgrade(config, "0006")
    assert not {"start_page", "end_page"} & _columns(db_path, "chunks")
    command.upgrade(config, "head")
    assert {"start_page", "end_page"} <= _columns(db_path, "chunks")


def test_the_job_history_table_is_created_empty_and_removed_again(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "test.db"
    config = _config(db_path)
    command.upgrade(config, "0007")
    assert "jobs" not in _tables(db_path)

    command.upgrade(config, "0008")
    assert _columns(db_path, "jobs") == {
        "id",
        "kind",
        "state",
        "message",
        "started_at",
        "finished_at",
        "added",
        "unchanged",
        "replaced",
        "missing",
        "skipped_excluded",
        "failed_files",
        "indexed",
        "chunks_done",
    }
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO jobs (kind, state, started_at) VALUES ('sync', 'done', '2026-10-05 00:00:00')"
    )
    connection.commit()
    row = connection.execute("SELECT added, unchanged, chunks_done FROM jobs").fetchone()
    connection.close()
    assert row == (0, 0, 0)  # the counters start at zero without being given

    command.downgrade(config, "0007")
    assert "jobs" not in _tables(db_path)


def test_the_conversation_tables_are_created_empty_cascade_and_are_removed_again(
    tmp_path: Path,
) -> None:
    import sqlite3

    db_path = tmp_path / "test.db"
    config = _config(db_path)
    command.upgrade(config, "0008")
    assert not {"conversations", "messages"} & set(_tables(db_path))

    command.upgrade(config, "0009")
    assert _columns(db_path, "conversations") == {"id", "title", "created_at", "updated_at"}
    assert _columns(db_path, "messages") == {
        "id",
        "conversation_id",
        "position",
        "role",
        "mode",
        "content",
        "payload",
        "cited_documents",
        "created_at",
    }
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "INSERT INTO conversations (title, created_at, updated_at) "
        "VALUES ('t', '2026-10-05 00:00:00', '2026-10-05 00:00:00')"
    )
    connection.execute(
        "INSERT INTO messages (conversation_id, position, role, mode, content, created_at) "
        "VALUES (1, 0, 'user', 'general', 'hello', '2026-10-05 00:00:00')"
    )
    connection.commit()
    assert connection.execute("SELECT cited_documents FROM messages").fetchone() == ("",)
    # A position is used once per conversation.
    try:
        connection.execute(
            "INSERT INTO messages (conversation_id, position, role, mode, content, created_at) "
            "VALUES (1, 0, 'user', 'general', 'again', '2026-10-05 00:00:00')"
        )
        raise AssertionError("a second message at the same position was accepted")
    except sqlite3.IntegrityError:
        pass
    # Deleting a conversation deletes its messages in the database itself.
    connection.execute("DELETE FROM conversations WHERE id = 1")
    connection.commit()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
    connection.close()

    command.downgrade(config, "0008")
    assert not {"conversations", "messages"} & set(_tables(db_path))
