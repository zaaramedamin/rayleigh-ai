import sqlite3
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import ConnectionPoolEntry

from app.security import keystore
from app.security.errors import KeystoreError, MigrationIncompleteError
from app.security.sqlalchemy_types import attach_vault
from app.security.vault import TEXT_PREFIX

DB_FILENAME = "reyleight.db"


class Base(DeclarativeBase):
    pass


def database_url(data_dir: Path) -> str:
    return f"sqlite:///{(data_dir / DB_FILENAME).as_posix()}"


def _contains_encrypted_values(db_path: Path) -> bool:
    """True if the database already holds encrypted text (read-only look)."""
    if not db_path.is_file():
        return False
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT 1 FROM chunks WHERE substr(text, 1, ?) = ? LIMIT 1",
            (len(TEXT_PREFIX), TEXT_PREFIX),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False  # no such table yet: a brand-new library
    finally:
        connection.close()


def create_db_engine(data_dir: Path, *, passphrase: str | None = None) -> Engine:
    """Create a SQLite engine whose database file lives inside data_dir.

    An encrypted library is unlocked here: with this Windows account's key, or else with
    `passphrase`. If neither works, LibraryLockedError is raised. Nothing is ever written to an
    encrypted library without its key, and an interrupted encryption must be finished first.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    state = keystore.library_state(data_dir)
    if state == "migrating":
        raise MigrationIncompleteError(
            "encrypting this library was started but not finished; "
            "run `python -m app encrypt-library` to resume"
        )
    if state == "plaintext" and _contains_encrypted_values(data_dir / DB_FILENAME):
        raise KeystoreError(
            "this library contains encrypted data but its key file (security.json) is missing; "
            "restore security.json from a backup"
        )

    engine = create_engine(database_url(data_dir))
    if state == "encrypted":
        attach_vault(engine, keystore.unlock(data_dir, passphrase=passphrase).vault)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(
        dbapi_connection: DBAPIConnection, _connection_record: ConnectionPoolEntry
    ) -> None:
        # SQLite ignores foreign keys (and ON DELETE CASCADE) unless this is enabled per connection.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine
