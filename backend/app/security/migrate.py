"""Encrypt an existing library in place (`python -m app encrypt-library`).

Safe by construction:
- a backup is made first (unless declined), and the state is recorded in the key file, so an
  interrupted run is detected and refused by every other command until it is resumed;
- the work is idempotent: values and files that are already encrypted are skipped, so running it
  again finishes the job without re-encrypting anything;
- before the library is declared encrypted, every value and file is decrypted again and checked;
- SQLite's `secure_delete` is turned on and the database is vacuumed afterwards, so the old
  plaintext does not stay behind inside the database file.

What it cannot do: data that was written to disk in plaintext earlier (the old file contents,
journal pages, Windows shadow copies, an SSD's spare blocks) may survive in space the file system
has not reused yet. Overwrite free space with `cipher /w:<folder>`, or use BitLocker.
"""

import hashlib
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.operations.backup import BackupError, create_backup
from app.security import keystore
from app.security.errors import DecryptionError, KeystoreError, SecurityError
from app.security.vault import TEXT_PREFIX, Vault
from app.storage.database import DB_FILENAME
from app.storage.files import FILES_SUBDIR, file_context, looks_encrypted, write_file_atomic

_BATCH = 500
_MAX_PASSES = 3

# (table, column, the context the model's EncryptedColumn uses) - must match app/storage/models.py
ENCRYPTED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("documents", "original_filename", "documents.original_filename"),
    ("chunks", "text", "chunks.text"),
    ("chunks", "heading_path", "chunks.heading_path"),
)

Progress = Callable[[str], None]
Checkpoint = Callable[[str], None]


@dataclass
class EncryptionReport:
    values_encrypted: int = 0
    files_encrypted: int = 0
    files_total: int = 0
    resumed: bool = False
    backup_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _encrypt_values(
    connection: sqlite3.Connection,
    vault: Vault,
    report: EncryptionReport,
    progress: Progress,
    checkpoint: Checkpoint,
) -> None:
    """Encrypt every plaintext value of the encrypted columns. Safe to repeat."""
    for table, column, context in ENCRYPTED_COLUMNS:
        if not _table_exists(connection, table):
            continue
        last_id = 0
        done = 0
        while True:
            rows = connection.execute(
                f"SELECT id, {column} FROM {table} "  # noqa: S608 - names come from the constant above
                f"WHERE id > ? AND {column} IS NOT NULL AND substr({column}, 1, ?) != ? "
                "ORDER BY id LIMIT ?",
                (last_id, len(TEXT_PREFIX), TEXT_PREFIX, _BATCH),
            ).fetchall()
            if not rows:
                break
            connection.executemany(
                f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608
                [(vault.encrypt_text(value, context), row_id) for row_id, value in rows],
            )
            connection.commit()
            last_id = rows[-1][0]
            done += len(rows)
            report.values_encrypted += len(rows)
            progress(f"encrypted {table}.{column}: {done}")
            checkpoint("values")


def _encrypt_files(
    files_root: Path,
    vault: Vault,
    report: EncryptionReport,
    progress: Progress,
    checkpoint: Checkpoint,
) -> None:
    if not files_root.is_dir():
        return
    paths = sorted(p for p in files_root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    report.files_total = len(paths)
    for path in paths:
        raw = path.read_bytes()
        if looks_encrypted(raw, path.name):
            continue
        if hashlib.sha256(raw).hexdigest() != path.name:
            report.warnings.append(
                "a stored file did not match its checksum; it was encrypted as it is"
            )
        write_file_atomic(path, vault.encrypt_bytes(raw, file_context(path.name)))
        report.files_encrypted += 1
        if report.files_encrypted % 25 == 0:
            progress(f"encrypted stored files: {report.files_encrypted}")
        checkpoint("files")


def plaintext_values(connection: sqlite3.Connection) -> int:
    total = 0
    for table, column, _context in ENCRYPTED_COLUMNS:
        if not _table_exists(connection, table):
            continue
        total += connection.execute(
            f"SELECT COUNT(*) FROM {table} "  # noqa: S608
            f"WHERE {column} IS NOT NULL AND substr({column}, 1, ?) != ?",
            (len(TEXT_PREFIX), TEXT_PREFIX),
        ).fetchone()[0]
    return total


def _verify(connection: sqlite3.Connection, files_root: Path, vault: Vault) -> list[str]:
    """Decrypt everything again. Returns a list of problems (empty means all good)."""
    problems: list[str] = []
    for table, column, context in ENCRYPTED_COLUMNS:
        if not _table_exists(connection, table):
            continue
        for row_id, value in connection.execute(
            f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL"  # noqa: S608
        ):
            try:
                vault.decrypt_text(value, context)
            except DecryptionError:
                problems.append(f"{table}.{column} row {row_id} does not decrypt")
    if files_root.is_dir():
        for path in sorted(p for p in files_root.rglob("*") if p.is_file()):
            raw = path.read_bytes()
            if not looks_encrypted(raw, path.name):
                problems.append("a stored file is not encrypted")
                continue
            try:
                vault.decrypt_bytes(raw, file_context(path.name))
            except DecryptionError:
                problems.append("a stored file does not decrypt")
    return problems


def _check_has_tables(db_path: Path) -> None:
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        if not _table_exists(connection, "documents"):
            raise SecurityError("the database has no tables yet; run `alembic upgrade head` first")
    finally:
        connection.close()


def _check_not_in_use(db_path: Path) -> None:
    connection = sqlite3.connect(db_path, timeout=0.5)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()
    except sqlite3.OperationalError as exc:
        raise SecurityError(
            "the library is in use by another program; stop the API server and any running "
            "command, then try again"
        ) from exc
    finally:
        connection.close()


def encrypt_library(
    data_dir: Path,
    passphrase: str,
    *,
    backup_to: Path | None,
    progress: Progress = lambda _message: None,
    checkpoint: Checkpoint = lambda _phase: None,
) -> EncryptionReport:
    """Encrypt the library in `data_dir`, or finish an interrupted encryption.

    `passphrase` becomes the recovery passphrase (or unlocks an interrupted run). `checkpoint`
    is called after every batch; it exists so tests can interrupt the work.
    """
    report = EncryptionReport()
    state = keystore.library_state(data_dir)
    if state == "encrypted":
        raise KeystoreError("this library is already encrypted")
    db_path = data_dir / DB_FILENAME
    files_root = data_dir / FILES_SUBDIR

    if state == "plaintext":
        keystore.validate_new_passphrase(passphrase)  # fail before anything is changed
        if db_path.is_file():
            _check_not_in_use(db_path)  # first: a locked database cannot even be inspected
            _check_has_tables(db_path)
            needed = 2 * db_path.stat().st_size + 50 * 1024 * 1024
            if shutil.disk_usage(data_dir).free < needed:
                raise SecurityError("there is not enough free disk space to encrypt safely")
            if backup_to is not None:
                progress("making a backup first ...")
                try:
                    report.backup_path = create_backup(data_dir, backup_to).path
                except BackupError as exc:
                    # nothing has been changed yet, so stopping here is always safe
                    raise SecurityError(f"the safety backup could not be made: {exc}") from exc
        vault = keystore.create_keys(data_dir, passphrase, state="migrating")
    else:
        report.resumed = True
        vault = keystore.unlock(data_dir, passphrase=passphrase).vault

    connection = sqlite3.connect(db_path) if db_path.is_file() else None
    try:
        if connection is not None:
            connection.execute("PRAGMA secure_delete = ON")
        for _attempt in range(_MAX_PASSES):
            if connection is not None:
                _encrypt_values(connection, vault, report, progress, checkpoint)
            _encrypt_files(files_root, vault, report, progress, checkpoint)
            leftovers = 0 if connection is None else plaintext_values(connection)
            if leftovers == 0:
                break
            progress(f"{leftovers} value(s) were added during the run; going over them again")
        else:
            raise SecurityError("plaintext values keep appearing; is another program writing?")

        progress("verifying that everything decrypts ...")
        problems = (
            _verify(connection, files_root, vault)
            if connection
            else _verify_files_only(files_root, vault)
        )
        if problems:
            raise SecurityError(
                f"verification failed ({len(problems)} problem(s), first: {problems[0]}); "
                "the library was left in its unfinished state"
            )
        if connection is not None:
            progress("removing leftovers of the old plaintext from the database file ...")
            connection.commit()
            connection.execute("VACUUM")
    finally:
        if connection is not None:
            connection.close()

    keystore.set_state(data_dir, "encrypted")
    return report


def _verify_files_only(files_root: Path, vault: Vault) -> list[str]:
    connection = sqlite3.connect(":memory:")
    try:
        return _verify(connection, files_root, vault)
    finally:
        connection.close()
