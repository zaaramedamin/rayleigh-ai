"""Back up and restore the library (database and stored files) as one verified archive.

What is saved: the SQLite database (copied with SQLite's own backup API, so a database that is
in use is still copied consistently) and the stored copies of your files, plus a manifest with a
SHA-256 checksum of every entry. The search vectors are NOT saved: they are rebuilt from the
chunks with `python -m app index`, and a restored library is marked as "not indexed yet".

A backup contains your private notes. This module never uploads it anywhere.
"""

import hashlib
import json
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app import __version__
from app.storage.database import DB_FILENAME
from app.storage.files import FILES_SUBDIR
from app.storage.migrations import schema_revision_status

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
_CHUNK = 1024 * 1024
_FILE_ENTRY = re.compile(rf"^{FILES_SUBDIR}/([0-9a-f]{{2}})/([0-9a-f]{{64}})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BackupError(RuntimeError):
    """A backup or restore could not be completed. The message says why."""


@dataclass
class BackupSummary:
    path: Path
    documents: int
    chunks: int
    files: int
    archive_bytes: int
    schema_revision: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RestoreSummary:
    target: Path
    documents: int
    chunks: int
    files: int
    schema_revision: str
    needs_migration: bool


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _is_inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def _read_counts(db_path: Path) -> tuple[int, int, str]:
    """(documents, chunks, schema revision) of a database file, opened read-only."""
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except sqlite3.Error as exc:
        raise BackupError(f"the database is not a valid Reyleight database: {exc}") from exc
    finally:
        connection.close()
    if row is None:
        raise BackupError("the database has no schema version (run `alembic upgrade head`)")
    return int(documents), int(chunks), str(row[0])


def create_backup(data_dir: Path, destination: Path) -> BackupSummary:
    """Write a verified backup archive of the library in `data_dir` to `destination`."""
    db_path = data_dir / DB_FILENAME
    if not db_path.is_file():
        raise BackupError(f"no library found: {db_path} does not exist")
    if destination.exists():
        raise BackupError(f"{destination} already exists; choose another name (never overwritten)")
    if _is_inside(destination, data_dir):
        raise BackupError("the backup must be saved outside the data folder it protects")

    destination.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    entries: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(prefix="reyleight-backup-", ignore_cleanup_errors=True) as tmp:
        db_copy = Path(tmp) / DB_FILENAME
        source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        target = sqlite3.connect(db_copy)
        try:
            source.backup(target)  # a consistent snapshot, even if the library is in use
        except sqlite3.Error as exc:
            raise BackupError(f"could not copy the database: {exc}") from exc
        finally:
            target.close()
            source.close()
        documents, chunks, revision = _read_counts(db_copy)

        partial = destination.with_name(destination.name + ".partial")
        try:
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(db_copy, DB_FILENAME)
                entries[DB_FILENAME] = {
                    "sha256": _sha256_of(db_copy),
                    "size": db_copy.stat().st_size,
                }

                files_root = data_dir / FILES_SUBDIR
                for path in sorted(files_root.rglob("*")) if files_root.is_dir() else []:
                    if not path.is_file():
                        continue
                    name = path.relative_to(data_dir).as_posix()
                    digest = _sha256_of(path)
                    if digest != path.name:
                        warnings.append(f"stored file {path.name[:12]}... does not match its hash")
                    archive.write(path, name)
                    entries[name] = {"sha256": digest, "size": path.stat().st_size}

                manifest = {
                    "format": FORMAT_VERSION,
                    "app_version": __version__,
                    "created_at": datetime.now(UTC).isoformat(),
                    "schema_revision": revision,
                    "documents": documents,
                    "chunks": chunks,
                    "entries": entries,
                }
                archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    return BackupSummary(
        path=destination,
        documents=documents,
        chunks=chunks,
        files=len(entries) - 1,
        archive_bytes=destination.stat().st_size,
        schema_revision=revision,
        warnings=warnings,
    )


def _load_manifest(archive: zipfile.ZipFile) -> dict[str, object]:
    try:
        manifest = json.loads(archive.read(MANIFEST_NAME))
    except (KeyError, ValueError) as exc:
        raise BackupError("this is not a Reyleight backup (no readable manifest)") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_VERSION:
        raise BackupError("this backup was made by an incompatible version")
    if not isinstance(manifest.get("entries"), dict):
        raise BackupError("the backup manifest is damaged")
    return manifest


def _check_entry_names(names: set[str], listed: set[str]) -> None:
    for name in listed:
        if name != DB_FILENAME and not _FILE_ENTRY.match(name):
            raise BackupError(f"the backup lists an unexpected entry: {name!r}")
    if DB_FILENAME not in listed:
        raise BackupError("the backup has no database")
    unlisted = names - listed - {MANIFEST_NAME}
    if unlisted:
        raise BackupError(
            f"the archive contains entries the manifest does not list: {sorted(unlisted)[:3]}"
        )
    missing = listed - names
    if missing:
        raise BackupError(f"the archive is missing entries: {sorted(missing)[:3]}")


def restore_backup(archive_path: Path, target_dir: Path) -> RestoreSummary:
    """Verify a backup, then restore it into an empty `target_dir`."""
    if not archive_path.is_file():
        raise BackupError(f"backup file not found: {archive_path}")
    if target_dir.exists() and any(target_dir.iterdir()):
        raise BackupError(f"{target_dir} is not empty; restore into an empty folder")

    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise BackupError("this file is not a valid backup archive") from exc

    with (
        archive,
        tempfile.TemporaryDirectory(prefix="reyleight-restore-", ignore_cleanup_errors=True) as tmp,
    ):
        manifest = _load_manifest(archive)
        entries = manifest["entries"]
        assert isinstance(entries, dict)
        _check_entry_names({i.filename for i in archive.infolist()}, set(entries))

        staging = Path(tmp)
        for name, info in entries.items():
            expected = info.get("sha256") if isinstance(info, dict) else None
            size = info.get("size") if isinstance(info, dict) else None
            if (
                not isinstance(expected, str)
                or not _SHA256.match(expected)
                or not isinstance(size, int)
            ):
                raise BackupError(f"the manifest entry for {name!r} is damaged")
            match = _FILE_ENTRY.match(name)
            if match and match.group(2) != expected:
                raise BackupError(f"stored file {name!r} does not match its own hash")

            destination = staging / name  # the name was validated against a strict pattern
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with archive.open(name) as source, destination.open("wb") as out:
                while block := source.read(_CHUNK):
                    written += len(block)
                    if written > size:
                        raise BackupError(f"{name!r} is larger than the manifest says")
                    digest.update(block)
                    out.write(block)
            if written != size or digest.hexdigest() != expected:
                raise BackupError(f"checksum mismatch for {name!r}: the backup is damaged")

        db_path = staging / DB_FILENAME
        documents, chunks, revision = _read_counts(db_path)
        needs_migration = schema_revision_status(revision) == "older"
        if schema_revision_status(revision) == "unknown":
            raise BackupError(
                f"the backup's database version ({revision}) is newer than this program "
                "or unknown; update Reyleight and try again"
            )

        # The vectors are not part of a backup, so nothing is searchable until `index` runs.
        connection = sqlite3.connect(db_path)
        try:
            check = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise BackupError(f"the database inside the backup is damaged: {check}")
            connection.execute("UPDATE documents SET indexed_model = NULL")
            connection.commit()
        except sqlite3.OperationalError:
            connection.rollback()  # an older schema without the column: nothing to reset
        finally:
            connection.close()

        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                final = target_dir / path.relative_to(staging)
                final.parent.mkdir(parents=True, exist_ok=True)
                path.replace(final)

    return RestoreSummary(
        target=target_dir,
        documents=documents,
        chunks=chunks,
        files=len(entries) - 1,
        schema_revision=revision,
        needs_migration=needs_migration,
    )
