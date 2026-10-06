import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.chunking.service import chunk_document
from app.operations.backup import BackupError, create_backup, restore_backup
from app.storage.database import DB_FILENAME, create_db_engine
from app.storage.files import save_file
from app.storage.migrations import latest_revision
from app.storage.models import Chunk, Document

NOTES = {
    "oats.md": "# Oats\n\nOats are a whole grain.\n\n## Cooking\n\nSimmer for five minutes.\n",
    "rice.txt": "Rice needs twice its volume of water.",
}


@pytest.fixture
def library(migrated_data_dir: Path) -> Path:
    """A real, migrated library with two documents, their chunks and stored files."""
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        for name, text in NOTES.items():
            document = save_file(session, migrated_data_dir, text.encode(), name)
            chunk_document(session, migrated_data_dir, document, 1000, 100)
            document.indexed_model = "test/model"  # pretend it was indexed
        session.commit()
    engine.dispose()
    return migrated_data_dir


@pytest.fixture
def archive(library: Path, tmp_path: Path) -> Path:
    path = tmp_path / "backups" / "library.zip"
    create_backup(library, path)
    return path


def _counts(data_dir: Path) -> tuple[int, int]:
    engine = create_db_engine(data_dir)
    with Session(engine) as session:
        result = (
            len(session.scalars(select(Document)).all()),
            len(session.scalars(select(Chunk)).all()),
        )
    engine.dispose()
    return result


def _rewrite_zip(source: Path, target: Path, change: dict[str, bytes | None]) -> None:
    """Copy a zip, replacing (bytes) or dropping (None) the named entries."""
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as out:
        for info in src.infolist():
            if info.filename in change:
                if change[info.filename] is None:
                    continue
                out.writestr(info.filename, change[info.filename] or b"")
            else:
                out.writestr(info.filename, src.read(info.filename))


# --- creating a backup ------------------------------------------------------------------------


def test_backup_contains_the_database_the_files_and_a_manifest(
    library: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "out" / "b.zip"

    summary = create_backup(library, destination)

    assert summary.documents == 2
    assert summary.chunks >= 2
    assert summary.files == 2
    assert summary.schema_revision == latest_revision()
    assert summary.warnings == []
    with zipfile.ZipFile(destination) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
    assert {DB_FILENAME, "manifest.json"} <= names
    assert sum(n.startswith("files/") for n in names) == 2
    assert manifest["documents"] == 2
    assert all(len(e["sha256"]) == 64 for e in manifest["entries"].values())
    assert not destination.with_name("b.zip.partial").exists()


def test_the_vectors_are_not_part_of_a_backup(library: Path, tmp_path: Path) -> None:
    (library / "qdrant").mkdir()
    (library / "qdrant" / "collection.bin").write_bytes(b"vectors")
    destination = tmp_path / "b.zip"

    create_backup(library, destination)

    with zipfile.ZipFile(destination) as z:
        assert not any("qdrant" in n for n in z.namelist())


def test_a_backup_is_never_overwritten(library: Path, archive: Path) -> None:
    with pytest.raises(BackupError, match="already exists"):
        create_backup(library, archive)


def test_a_backup_cannot_be_saved_inside_the_data_folder(library: Path) -> None:
    with pytest.raises(BackupError, match="outside the data folder"):
        create_backup(library, library / "inside.zip")


def test_backing_up_a_missing_library_is_a_clear_error(data_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="no library found"):
        create_backup(data_dir, tmp_path / "b.zip")


def test_a_damaged_stored_file_is_reported_but_the_backup_still_works(
    library: Path, tmp_path: Path
) -> None:
    stored = next((library / "files").rglob("*.*"), None) or next(
        p for p in (library / "files").rglob("*") if p.is_file()
    )
    stored.write_bytes(b"corrupted on disk")

    summary = create_backup(library, tmp_path / "b.zip")

    assert len(summary.warnings) == 1
    assert "does not match its hash" in summary.warnings[0]


def test_the_copy_is_consistent_even_while_another_connection_is_writing(
    library: Path, tmp_path: Path
) -> None:
    writer = sqlite3.connect(library / DB_FILENAME)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "INSERT INTO documents (original_filename, content_hash, stored_path, size_bytes,"
        " media_type, created_at)"
        " VALUES ('uncommitted.txt', 'x', 'files/x/x', 1, 't', '2026-01-01')"
    )

    try:
        summary = create_backup(library, tmp_path / "b.zip")
    finally:
        writer.rollback()
        writer.close()

    assert summary.documents == 2  # the uncommitted row is not in the backup


# --- restoring --------------------------------------------------------------------------------


def test_a_restore_recreates_the_library_exactly(
    library: Path, archive: Path, tmp_path: Path
) -> None:
    target = tmp_path / "restored"

    summary = restore_backup(archive, target)

    assert (summary.documents, summary.files, summary.needs_migration) == (2, 2, False)
    assert _counts(target) == _counts(library)
    originals = sorted(p for p in (library / "files").rglob("*") if p.is_file())
    restored = sorted(p for p in (target / "files").rglob("*") if p.is_file())
    assert [p.name for p in restored] == [p.name for p in originals]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(originals, restored, strict=True))


def test_a_restored_library_is_marked_as_not_searchable_yet(archive: Path, tmp_path: Path) -> None:
    target = tmp_path / "restored"

    restore_backup(archive, target)

    connection = sqlite3.connect(target / DB_FILENAME)
    try:
        values = connection.execute("SELECT DISTINCT indexed_model FROM documents").fetchall()
    finally:
        connection.close()
    assert values == [(None,)]


def test_restoring_never_changes_the_original_library(
    library: Path, archive: Path, tmp_path: Path
) -> None:
    before = (library / DB_FILENAME).read_bytes()

    restore_backup(archive, tmp_path / "restored")

    assert (library / DB_FILENAME).read_bytes() == before


def test_a_non_empty_target_is_refused_and_left_untouched(archive: Path, tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "important.txt").write_text("keep me")

    with pytest.raises(BackupError, match="not empty"):
        restore_backup(archive, target)

    assert [p.name for p in target.iterdir()] == ["important.txt"]


def test_an_existing_empty_target_is_fine(archive: Path, tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()

    assert restore_backup(archive, target).documents == 2


def test_a_tampered_stored_file_is_detected_and_nothing_is_written(
    archive: Path, tmp_path: Path
) -> None:
    with zipfile.ZipFile(archive) as z:
        victim = next(n for n in z.namelist() if n.startswith("files/"))
    bad = tmp_path / "tampered.zip"
    _rewrite_zip(archive, bad, {victim: b"someone changed this"})
    target = tmp_path / "restored"

    with pytest.raises(BackupError, match="checksum mismatch"):
        restore_backup(bad, target)

    assert not target.exists() or not any(target.iterdir())


def test_a_tampered_database_is_detected(archive: Path, tmp_path: Path) -> None:
    bad = tmp_path / "tampered.zip"
    _rewrite_zip(archive, bad, {DB_FILENAME: b"not a database"})

    with pytest.raises(BackupError, match="checksum mismatch"):
        restore_backup(bad, tmp_path / "restored")


def test_a_missing_entry_is_detected(archive: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        victim = next(n for n in z.namelist() if n.startswith("files/"))
    bad = tmp_path / "missing.zip"
    _rewrite_zip(archive, bad, {victim: None})

    with pytest.raises(BackupError, match="missing entries"):
        restore_backup(bad, tmp_path / "restored")


def test_an_extra_unlisted_entry_is_detected(archive: Path, tmp_path: Path) -> None:
    bad = tmp_path / "extra.zip"
    shutil.copy(archive, bad)
    with zipfile.ZipFile(bad, "a") as z:
        z.writestr("files/ab/" + "0" * 64, b"smuggled")

    with pytest.raises(BackupError, match="manifest does not list"):
        restore_backup(bad, tmp_path / "restored")


@pytest.mark.parametrize(
    "name",
    ["../evil.txt", "files/../../evil.txt", "C:/evil.txt", "files/zz/" + "a" * 64, "other.db"],
)
def test_unsafe_entry_names_are_refused_even_with_a_valid_looking_manifest(
    archive: Path, tmp_path: Path, name: str
) -> None:
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["entries"][name] = {"sha256": "a" * 64, "size": 1}
    bad = tmp_path / "unsafe.zip"
    _rewrite_zip(archive, bad, {"manifest.json": json.dumps(manifest).encode()})
    with zipfile.ZipFile(bad, "a") as z:
        z.writestr(name, b"x")
    target = tmp_path / "restored"

    with pytest.raises(BackupError, match="unexpected entry"):
        restore_backup(bad, target)

    assert not (tmp_path / "evil.txt").exists()
    assert not target.exists() or not any(target.iterdir())


def test_a_file_that_does_not_match_its_own_hash_name_is_refused(
    archive: Path, tmp_path: Path
) -> None:
    import hashlib

    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        victim = next(n for n in manifest["entries"] if n.startswith("files/"))
    swapped_in = b"plain content stored under the wrong name"
    # an internally consistent archive: the manifest matches the new bytes, but the file name
    # (which must be the hash of the content) does not
    manifest["entries"][victim] = {
        "sha256": hashlib.sha256(swapped_in).hexdigest(),
        "size": len(swapped_in),
    }
    bad = tmp_path / "renamed.zip"
    _rewrite_zip(archive, bad, {"manifest.json": json.dumps(manifest).encode(), victim: swapped_in})

    with pytest.raises(BackupError, match="does not match its own hash"):
        restore_backup(bad, tmp_path / "restored")


def test_an_archive_that_lies_about_sizes_is_stopped(archive: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["entries"][DB_FILENAME]["size"] = 1
    bad = tmp_path / "bomb.zip"
    _rewrite_zip(archive, bad, {"manifest.json": json.dumps(manifest).encode()})

    with pytest.raises(BackupError, match="larger than the manifest says"):
        restore_backup(bad, tmp_path / "restored")


def test_a_database_from_a_newer_version_is_refused(library: Path, tmp_path: Path) -> None:
    connection = sqlite3.connect(library / DB_FILENAME)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()
    future = tmp_path / "future.zip"
    create_backup(library, future)

    with pytest.raises(BackupError, match="newer than this program"):
        restore_backup(future, tmp_path / "restored")


def test_a_database_from_an_older_version_restores_and_asks_for_a_migration(
    library: Path, tmp_path: Path
) -> None:
    connection = sqlite3.connect(library / DB_FILENAME)
    connection.execute("UPDATE alembic_version SET version_num = '0002'")
    connection.execute("ALTER TABLE documents DROP COLUMN indexed_model")
    connection.commit()
    connection.close()
    older = tmp_path / "older.zip"
    create_backup(library, older)

    summary = restore_backup(older, tmp_path / "restored")

    assert summary.needs_migration is True
    assert summary.schema_revision == "0002"


def test_files_that_are_not_backups_are_rejected_clearly(tmp_path: Path) -> None:
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"this is not a zip file")
    not_a_backup = tmp_path / "other.zip"
    with zipfile.ZipFile(not_a_backup, "w") as z:
        z.writestr("hello.txt", "hi")

    with pytest.raises(BackupError, match="not a valid backup archive"):
        restore_backup(junk, tmp_path / "r1")
    with pytest.raises(BackupError, match="no readable manifest"):
        restore_backup(not_a_backup, tmp_path / "r2")
    with pytest.raises(BackupError, match="not found"):
        restore_backup(tmp_path / "missing.zip", tmp_path / "r3")


def test_the_restored_library_works_with_the_real_migrations(archive: Path, tmp_path: Path) -> None:
    from app.storage.migrations import database_is_up_to_date

    target = tmp_path / "restored"
    restore_backup(archive, target)

    engine = create_db_engine(target)
    try:
        assert database_is_up_to_date(engine)
    finally:
        engine.dispose()
