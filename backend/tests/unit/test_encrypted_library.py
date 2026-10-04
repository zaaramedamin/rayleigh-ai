import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.knowledge.answering.service import compose_answer
from app.knowledge.chunking.service import chunk_document
from app.knowledge.indexing.service import index_pending
from app.knowledge.retrieval.service import retrieve
from app.operations.backup import restore_backup
from app.security import keystore
from app.security.errors import (
    DecryptionError,
    KeystoreError,
    LibraryLockedError,
    MigrationIncompleteError,
    SecurityError,
    WrongPassphraseError,
)
from app.security.migrate import encrypt_library
from app.security.sqlalchemy_types import attach_vault, vault_of
from app.security.vault import TEXT_PREFIX, Vault
from app.storage.database import DB_FILENAME, create_db_engine, database_url
from app.storage.files import read_file, save_file
from app.storage.models import Chunk, Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import FakeLLM, HashingEmbedder

PASSPHRASE = "correct horse battery staple"
SECRET_TEXT = "ZEBRA-GARAGE-4821"
SECRET_HEADING = "Heading Phoenix"
SECRET_NAME = "secret-plans.md"
NEEDLES = [SECRET_TEXT.encode(), SECRET_HEADING.encode(), SECRET_NAME.encode(), b"Oats are"]


@pytest.fixture
def plain_library(migrated_data_dir: Path) -> Path:
    """A real plaintext library holding recognisable secrets in a name, headings and text."""
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        for name, text in {
            SECRET_NAME: (
                f"# {SECRET_HEADING}\n\nThe code is {SECRET_TEXT}.\n\n## Sub\n\nMore text.\n"
            ),
            "oats.txt": "Oats are a whole grain rich in fibre.",
        }.items():
            document = save_file(session, migrated_data_dir, text.encode(), name)
            chunk_document(session, migrated_data_dir, document, 1000, 100)
    engine.dispose()
    return migrated_data_dir


def files_containing(root: Path, needles: list[bytes]) -> list[str]:
    """Every file under `root` whose raw bytes contain any needle."""
    hits = []
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if any(needle in data for needle in needles):
                hits.append(path.relative_to(root).as_posix())
    return hits


def drop_windows_copy(folder: Path) -> None:
    keyfile = keystore.read_keyfile(folder)
    assert keyfile is not None
    keystore._write_keyfile(folder, keystore._replace(keyfile, wrapped_by_windows=None))


def encrypt(folder: Path, backup: Path | None = None, **kwargs: object) -> object:
    return encrypt_library(folder, PASSPHRASE, backup_to=backup, **kwargs)  # type: ignore[arg-type]


# --- the whole point: nothing readable is left on disk ----------------------------------------


def test_before_encryption_the_secrets_are_readable_on_disk(plain_library: Path) -> None:
    assert files_containing(plain_library, NEEDLES), "the test library must start in plaintext"


def test_after_encryption_no_secret_appears_in_any_file_of_the_data_folder(
    plain_library: Path, tmp_path: Path
) -> None:
    encrypt(plain_library, backup=tmp_path / "backup.zip")

    assert files_containing(plain_library, NEEDLES) == []
    assert sorted(p.name for p in plain_library.iterdir() if p.is_file()) == [
        DB_FILENAME,
        "security.json",
    ]


def test_the_application_still_reads_everything_after_encryption(
    plain_library: Path, tmp_path: Path
) -> None:
    encrypt(plain_library, backup=tmp_path / "backup.zip")

    engine = create_db_engine(plain_library)  # unlocks through this Windows account
    with Session(engine) as session:
        names = {d.original_filename for d in session.scalars(select(Document))}
        chunks = list(session.scalars(select(Chunk)))
        assert SECRET_NAME in names
        assert any(SECRET_TEXT in c.text for c in chunks)
        assert any(c.heading_path == SECRET_HEADING for c in chunks)
        assert any(c.heading_path == f"{SECRET_HEADING} > Sub" for c in chunks)
        secret_doc = next(
            d for d in session.scalars(select(Document)) if d.original_filename == SECRET_NAME
        )
        assert SECRET_TEXT.encode() in read_file(plain_library, secret_doc)
    engine.dispose()


def test_raw_database_values_are_ciphertext_and_structure_stays_readable(
    plain_library: Path,
) -> None:
    encrypt(plain_library, backup=None)

    connection = sqlite3.connect(plain_library / DB_FILENAME)
    try:
        for table, column in (
            ("documents", "original_filename"),
            ("chunks", "text"),
            ("chunks", "heading_path"),
        ):
            values = [r[0] for r in connection.execute(f"SELECT {column} FROM {table}")]
            assert values and all(v.startswith(TEXT_PREFIX) for v in values), (table, column)
        # structure and numbers are not secret and stay queryable
        assert connection.execute("SELECT MIN(start_line) FROM chunks").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    finally:
        connection.close()


def test_new_documents_added_after_encryption_are_encrypted_too(
    plain_library: Path, tmp_path: Path
) -> None:
    encrypt(plain_library, backup=None)
    engine = create_db_engine(plain_library)
    with Session(engine) as session:
        document = save_file(
            session, plain_library, b"A brand new note: PLATYPUS-9917", "new-note-name.txt"
        )
        chunk_document(session, plain_library, document, 1000, 100)
        assert read_file(plain_library, document) == b"A brand new note: PLATYPUS-9917"
    engine.dispose()

    assert files_containing(plain_library, [b"PLATYPUS-9917", b"new-note-name"]) == []


def test_search_and_answers_work_on_an_encrypted_library_and_leave_no_plaintext_behind(
    plain_library: Path,
) -> None:
    encrypt(plain_library, backup=None)
    embedder = HashingEmbedder()
    engine = create_db_engine(plain_library)
    store = QdrantVectorStore.open_local(plain_library / "qdrant", "test", embedder.dimension)
    with Session(engine) as session:
        index_pending(session, embedder, store)
        results = retrieve(session, embedder, store, "what is the garage code", top_k=3)
        assert results[0].source == SECRET_NAME
        assert SECRET_TEXT in results[0].text
        answer = compose_answer(
            FakeLLM("The code is ZEBRA-GARAGE-4821 [1]."), "garage code?", results, min_score=0.1
        )
        assert answer.grounded and answer.sources[0].source == SECRET_NAME
    store.close()
    engine.dispose()

    # the strongest end-to-end claim: after indexing and searching, still nothing readable on disk
    assert files_containing(plain_library, NEEDLES) == []


# --- the backup made first ---------------------------------------------------------------------


def test_the_backup_taken_before_encrypting_restores_the_original_library(
    plain_library: Path, tmp_path: Path
) -> None:
    backup = tmp_path / "before.zip"

    report = encrypt(plain_library, backup=backup)

    assert report.backup_path == backup  # type: ignore[attr-defined]
    restored = tmp_path / "restored"
    restore_backup(backup, restored)
    connection = sqlite3.connect(restored / DB_FILENAME)
    try:
        names = [r[0] for r in connection.execute("SELECT original_filename FROM documents")]
    finally:
        connection.close()
    assert SECRET_NAME in names  # the safety backup is the plaintext original, as warned


def test_no_backup_is_made_when_declined(plain_library: Path, tmp_path: Path) -> None:
    report = encrypt(plain_library, backup=None)

    assert report.backup_path is None  # type: ignore[attr-defined]
    assert not any(tmp_path.glob("*.zip"))


# --- interruption and resuming ---------------------------------------------------------------


@pytest.mark.parametrize("phase", ["values", "files"])
def test_an_interrupted_encryption_is_refused_everywhere_until_it_is_resumed(
    plain_library: Path, phase: str
) -> None:
    class Interrupted(Exception):
        pass

    def stop(current: str) -> None:
        if current == phase:
            raise Interrupted

    with pytest.raises(Interrupted):
        encrypt(plain_library, backup=None, checkpoint=stop)

    assert keystore.library_state(plain_library) == "migrating"
    with pytest.raises(MigrationIncompleteError, match="resume"):
        create_db_engine(plain_library)

    report = encrypt(plain_library, backup=None)
    assert report.resumed is True  # type: ignore[attr-defined]
    assert keystore.library_state(plain_library) == "encrypted"
    assert files_containing(plain_library, NEEDLES) == []
    engine = create_db_engine(plain_library)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Chunk)) == 3
    engine.dispose()


def test_resuming_does_not_encrypt_twice(plain_library: Path) -> None:
    class Interrupted(Exception):
        pass

    seen: list[str] = []

    def stop_once(phase: str) -> None:
        seen.append(phase)
        if len(seen) == 2:
            raise Interrupted

    with pytest.raises(Interrupted):
        encrypt(plain_library, backup=None, checkpoint=stop_once)
    report = encrypt(plain_library, backup=None)

    # everything already done before the interruption is skipped, so far fewer than the total
    total = 2 + 3 + 3  # filenames + chunk texts + headings
    assert 0 < report.values_encrypted <= total  # type: ignore[attr-defined]


def test_an_already_encrypted_library_cannot_be_encrypted_again(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)

    with pytest.raises(KeystoreError, match="already encrypted"):
        encrypt(plain_library, backup=None)


def test_a_weak_passphrase_changes_nothing_at_all(plain_library: Path) -> None:
    before = (plain_library / DB_FILENAME).read_bytes()

    with pytest.raises(ValueError, match="passphrase"):
        encrypt_library(plain_library, "weak", backup_to=None)

    assert (plain_library / DB_FILENAME).read_bytes() == before
    assert not (plain_library / "security.json").exists()
    assert keystore.library_state(plain_library) == "plaintext"


def test_a_library_in_use_by_another_program_is_not_touched(plain_library: Path) -> None:
    other = sqlite3.connect(plain_library / DB_FILENAME)
    other.execute("BEGIN EXCLUSIVE")
    before = (plain_library / DB_FILENAME).read_bytes()
    try:
        with pytest.raises(SecurityError, match="in use by another program"):
            encrypt(plain_library, backup=None)
    finally:
        other.rollback()
        other.close()

    assert (plain_library / DB_FILENAME).read_bytes() == before
    assert not (plain_library / "security.json").exists()


def test_resuming_with_the_wrong_passphrase_is_refused_when_windows_cannot_help(
    plain_library: Path,
) -> None:
    class Interrupted(Exception):
        pass

    def stop(_phase: str) -> None:
        raise Interrupted

    with pytest.raises(Interrupted):
        encrypt(plain_library, backup=None, checkpoint=stop)
    drop_windows_copy(plain_library)

    with pytest.raises(WrongPassphraseError):
        encrypt_library(plain_library, "this is not the passphrase", backup_to=None)


# --- unusual libraries -------------------------------------------------------------------------


def test_a_library_whose_files_folder_is_missing_still_encrypts(
    plain_library: Path, tmp_path: Path
) -> None:
    import shutil

    shutil.rmtree(plain_library / "files")  # like a library whose stored copies were deleted

    report = encrypt(plain_library, backup=None)

    assert report.files_total == 0  # type: ignore[attr-defined]
    assert files_containing(plain_library, NEEDLES) == []
    engine = create_db_engine(plain_library)
    with Session(engine) as session:
        assert SECRET_NAME in {d.original_filename for d in session.scalars(select(Document))}
    engine.dispose()


def test_an_empty_new_library_gets_keys_and_is_encrypted_from_the_start(
    data_dir: Path,
) -> None:
    data_dir.mkdir()

    encrypt(data_dir, backup=None)

    assert keystore.library_state(data_dir) == "encrypted"


def test_a_damaged_stored_file_is_encrypted_anyway_with_a_warning(plain_library: Path) -> None:
    victim = next(p for p in (plain_library / "files").rglob("*") if p.is_file())
    victim.write_bytes(b"corrupted: DAMAGED-SECRET-77")

    report = encrypt(plain_library, backup=None)

    assert any("checksum" in w for w in report.warnings)  # type: ignore[attr-defined]
    assert files_containing(plain_library, [b"DAMAGED-SECRET-77"]) == []


def test_a_plaintext_file_that_happens_to_start_with_the_marker_bytes_is_handled(
    migrated_data_dir: Path,
) -> None:
    tricky = b"RLE1" + bytes(range(60))  # looks like an encrypted blob, but is plain content
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        save_file(session, migrated_data_dir, tricky, "tricky.bin")
    engine.dispose()

    encrypt(migrated_data_dir, backup=None)

    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        document = session.scalars(select(Document)).one()
        assert read_file(migrated_data_dir, document) == tricky
    engine.dispose()


def test_unicode_names_empty_headings_and_large_text_round_trip(migrated_data_dir: Path) -> None:
    big = "café 日本語 \U0001f642 " * 20_000
    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        document = save_file(session, migrated_data_dir, big.encode(), "résumé 日本.txt")
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                text=big,
                heading_path="",
                start_line=1,
                end_line=1,
                char_count=len(big),
            )
        )
        session.commit()
    engine.dispose()

    encrypt(migrated_data_dir, backup=None)

    engine = create_db_engine(migrated_data_dir)
    with Session(engine) as session:
        document = session.scalars(select(Document)).one()
        chunk = session.scalars(select(Chunk)).one()
        assert document.original_filename == "résumé 日本.txt"
        assert chunk.text == big
        assert chunk.heading_path == ""
    engine.dispose()


# --- locking, keys and refusing to guess ---------------------------------------------------------


def test_without_the_windows_copy_the_library_stays_locked_until_the_passphrase_is_given(
    plain_library: Path,
) -> None:
    encrypt(plain_library, backup=None)
    drop_windows_copy(plain_library)

    with pytest.raises(LibraryLockedError):
        create_db_engine(plain_library)
    with pytest.raises(WrongPassphraseError):
        create_db_engine(plain_library, passphrase="nope nope nope nope")

    engine = create_db_engine(plain_library, passphrase=PASSPHRASE)
    engine.dispose()
    engine = create_db_engine(plain_library)  # remembered for this Windows account
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 2
    engine.dispose()


def test_a_missing_key_file_is_an_error_never_a_silent_plaintext_library(
    plain_library: Path,
) -> None:
    encrypt(plain_library, backup=None)
    (plain_library / "security.json").unlink()

    with pytest.raises(KeystoreError, match="key file .* is missing"):
        create_db_engine(plain_library)


def test_encrypted_data_is_never_returned_as_text_without_a_key(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    keyless = create_engine(database_url(plain_library))  # an engine nobody attached a key to

    with Session(keyless) as session, pytest.raises(LibraryLockedError):
        session.scalars(select(Chunk)).all()
    keyless.dispose()


def test_a_modified_ciphertext_value_is_detected_on_read(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    connection = sqlite3.connect(plain_library / DB_FILENAME)
    value = connection.execute("SELECT text FROM chunks WHERE id = 1").fetchone()[0]
    flipped = value[:-3] + ("A" if value[-3] != "A" else "B") + value[-2:]
    connection.execute("UPDATE chunks SET text = ? WHERE id = 1", (flipped,))
    connection.commit()
    connection.close()

    engine = create_db_engine(plain_library)
    with Session(engine) as session, pytest.raises(DecryptionError):
        session.scalars(select(Chunk).where(Chunk.id == 1)).one()
    engine.dispose()


def test_a_value_moved_to_another_column_does_not_decrypt(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    connection = sqlite3.connect(plain_library / DB_FILENAME)
    heading = connection.execute("SELECT heading_path FROM chunks WHERE id = 1").fetchone()[0]
    connection.execute("UPDATE chunks SET text = ? WHERE id = 1", (heading,))
    connection.commit()
    connection.close()

    engine = create_db_engine(plain_library)
    with Session(engine) as session, pytest.raises(DecryptionError):
        session.scalars(select(Chunk).where(Chunk.id == 1)).one()
    engine.dispose()


def test_a_stored_file_swapped_for_another_one_does_not_decrypt(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    paths = sorted(p for p in (plain_library / "files").rglob("*") if p.is_file())
    first, second = paths[0], paths[1]
    first.write_bytes(second.read_bytes())

    engine = create_db_engine(plain_library)
    with Session(engine) as session:
        documents = session.scalars(select(Document)).all()
        victim = next(d for d in documents if d.stored_path.endswith(first.name))
        with pytest.raises(DecryptionError):
            read_file(plain_library, victim)
    engine.dispose()


def test_a_plaintext_library_and_an_encrypted_one_can_be_open_in_the_same_process(
    plain_library: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    keystore.create_keys(other, PASSPHRASE, state="encrypted")
    encrypted_engine = create_db_engine(other)
    plain_engine = create_engine(database_url(plain_library))

    assert vault_of(encrypted_engine) is not None
    assert vault_of(plain_engine) is None
    encrypted_engine.dispose()
    plain_engine.dispose()


def test_attaching_a_vault_to_an_engine_does_not_affect_other_engines(tmp_path: Path) -> None:
    first = create_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
    second = create_engine(f"sqlite:///{(tmp_path / 'b.db').as_posix()}")

    attach_vault(first, Vault(bytes(32)))

    assert vault_of(first) is not None
    assert vault_of(second) is None


def test_reading_from_many_threads_at_once_works(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    engine = create_db_engine(plain_library)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(20):
                with Session(engine) as session:
                    assert session.scalars(select(Chunk)).all()
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    engine.dispose()

    assert errors == []


# --- structure stays queryable ------------------------------------------------------------------


def test_counts_and_structure_queries_work_on_an_encrypted_library(plain_library: Path) -> None:
    encrypt(plain_library, backup=None)
    engine = create_db_engine(plain_library)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Chunk)) == 3
        assert session.scalar(select(func.max(Chunk.char_count))) > 0
    engine.dispose()


# --- backup and restore of an encrypted library ---------------------------------------------


def test_a_backup_of_an_encrypted_library_contains_no_readable_notes_and_the_key_file(
    plain_library: Path, tmp_path: Path
) -> None:
    from app.operations.backup import create_backup

    encrypt(plain_library, backup=None)
    archive = tmp_path / "enc.zip"

    summary = create_backup(plain_library, archive)

    assert summary.encrypted is True
    assert summary.warnings == []  # encrypted files do not "mismatch" their hash-names
    import zipfile

    with zipfile.ZipFile(archive) as z:
        assert "security.json" in z.namelist()
        # nothing readable inside: not even after decompressing every entry
        assert all(needle not in z.read(name) for name in z.namelist() for needle in NEEDLES)


def test_an_encrypted_library_restores_and_opens_with_the_recovery_passphrase(
    plain_library: Path, tmp_path: Path
) -> None:
    from app.operations.backup import create_backup

    encrypt(plain_library, backup=None)
    archive = tmp_path / "enc.zip"
    create_backup(plain_library, archive)
    restored = tmp_path / "restored"

    summary = restore_backup(archive, restored)

    assert summary.encrypted is True
    drop_windows_copy(restored)  # as on another computer
    with pytest.raises(LibraryLockedError):
        create_db_engine(restored)
    engine = create_db_engine(restored, passphrase=PASSPHRASE)
    with Session(engine) as session:
        assert SECRET_NAME in {d.original_filename for d in session.scalars(select(Document))}
        secret_doc = next(
            d for d in session.scalars(select(Document)) if d.original_filename == SECRET_NAME
        )
        assert SECRET_TEXT.encode() in read_file(restored, secret_doc)
    engine.dispose()
    assert files_containing(restored, NEEDLES) == []


def test_a_backup_of_a_plaintext_library_says_it_is_not_encrypted(
    plain_library: Path, tmp_path: Path
) -> None:
    from app.operations.backup import create_backup

    assert create_backup(plain_library, tmp_path / "plain.zip").encrypted is False


def test_a_damaged_key_file_inside_a_backup_is_refused(plain_library: Path, tmp_path: Path) -> None:
    import hashlib
    import json
    import zipfile

    from app.operations.backup import BackupError, create_backup

    encrypt(plain_library, backup=None)
    good = tmp_path / "enc.zip"
    create_backup(plain_library, good)
    junk = b"{ this is not a key file"
    with zipfile.ZipFile(good) as src:
        manifest = json.loads(src.read("manifest.json"))
        manifest["entries"]["security.json"] = {
            "sha256": hashlib.sha256(junk).hexdigest(),
            "size": len(junk),
        }
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as out:
            for info in src.infolist():
                if info.filename == "security.json":
                    out.writestr(info.filename, junk)
                elif info.filename == "manifest.json":
                    out.writestr(info.filename, json.dumps(manifest))
                else:
                    out.writestr(info.filename, src.read(info.filename))

    with pytest.raises(BackupError, match="key file inside the backup is damaged"):
        restore_backup(bad, tmp_path / "restored")


# --- failing safely ---------------------------------------------------------------------------


def test_a_database_without_tables_is_refused_with_a_hint_and_nothing_is_changed(
    data_dir: Path,
) -> None:
    data_dir.mkdir()
    sqlite3.connect(data_dir / DB_FILENAME).close()  # a database file with no tables yet

    with pytest.raises(SecurityError, match="alembic upgrade head"):
        encrypt(data_dir, backup=None)

    assert not (data_dir / "security.json").exists()


def test_a_failing_safety_backup_stops_before_anything_is_changed(
    plain_library: Path, tmp_path: Path
) -> None:
    (tmp_path / "taken.zip").write_bytes(b"already here")  # a backup never overwrites a file
    before = (plain_library / DB_FILENAME).read_bytes()

    with pytest.raises(SecurityError, match="safety backup could not be made"):
        encrypt(plain_library, backup=tmp_path / "taken.zip")

    assert (plain_library / DB_FILENAME).read_bytes() == before
    assert not (plain_library / "security.json").exists()
    assert files_containing(plain_library, NEEDLES), "the library must be untouched"
