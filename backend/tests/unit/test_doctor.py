import sqlite3
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import app.operations.doctor as doctor
from app.ai.llm.base import LLMUnavailableError
from app.core.config import Settings
from app.knowledge.chunking.service import chunk_document
from app.knowledge.components import vector_store_path
from app.knowledge.indexing.service import index_pending
from app.operations.doctor import Check, format_checks, run_doctor
from app.operations.windows import BlockedFile
from app.storage.database import DB_FILENAME, create_db_engine
from app.storage.files import save_file
from app.storage.vector_store import QdrantVectorStore, collection_name
from tests.fakes import FakeLLM, HashingEmbedder

MakeSettings = Callable[..., Settings]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(autouse=True)
def _no_real_ollama_or_registry(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM(model_name="qwen3.5:4b", installed=["qwen3.5:4b"])
    monkeypatch.setattr(doctor, "create_llm", lambda *_a, **_k: llm)
    monkeypatch.setattr(doctor, "smart_app_control_state", lambda: None)
    return llm


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    return folder


@pytest.fixture
def downloaded_model(models_dir: Path) -> None:
    folder = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")


@pytest.fixture
def healthy(migrated_data_dir: Path, downloaded_model: None) -> Path:
    """A migrated library, with two documents that are chunked, indexed and searchable."""
    embedder = HashingEmbedder(model_name=MODEL)
    engine = create_db_engine(migrated_data_dir)
    store = QdrantVectorStore.open_local(
        vector_store_path(Settings(_env_file=None, data_dir=migrated_data_dir)),
        collection_name(MODEL),
        embedder.dimension,
    )
    try:
        with Session(engine) as session:
            for name, text in {"a.md": "# A\n\nalpha text", "b.txt": "beta text"}.items():
                document = save_file(session, migrated_data_dir, text.encode(), name)
                chunk_document(session, migrated_data_dir, document, 1000, 100)
            index_pending(session, embedder, store)
    finally:
        store.close()
        engine.dispose()
    return migrated_data_dir


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_a_healthy_plaintext_library_only_warns_that_it_is_not_encrypted(
    healthy: Path, make_settings: MakeSettings, notes: Path
) -> None:
    checks = by_name(run_doctor(make_settings(allowed_folders=[notes])))

    expected = {
        "database": "ok",
        "encryption": "warn",
        "stored files": "ok",
        "library": "ok",
        "chunks": "ok",
        "search index": "ok",
        "allowed folders": "ok",
        "disk space": "ok",
        "embedding model": "ok",
        "embedding library": "ok",
        "local LLM": "ok",
        "scikit-learn": "ok",
    }
    if sys.platform == "win32":  # questions that only Windows can answer
        expected |= {"BitLocker": "ok", "Blocked files": "ok"}
    assert {name: c.status for name, c in checks.items()} == expected
    assert "checksums verified" in checks["stored files"].message


def test_a_missing_database_is_a_failure_and_nothing_is_created(
    data_dir: Path, make_settings: MakeSettings
) -> None:
    checks = by_name(run_doctor(make_settings()))

    assert checks["database"].status == "fail"
    assert checks["database"].fix == "python -m app migrate"
    assert "stored files" not in checks  # dependent checks are skipped, not guessed
    assert not data_dir.exists()


def test_an_unmigrated_database_is_a_failure(data_dir: Path, make_settings: MakeSettings) -> None:
    engine = create_db_engine(data_dir)  # creates the file but never ran the migrations
    engine.connect().close()
    engine.dispose()
    sqlite3.connect(data_dir / DB_FILENAME).close()

    checks = by_name(run_doctor(make_settings()))

    assert checks["database"].status == "fail"
    assert "out of date" in checks["database"].message


def test_a_corrupted_stored_file_is_detected(healthy: Path, make_settings: MakeSettings) -> None:
    victim = next(p for p in (healthy / "files").rglob("*") if p.is_file())
    victim.write_bytes(b"damaged")

    checks = by_name(run_doctor(make_settings()))

    assert checks["stored files"].status == "fail"
    assert "1 stored file(s) do not match their checksum" in checks["stored files"].message
    assert "python -m app ingest" in (checks["stored files"].fix or "")


def test_quick_mode_skips_checksums_but_still_finds_missing_files(
    healthy: Path, make_settings: MakeSettings
) -> None:
    files = [p for p in (healthy / "files").rglob("*") if p.is_file()]
    files[0].write_bytes(b"damaged")  # not noticed in quick mode
    quick = by_name(run_doctor(make_settings(), quick=True))
    assert quick["stored files"].status == "ok"

    files[1].unlink()
    quick = by_name(run_doctor(make_settings(), quick=True))
    assert quick["stored files"].status == "fail"
    assert "1 stored file(s) are missing" in quick["stored files"].message


def test_orphaned_files_are_reported_but_never_deleted(
    healthy: Path, make_settings: MakeSettings
) -> None:
    orphan = healthy / "files" / "ab" / ("c" * 64)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"belongs to nothing")

    checks = by_name(run_doctor(make_settings(), repair=True))

    assert checks["orphaned files"].status == "warn"
    assert orphan.exists()


def test_a_path_pointing_outside_the_data_folder_is_flagged(
    healthy: Path, make_settings: MakeSettings
) -> None:
    connection = sqlite3.connect(healthy / DB_FILENAME)
    connection.execute("UPDATE documents SET stored_path = '../../outside' WHERE id = 1")
    connection.commit()
    connection.close()

    checks = by_name(run_doctor(make_settings()))

    assert checks["stored files"].status == "fail"
    assert "outside the data folder" in checks["stored files"].message


def test_a_vector_index_that_disagrees_with_the_database_is_found_and_marked_for_reindexing(
    healthy: Path, make_settings: MakeSettings
) -> None:
    store = QdrantVectorStore.open_local(
        vector_store_path(make_settings()), collection_name(MODEL), 256
    )
    store.reset()  # the vectors vanish, the database still says "indexed"
    store.close()

    found = by_name(run_doctor(make_settings()))
    assert found["search index"].status == "warn"
    assert "disagree" in found["search index"].message
    connection = sqlite3.connect(healthy / DB_FILENAME)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE indexed_model IS NOT NULL"
        ).fetchone()[0]
        == 2
    )
    connection.close()

    fixed = by_name(run_doctor(make_settings(), repair=True))
    assert "Marked every document for re-indexing" in fixed["search index"].message
    connection = sqlite3.connect(healthy / DB_FILENAME)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE indexed_model IS NOT NULL"
        ).fetchone()[0]
        == 0
    )
    connection.close()

    after = by_name(run_doctor(make_settings()))
    assert after["search index"].message.startswith("2 of 2 document(s) are not searchable yet")


def test_repair_never_runs_without_being_asked(healthy: Path, make_settings: MakeSettings) -> None:
    store = QdrantVectorStore.open_local(
        vector_store_path(make_settings()), collection_name(MODEL), 256
    )
    store.reset()
    store.close()

    run_doctor(make_settings())

    connection = sqlite3.connect(healthy / DB_FILENAME)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE indexed_model IS NOT NULL"
        ).fetchone()[0]
        == 2
    )
    connection.close()


def test_a_busy_vector_store_is_a_warning_not_a_crash(
    healthy: Path, make_settings: MakeSettings
) -> None:
    holder = QdrantVectorStore.open_local(
        vector_store_path(make_settings()), collection_name(MODEL), 256
    )
    try:
        checks = by_name(run_doctor(make_settings()))
    finally:
        holder.close()

    assert checks["search index"].status == "warn"
    assert "could not be checked" in checks["search index"].message


def test_documents_without_chunks_are_reported(healthy: Path, make_settings: MakeSettings) -> None:
    engine = create_db_engine(healthy)
    with Session(engine) as session:
        save_file(session, healthy, b"never chunked", "raw.txt")
    engine.dispose()

    checks = by_name(run_doctor(make_settings()))

    assert checks["chunks"].status == "warn"
    assert "1 document(s) have no chunks" in checks["chunks"].message


def test_an_allow_list_that_overlaps_the_data_folder_is_a_failure(
    healthy: Path, make_settings: MakeSettings
) -> None:
    checks = by_name(run_doctor(make_settings(allowed_folders=[healthy])))

    assert checks["data folder"].status == "fail"


def test_missing_allowed_folders_and_an_empty_allow_list_are_warnings(
    healthy: Path, make_settings: MakeSettings, tmp_path: Path
) -> None:
    missing = by_name(run_doctor(make_settings(allowed_folders=[tmp_path / "nope"])))
    empty = by_name(run_doctor(make_settings(allowed_folders=[])))

    assert missing["allowed folders"].status == "warn"
    assert "do not exist" in missing["allowed folders"].message
    assert empty["allowed folders"].status == "warn"


def test_an_undownloaded_model_is_a_warning_with_the_fix(
    migrated_data_dir: Path, make_settings: MakeSettings
) -> None:
    checks = by_name(run_doctor(make_settings()))

    assert checks["embedding model"].status == "warn"
    assert checks["embedding model"].fix == "python -m app download-model"


def test_ollama_down_or_missing_the_model_is_a_warning(
    healthy: Path, make_settings: MakeSettings, _no_real_ollama_or_registry: FakeLLM
) -> None:
    _no_real_ollama_or_registry.error = LLMUnavailableError("down")
    down = by_name(run_doctor(make_settings()))["local LLM"]
    _no_real_ollama_or_registry.error = None
    _no_real_ollama_or_registry.installed = ["something-else:1b"]
    missing = by_name(run_doctor(make_settings()))["local LLM"]

    assert (down.status, "ollama serve" in (down.fix or "")) == ("warn", True)
    assert (missing.status, missing.fix) == ("warn", "ollama pull qwen3.5:4b")


def test_low_disk_space_is_a_warning(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Usage:
        free = 100 * 1024 * 1024  # 100 MB

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _p: Usage())

    checks = by_name(run_doctor(make_settings()))

    assert checks["disk space"].status == "warn"


@pytest.mark.parametrize("state", ["on", "evaluation", "off"])
def test_smart_app_control_is_reported_but_never_a_failure(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    monkeypatch.setattr(doctor, "smart_app_control_state", lambda: state)

    checks = by_name(run_doctor(make_settings()))

    assert checks["Windows Smart App Control"].status == "ok"
    assert state in checks["Windows Smart App Control"].message


def test_a_check_that_crashes_does_not_stop_the_doctor(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_settings: Settings) -> Check:
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor, "_check_model", explode)

    checks = by_name(run_doctor(make_settings()))

    assert checks["embedding model"].status == "fail"
    assert "RuntimeError" in checks["embedding model"].message
    assert checks["local LLM"].status == "ok"  # the rest still ran


def test_the_report_lists_fixes_only_for_problems_and_a_summary() -> None:
    text = format_checks(
        [
            Check("alpha", "ok", "fine", fix="ignored when ok"),
            Check("beta", "warn", "careful", fix="do this"),
            Check("gamma", "fail", "broken"),
        ]
    )

    assert "fix: do this" in text
    assert "ignored when ok" not in text
    assert text.endswith("1 ok, 1 warning(s), 1 failure(s)")


# --- encrypted libraries -----------------------------------------------------------------------


PASSPHRASE = "correct horse battery staple"


@pytest.fixture
def encrypted(healthy: Path) -> Path:
    from app.security.migrate import encrypt_library

    encrypt_library(healthy, PASSPHRASE, backup_to=None)
    return healthy


def test_a_healthy_encrypted_library_is_all_ok(
    encrypted: Path, make_settings: MakeSettings, notes: Path
) -> None:
    checks = by_name(run_doctor(make_settings(allowed_folders=[notes])))

    assert checks["encryption"].status == "ok"
    assert "AES-256-GCM" in checks["encryption"].message
    assert "Windows unlocks it automatically" in checks["encryption"].message
    assert checks["stored files"].status == "ok"
    assert "checksums verified" in checks["stored files"].message


def test_the_plaintext_library_warning_names_the_fix(
    healthy: Path, make_settings: MakeSettings
) -> None:
    check = by_name(run_doctor(make_settings()))["encryption"]

    assert check.status == "warn"
    assert "NOT encrypted" in check.message
    assert check.fix == "python -m app encrypt-library"


def test_a_locked_library_is_reported_with_a_plain_fix_and_nothing_else_is_guessed(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    from app.security import keystore

    keyfile = keystore.read_keyfile(encrypted)
    assert keyfile is not None
    keystore._write_keyfile(encrypted, keystore._replace(keyfile, wrapped_by_windows=None))

    checks = by_name(run_doctor(make_settings()))

    assert checks["encryption"].status == "fail"
    assert "locked" in checks["encryption"].message
    assert "recovery passphrase" in (checks["encryption"].fix or "")
    assert "stored files" not in checks  # nothing is read while locked


def test_the_doctor_can_unlock_with_the_passphrase_and_then_check_everything(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    from app.security import keystore

    keyfile = keystore.read_keyfile(encrypted)
    assert keyfile is not None
    keystore._write_keyfile(encrypted, keystore._replace(keyfile, wrapped_by_windows=None))

    checks = by_name(run_doctor(make_settings(), passphrase=PASSPHRASE))

    assert checks["stored files"].status == "ok"


def test_a_wrong_passphrase_is_a_failure_not_a_crash(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    from app.security import keystore

    keyfile = keystore.read_keyfile(encrypted)
    assert keyfile is not None
    keystore._write_keyfile(encrypted, keystore._replace(keyfile, wrapped_by_windows=None))

    checks = by_name(run_doctor(make_settings(), passphrase="definitely not it at all"))

    assert checks["encryption"].status == "fail"
    assert "not correct" in checks["encryption"].message


def test_a_half_finished_encryption_is_a_failure_that_says_how_to_resume(
    healthy: Path, make_settings: MakeSettings
) -> None:
    from app.security import keystore

    keystore.create_keys(healthy, PASSPHRASE, state="migrating")

    check = by_name(run_doctor(make_settings()))["encryption"]

    assert check.status == "fail"
    assert check.fix == "python -m app encrypt-library"


def test_plaintext_values_left_in_an_encrypted_library_are_found(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    connection = sqlite3.connect(encrypted / DB_FILENAME)
    connection.execute("UPDATE chunks SET text = 'left behind in plaintext' WHERE id = 1")
    connection.commit()
    connection.close()

    check = by_name(run_doctor(make_settings()))["encryption"]

    assert check.status == "fail"
    assert "1 value(s) in the database are not encrypted" in check.message


def test_a_plaintext_stored_file_in_an_encrypted_library_is_found(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    import hashlib

    victim = next(p for p in (encrypted / "files").rglob("*") if p.is_file())
    # a plaintext file stored under its own hash name, as an unencrypted leftover would be
    content = b"plain leftover"
    stored = victim.with_name(hashlib.sha256(content).hexdigest())
    stored.write_bytes(content)
    connection = sqlite3.connect(encrypted / DB_FILENAME)
    connection.execute(
        "UPDATE documents SET content_hash = ?, stored_path = ? WHERE id = 1",
        (stored.name, f"files/{stored.parent.name}/{stored.name}"),
    )
    connection.commit()
    connection.close()

    checks = by_name(run_doctor(make_settings()))

    assert checks["stored files"].status == "fail"
    assert "not encrypted in an encrypted library" in checks["stored files"].message


def test_a_tampered_encrypted_stored_file_is_found(
    encrypted: Path, make_settings: MakeSettings
) -> None:
    victim = next(p for p in (encrypted / "files").rglob("*") if p.is_file())
    data = bytearray(victim.read_bytes())
    data[-5] ^= 1
    victim.write_bytes(bytes(data))

    checks = by_name(run_doctor(make_settings()))

    assert checks["stored files"].status == "fail"
    assert "do not match their checksum" in checks["stored files"].message


# --- edited and deleted files -------------------------------------------------------------------


def _set_status(library: Path, name: str, status: str) -> None:
    from sqlalchemy import update

    from app.storage.models import Document

    engine = create_db_engine(library)
    with Session(engine) as session:
        session.execute(
            update(Document).where(Document.original_filename == name).values(status=status)
        )
        session.commit()
    engine.dispose()


def test_a_document_whose_file_is_gone_is_a_warning_with_a_way_to_see_it(
    healthy: Path, make_settings: MakeSettings, notes: Path
) -> None:
    _set_status(healthy, "b.txt", "missing")

    checks = by_name(run_doctor(make_settings(allowed_folders=[notes]), quick=True))

    assert checks["library"].status == "warn"
    assert "1 document(s) were not found in your folders" in checks["library"].message
    assert "prune --dry-run" in (checks["library"].fix or "")
    # It still has vectors until the next index run removes them; the doctor says so.
    assert checks["search index"].status == "warn"
    assert "1 replaced or missing document(s) still have vectors" in checks["search index"].message


def test_older_versions_are_reported_as_history_not_as_a_problem(
    healthy: Path, make_settings: MakeSettings, notes: Path
) -> None:
    _set_status(healthy, "b.txt", "superseded")

    checks = by_name(run_doctor(make_settings(allowed_folders=[notes]), quick=True))

    assert checks["library"].status == "ok"
    assert "1 current document(s), 1 older version(s) kept as history" in checks["library"].message


# --- what Windows says: BitLocker, blocked files, scikit-learn ---------------------------------

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="these questions are about Windows")


@on_windows
def test_bitlocker_on_is_reported_as_fine(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "bitlocker_protection", lambda _p: 1)

    check = by_name(run_doctor(make_settings()))["BitLocker"]

    assert check.status == "ok" and "is on" in check.message


@on_windows
def test_a_drive_that_is_not_protected_is_a_warning_with_both_ways_to_fix_it(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "bitlocker_protection", lambda _p: 0)

    check = by_name(run_doctor(make_settings()))["BitLocker"]

    assert check.status == "warn" and "it said 0" in check.message
    assert check.fix is not None
    assert "BitLocker" in check.fix and "encrypt-library" in check.fix and "manage-bde" in check.fix


@on_windows
def test_an_unprotected_drive_is_fine_when_the_library_is_encrypted_by_reyleight_itself(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "bitlocker_protection", lambda _p: 2)
    monkeypatch.setattr(doctor.keystore, "library_state", lambda _d: "encrypted")

    check = by_name(run_doctor(make_settings()))["BitLocker"]

    assert check.status == "ok" and "encrypted by Reyleight itself" in check.message


@on_windows
def test_a_drive_windows_cannot_be_asked_about_is_not_a_problem_and_says_how_to_check(
    healthy: Path, make_settings: MakeSettings
) -> None:
    check = by_name(run_doctor(make_settings()))["BitLocker"]  # the default test answer is None

    assert (
        check.status == "ok" and "could not ask" in check.message and "manage-bde" in check.message
    )


@on_windows
def test_files_windows_blocked_are_listed_with_what_to_do(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2026, 10, 5, 22, 32, tzinfo=UTC)
    blocked = [BlockedFile("scipy\\fft\\x.pyd", 3, when), BlockedFile("sklearn\\y.pyd", 1, when)]
    monkeypatch.setattr(doctor, "recent_blocked_files", lambda *_a, **_k: blocked)

    check = by_name(run_doctor(make_settings()))["Blocked files"]

    assert check.status == "warn"
    assert "2 file(s)" in check.message and "scipy\\fft\\x.pyd (3 times)" in check.message
    assert check.fix is not None and "requirements.lock" in check.fix
    assert "never changes Windows security settings" in check.fix


@on_windows
def test_no_blocked_files_and_an_unreadable_log_are_both_fine(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    none_blocked = by_name(run_doctor(make_settings()))["Blocked files"]
    monkeypatch.setattr(doctor, "recent_blocked_files", lambda *_a, **_k: None)
    unreadable = by_name(run_doctor(make_settings()))["Blocked files"]

    assert none_blocked.status == "ok" and "has not blocked any file" in none_blocked.message
    assert unreadable.status == "ok" and "could not read" in unreadable.message


def test_the_scikit_learn_placeholder_is_explained_when_it_is_needed(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = by_name(run_doctor(make_settings()))["scikit-learn"]
    monkeypatch.setattr(doctor, "scikit_learn_loads", lambda: False)
    blocked = by_name(run_doctor(make_settings()))["scikit-learn"]
    monkeypatch.setattr(doctor, "scikit_learn_loads", lambda: None)
    unknown = by_name(run_doctor(make_settings()))

    assert loads.status == "ok" and "loads normally" in loads.message
    assert blocked.status == "ok" and "placeholder" in blocked.message
    assert "does not affect embeddings" in blocked.message
    assert "scikit-learn" not in unknown  # nothing to say when it could not be asked


def test_a_failing_windows_question_cannot_stop_the_doctor(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_a: object, **_k: object) -> None:
        raise OSError("event log service stopped")

    monkeypatch.setattr(doctor, "recent_blocked_files", explode)

    checks = by_name(run_doctor(make_settings()))

    if sys.platform == "win32":
        blocked = checks["Blocked files"]
        assert blocked.status == "fail" and "OSError" in blocked.message
    assert checks["local LLM"].status == "ok"


# --- the library that turns text into vectors --------------------------------------------------

BLOCKED = (
    "ImportError: DLL load failed while importing _C: An Application Control policy has "
    "blocked this file."
)


def test_an_embedding_library_that_loads_is_ok(healthy: Path, make_settings: MakeSettings) -> None:
    check = by_name(run_doctor(make_settings()))["embedding library"]

    assert check.status == "ok" and "load" in check.message


def test_a_library_blocked_by_windows_is_a_failure_that_says_what_still_works(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "embedding_library_loads", lambda: (False, BLOCKED))

    checks = by_name(run_doctor(make_settings()))
    check = checks["embedding library"]

    assert check.status == "fail" and BLOCKED in check.message
    assert "cannot be searched, indexed or answered from" in check.message
    assert check.fix is not None
    assert (
        "Smart App Control" in check.fix and "never changes Windows security settings" in check.fix
    )
    assert "MY NOTES off" in check.fix  # chat with the model alone does not need the library
    assert checks["local LLM"].status == "ok"  # the rest of the report is unaffected


def test_a_library_that_fails_for_another_reason_points_at_the_dependencies(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        doctor, "embedding_library_loads", lambda: (False, "ModuleNotFoundError: torch")
    )

    check = by_name(run_doctor(make_settings()))["embedding library"]

    assert check.status == "fail" and "ModuleNotFoundError: torch" in check.message
    assert check.fix is not None and "requirements.lock" in check.fix
    assert "Smart App Control" not in check.fix


def test_a_library_that_could_not_be_asked_about_is_left_out_of_the_report(
    healthy: Path, make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "embedding_library_loads", lambda: (None, ""))

    assert "embedding library" not in by_name(run_doctor(make_settings()))
