import sqlite3
from collections.abc import Callable
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


def test_a_healthy_setup_has_no_failures_and_no_warnings(
    healthy: Path, make_settings: MakeSettings, notes: Path
) -> None:
    checks = by_name(run_doctor(make_settings(allowed_folders=[notes])))

    assert {name: c.status for name, c in checks.items()} == {
        "database": "ok",
        "stored files": "ok",
        "chunks": "ok",
        "search index": "ok",
        "allowed folders": "ok",
        "disk space": "ok",
        "embedding model": "ok",
        "local LLM": "ok",
    }
    assert "checksums verified" in checks["stored files"].message


def test_a_missing_database_is_a_failure_and_nothing_is_created(
    data_dir: Path, make_settings: MakeSettings
) -> None:
    checks = by_name(run_doctor(make_settings()))

    assert checks["database"].status == "fail"
    assert checks["database"].fix == "alembic upgrade head"
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
