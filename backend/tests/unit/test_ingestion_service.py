from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.knowledge.ingestion import service
from app.knowledge.ingestion.scanner import ScanResult
from app.knowledge.ingestion.service import ingest_folders
from app.storage.models import Chunk, Document

MAX_BYTES = 1024


@pytest.fixture
def allowed(tmp_path: Path) -> Path:
    root = tmp_path / "allowed"
    root.mkdir()
    return root


def _doc_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Document)) or 0


def test_ingests_supported_files_and_creates_documents(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "a.txt").write_text("alpha")
    (allowed / "b.md").write_text("# Beta")
    (allowed / "c.pdf").write_bytes(b"%PDF")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 2
    assert summary.skipped_unsupported == 1
    assert _doc_count(session) == 2


def test_reingesting_creates_no_duplicates(session: Session, data_dir: Path, allowed: Path) -> None:
    (allowed / "a.txt").write_text("alpha")
    (allowed / "b.md").write_text("# Beta")
    ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 0
    assert summary.unchanged == 2
    assert _doc_count(session) == 2


def test_changed_file_is_added_as_new_content(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    note = allowed / "a.txt"
    note.write_text("v1")
    ingest_folders(session, data_dir, [allowed], MAX_BYTES)
    note.write_text("v2")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 1


def test_empty_allow_list_ingests_nothing(session: Session, data_dir: Path) -> None:
    summary = ingest_folders(session, data_dir, [], MAX_BYTES)

    assert summary.added == 0
    assert _doc_count(session) == 0
    assert not (data_dir / "files").exists()


def test_empty_files_are_skipped(session: Session, data_dir: Path, allowed: Path) -> None:
    (allowed / "empty.txt").write_bytes(b"")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.skipped_empty == 1
    assert _doc_count(session) == 0


def test_files_over_size_limit_are_skipped(session: Session, data_dir: Path, allowed: Path) -> None:
    (allowed / "big.txt").write_bytes(b"x" * (MAX_BYTES + 1))
    (allowed / "exact.txt").write_bytes(b"y" * MAX_BYTES)

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.skipped_too_large == 1
    assert summary.added == 1


def test_undecodable_and_binary_files_fail_without_crashing(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "latin1.txt").write_bytes(b"\x80abc")
    (allowed / "binary.txt").write_bytes(b"abc\x00def")
    (allowed / "good.txt").write_text("fine")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 1
    assert summary.failed == {"not_utf8": 1, "binary": 1}


def test_bom_file_is_ingested_with_original_bytes_preserved(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    raw = b"\xef\xbb\xbfhello"
    (allowed / "bom.txt").write_bytes(raw)

    ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    document = session.scalars(select(Document)).one()
    assert (data_dir / document.stored_path).read_bytes() == raw


def test_service_refuses_file_outside_allow_list_even_if_scanner_returns_it(
    session: Session,
    data_dir: Path,
    allowed: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    monkeypatch.setattr(
        service, "scan_allowed_folders", lambda _folders: ScanResult(files=[outside.resolve()])
    )

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 0
    assert summary.skipped_outside_allowlist == 1
    assert _doc_count(session) == 0


def test_missing_folder_is_reported(session: Session, data_dir: Path, tmp_path: Path) -> None:
    summary = ingest_folders(session, data_dir, [tmp_path / "nope"], MAX_BYTES)

    assert summary.folders_missing == 1


def test_ingests_text_like_formats(session: Session, data_dir: Path, allowed: Path) -> None:
    (allowed / "food.csv").write_text("name,kcal\noats,389\n")
    (allowed / "meta.json").write_text('{"a": 1}')
    (allowed / "page.html").write_text("<p>hello</p>")
    (allowed / "conf.yaml").write_text("key: value\n")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.added == 4
    assert not summary.failed


def test_invalid_json_fails_with_reason(session: Session, data_dir: Path, allowed: Path) -> None:
    (allowed / "broken.json").write_text('{"a": ')

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.failed == {"invalid_json": 1}
    assert _doc_count(session) == 0


def test_files_with_no_extractable_text_are_skipped_as_empty(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "blank.txt").write_text("   \n\n  ")
    (allowed / "scripts_only.html").write_text("<script>var x = 1</script>")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.skipped_empty == 2
    assert _doc_count(session) == 0


def test_summary_reports_unsupported_types_by_extension(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "a.pdf").write_bytes(b"%PDF")
    (allowed / "b.docx").write_bytes(b"PK")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.unsupported_by_extension == {".pdf": 1, ".docx": 1}


def test_ingestion_creates_chunks_for_new_documents(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "a.md").write_text("# T\n\nbody\n")
    (allowed / "b.txt").write_text("plain note")

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.chunks_created == 2
    assert session.scalar(select(func.count()).select_from(Chunk)) == 2


def test_reingesting_does_not_rechunk_or_duplicate_chunks(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "a.md").write_text("# T\n\nbody\n")
    ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.unchanged == 1
    assert summary.chunks_created == 0
    assert session.scalar(select(func.count()).select_from(Chunk)) == 1


def test_unchanged_document_without_chunks_gets_chunked(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    (allowed / "a.md").write_text("# T\n\nbody\n")
    ingest_folders(session, data_dir, [allowed], MAX_BYTES)
    session.execute(delete(Chunk))  # as if ingested before chunking existed
    session.commit()

    summary = ingest_folders(session, data_dir, [allowed], MAX_BYTES)

    assert summary.unchanged == 1
    assert summary.chunks_created == 1


def test_chunk_settings_are_applied_during_ingestion(
    session: Session, data_dir: Path, allowed: Path
) -> None:
    paragraphs = "\n\n".join(f"paragraph {i} " + "word " * 10 for i in range(20))
    (allowed / "a.txt").write_text(paragraphs)

    summary = ingest_folders(
        session, data_dir, [allowed], 10 * MAX_BYTES, chunk_size=200, chunk_overlap=20
    )

    assert summary.chunks_created > 3
    assert all(c.char_count <= 200 for c in session.scalars(select(Chunk)))
