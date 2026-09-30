import hashlib
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.storage.files import (
    UnsafePathError,
    display_name,
    read_file,
    resolve_inside,
    save_file,
)
from app.storage.models import Document


def _all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def test_save_file_stores_bytes_at_hashed_path_and_records_document(
    session: Session, data_dir: Path
) -> None:
    data = b"hello notes"
    digest = hashlib.sha256(data).hexdigest()

    document = save_file(session, data_dir, data, "notes.txt")

    assert document.content_hash == digest
    assert document.stored_path == f"files/{digest[:2]}/{digest}"
    assert document.original_filename == "notes.txt"
    assert document.size_bytes == len(data)
    assert document.media_type == "text/plain"
    assert (data_dir / document.stored_path).read_bytes() == data
    assert read_file(data_dir, document) == data


def test_saving_same_content_twice_creates_one_row_and_one_file(
    session: Session, data_dir: Path
) -> None:
    first = save_file(session, data_dir, b"same", "a.txt")
    second = save_file(session, data_dir, b"same", "b.txt")

    assert first.id == second.id
    assert session.scalar(select(func.count()).select_from(Document)) == 1
    assert len(_all_files(data_dir / "files")) == 1


@pytest.mark.parametrize(
    "filename",
    [
        "../../evil.txt",
        "..\\..\\evil.txt",
        "C:\\Windows\\system32\\x.txt",
        "/etc/passwd",
        "a/b/../../c",
        "..",
    ],
)
def test_hostile_filenames_never_affect_where_the_file_is_written(
    session: Session, data_dir: Path, filename: str
) -> None:
    document = save_file(session, data_dir, filename.encode(), filename)

    files_root = (data_dir / "files").resolve()
    written = _all_files(data_dir)
    # Exactly the one stored file plus the sqlite db; nothing written elsewhere.
    stored = [p for p in written if p.suffix != ".db"]
    assert len(stored) == 1
    assert stored[0].resolve().is_relative_to(files_root)
    assert stored[0].name == document.content_hash
    assert "/" not in document.original_filename
    assert "\\" not in document.original_filename
    assert not (data_dir.parent / "evil.txt").exists()


def test_resolve_inside_rejects_escape(data_dir: Path) -> None:
    data_dir.mkdir()

    with pytest.raises(UnsafePathError):
        resolve_inside(data_dir, "../outside.txt")
    with pytest.raises(UnsafePathError):
        resolve_inside(data_dir, "files/../../outside.txt")


def test_read_file_refuses_tampered_record(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, b"data", "x.txt")
    document.stored_path = "../../secret.txt"

    with pytest.raises(UnsafePathError):
        read_file(data_dir, document)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("notes.txt", "notes.txt"),
        ("../../evil.txt", "evil.txt"),
        ("C:\\dir\\file.md", "file.md"),
        ("", "unnamed"),
        ("..", "unnamed"),
        ("bad\x00name.txt", "badname.txt"),
    ],
)
def test_display_name(raw: str, expected: str) -> None:
    assert display_name(raw) == expected
