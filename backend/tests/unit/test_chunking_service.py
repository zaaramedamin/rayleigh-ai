from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.knowledge.chunking.service import chunk_document, has_chunks, rechunk_all
from app.knowledge.ingestion.parsers import ParseError
from app.storage.files import save_file
from app.storage.models import Chunk, Document

MD = b"# Title\n\nintro\n\n## Part\n\nbody text\n"


def _chunks(session: Session, document: Document) -> list[Chunk]:
    return list(
        session.scalars(
            select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
        )
    )


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def _new_chunk(document_id: int, chunk_index: int = 0) -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text="x",
        heading_path="",
        start_line=1,
        end_line=1,
        char_count=1,
    )


def test_chunk_document_stores_chunks_with_provenance(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, MD, "notes.md")

    created = chunk_document(session, data_dir, document, 1000, 100)

    chunks = _chunks(session, document)
    assert created == len(chunks) == 2
    assert [c.heading_path for c in chunks] == ["Title", "Title > Part"]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 7
    assert chunks[1].char_count == len(chunks[1].text)
    assert chunks[1].citation_id == f"{document.id}:1"


def test_rechunking_replaces_chunks_without_duplicates(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    chunk_document(session, data_dir, document, 1000, 100)
    chunk_document(session, data_dir, document, 1000, 100)

    assert _count(session, Chunk) == 2


def test_non_markdown_file_ignores_heading_syntax(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, b"# not a heading\n\ntext\n", "notes.txt")

    chunk_document(session, data_dir, document, 1000, 100)

    assert [c.heading_path for c in _chunks(session, document)] == [""]


def test_html_documents_are_chunked_from_extracted_text(session: Session, data_dir: Path) -> None:
    html = b"<html><script>secret()</script><p>Hello</p><p>World</p></html>"
    document = save_file(session, data_dir, html, "page.html")

    chunk_document(session, data_dir, document, 1000, 100)

    text = " ".join(c.text for c in _chunks(session, document))
    assert "Hello" in text
    assert "World" in text
    assert "secret" not in text


def test_has_chunks(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    assert not has_chunks(session, document)

    chunk_document(session, data_dir, document, 1000, 100)

    assert has_chunks(session, document)


def test_rechunk_all_uses_new_settings(session: Session, data_dir: Path) -> None:
    paragraphs = "\n\n".join(f"paragraph {i} " + "word " * 20 for i in range(10))
    document = save_file(session, data_dir, f"# T\n\n{paragraphs}".encode(), "a.md")
    chunk_document(session, data_dir, document, 5000, 100)
    assert _count(session, Chunk) == 1

    total = rechunk_all(session, data_dir, 300, 50)

    assert total == _count(session, Chunk)
    assert total > 1


def test_deleting_a_document_removes_its_chunks(session: Session, data_dir: Path) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    chunk_document(session, data_dir, document, 1000, 100)

    session.delete(document)
    session.commit()

    assert _count(session, Chunk) == 0


def test_database_cascade_removes_chunks_even_without_the_orm(
    session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    chunk_document(session, data_dir, document, 1000, 100)

    session.execute(delete(Document).where(Document.id == document.id))
    session.commit()

    assert _count(session, Chunk) == 0


def test_duplicate_chunk_index_is_rejected_by_the_database(
    session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    session.add_all([_new_chunk(document.id), _new_chunk(document.id)])

    with pytest.raises(IntegrityError):
        session.commit()


def test_chunk_for_missing_document_is_rejected_by_foreign_key(session: Session) -> None:
    session.add(_new_chunk(999))

    with pytest.raises(IntegrityError):
        session.commit()


def test_chunking_a_document_whose_bytes_are_no_longer_text_raises(
    session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, MD, "notes.md")
    (data_dir / document.stored_path).write_bytes(b"\x80\x81 not utf8")

    with pytest.raises(ParseError):
        chunk_document(session, data_dir, document, 1000, 100)


def test_chunks_are_ordered_by_index_and_cover_all_text(session: Session, data_dir: Path) -> None:
    paragraphs = "\n\n".join(f"unique-marker-{i} " + "word " * 20 for i in range(10))
    document = save_file(session, data_dir, paragraphs.encode(), "a.txt")

    chunk_document(session, data_dir, document, 300, 50)

    joined = "\n".join(c.text for c in _chunks(session, document))
    assert all(f"unique-marker-{i}" in joined for i in range(10))
    assert _count(session, Document) == 1
    assert session.scalar(select(func.min(Chunk.chunk_index))) == 0
