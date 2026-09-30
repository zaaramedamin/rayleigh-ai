import logging
from pathlib import Path

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from app.knowledge.chunking.chunker import chunk_text
from app.knowledge.ingestion.file_types import extract_text
from app.storage.files import read_file
from app.storage.models import Chunk, Document

logger = logging.getLogger(__name__)

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


def has_chunks(session: Session, document: Document) -> bool:
    return bool(session.scalar(select(exists().where(Chunk.document_id == document.id))))


def chunk_document(
    session: Session, data_dir: Path, document: Document, chunk_size: int, overlap: int
) -> int:
    """(Re)build the chunks of one document from its stored bytes. Returns the chunk count.

    Existing chunks for the document are replaced, so running this twice never duplicates.
    Raises ParseError if the stored bytes can no longer be read as text.
    """
    suffix = Path(document.original_filename).suffix
    text = extract_text(suffix, read_file(data_dir, document))
    pieces = chunk_text(
        text,
        markdown=suffix.lower() in MARKDOWN_SUFFIXES,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    session.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=piece.index,
            text=piece.text,
            heading_path=piece.heading_path,
            start_line=piece.start_line,
            end_line=piece.end_line,
            char_count=len(piece.text),
        )
        for piece in pieces
    )
    document.indexed_model = None  # its vectors no longer match the new chunks
    session.commit()
    return len(pieces)


def rechunk_all(session: Session, data_dir: Path, chunk_size: int, overlap: int) -> int:
    """Rebuild chunks for every stored document, e.g. after changing the chunk settings."""
    total = 0
    for document in session.scalars(select(Document).order_by(Document.id)).all():
        total += chunk_document(session, data_dir, document, chunk_size, overlap)
    logger.info("rechunk finished chunks=%d", total)
    return total
