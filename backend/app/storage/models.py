from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.security.sqlalchemy_types import EncryptedString, EncryptedText
from app.storage.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """Metadata for one stored original file. The bytes live on disk at stored_path."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Display only. Never used to build a filesystem path.
    original_filename: Mapped[str] = mapped_column(
        EncryptedString("documents.original_filename", 255)
    )
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # Relative to DATA_DIR, always derived from content_hash.
    stored_path: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer)
    media_type: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Embedding model whose vectors for this document's current chunks are in the vector store.
    # None means "not searchable yet": never indexed, or re-chunked since.
    indexed_model: Mapped[str | None] = mapped_column(String(255), default=None)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """A searchable piece of a document, with provenance back to its source.

    start_line/end_line are 1-based lines of the document's extracted text. For plain text and
    Markdown this is the same as the line numbers in the original file.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(EncryptedText("chunks.text"))
    # e.g. "Project > Setup". Empty for text without headings.
    heading_path: Mapped[str] = mapped_column(
        EncryptedString("chunks.heading_path", 1000), default=""
    )
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")

    @property
    def citation_id(self) -> str:
        """Stable, application-generated id for citations: '<document_id>:<chunk_index>'."""
        return f"{self.document_id}:{self.chunk_index}"
