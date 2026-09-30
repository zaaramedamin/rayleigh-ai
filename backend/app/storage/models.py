from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """Metadata for one stored original file. The bytes live on disk at stored_path."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Display only. Never used to build a filesystem path.
    original_filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # Relative to DATA_DIR, always derived from content_hash.
    stored_path: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer)
    media_type: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
