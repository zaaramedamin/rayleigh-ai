from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.security.sqlalchemy_types import EncryptedString, EncryptedText
from app.storage.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Where a document stands. Only an "active" document is searched and answered from.
#   active:     the current version of a file that is still where it was found
#   superseded: an older version, replaced when the file was edited (its text is kept as history)
#   missing:    every place it was found is gone (deleted, moved out of the allowed folders)
DOC_ACTIVE = "active"
DOC_SUPERSEDED = "superseded"
DOC_MISSING = "missing"

# What a document is. The profile note is written by the owner in the interface, not read from a
# folder. This is a plain column because names are encrypted and cannot be searched for.
KIND_NOTE = "note"
KIND_PROFILE = "profile"

# A place a file was found. "missing" only after it was absent from two syncs in a row.
LOCATION_PRESENT = "present"
LOCATION_MISSING = "missing"


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
    status: Mapped[str] = mapped_column(String(12), default=DOC_ACTIVE, server_default=DOC_ACTIVE)
    # The document this one replaced when its file was edited. Not a foreign key: removing the old
    # version must not touch the new one.
    supersedes_id: Mapped[int | None] = mapped_column(Integer, default=None)
    kind: Mapped[str] = mapped_column(String(12), default=KIND_NOTE, server_default=KIND_NOTE)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentSource(Base):
    """One place a document's file was found: an allowed folder and a path inside it.

    The same content in two places is one Document with two locations, and an edited file keeps its
    location while the document behind it changes. Both path columns are encrypted like file
    names, so a location cannot be looked up in SQL: sync loads them and compares in Python.
    """

    __tablename__ = "document_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # The allowed folder, as an absolute path.
    source_root: Mapped[str] = mapped_column(EncryptedString("document_sources.source_root", 1000))
    # The file inside it, relative, with forward slashes.
    source_path: Mapped[str] = mapped_column(EncryptedString("document_sources.source_path", 1000))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    status: Mapped[str] = mapped_column(
        String(10), default=LOCATION_PRESENT, server_default=LOCATION_PRESENT
    )
    # Consecutive syncs that did not find the file. Two make it "missing".
    misses: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    missing_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


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
    # The pages the chunk comes from (1-based), for formats that have pages such as PDF.
    start_page: Mapped[int | None] = mapped_column(Integer, default=None)
    end_page: Mapped[int | None] = mapped_column(Integer, default=None)
    char_count: Mapped[int] = mapped_column(Integer)
    # The embedding model whose vector for this chunk is in the vector store. None: not
    # embedded yet. An interrupted index run continues at the first chunk without one.
    indexed_model: Mapped[str | None] = mapped_column(String(255), default=None)

    document: Mapped[Document] = relationship(back_populates="chunks")

    @property
    def citation_id(self) -> str:
        """Stable, application-generated id for citations: '<document_id>:<chunk_index>'."""
        return f"{self.document_id}:{self.chunk_index}"


class Memory(Base):
    """One thing the assistant keeps about the owner from one conversation to the next."""

    __tablename__ = "assistant_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(EncryptedText("assistant_memories.text"))
    # Who saved it: "owner" (typed on the Profile page) or "assistant" (during a conversation).
    origin: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AssistantSetting(Base):
    """One setting of the assistant's identity (its name, its role, ...), as key and value."""

    __tablename__ = "assistant_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(50), unique=True)
    value: Mapped[str] = mapped_column(EncryptedText("assistant_settings.value"))


class Job(Base):
    """One run of a library update. Counts and fixed sentences only: no names, paths or text."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))
    # running, done, failed, or interrupted (the program stopped while it was running).
    state: Mapped[str] = mapped_column(String(12))
    message: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    added: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    unchanged: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    replaced: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    missing: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    skipped_excluded: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    failed_files: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    indexed: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunks_done: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


class Conversation(Base):
    """A conversation with the assistant, kept so it can be reopened later.

    The title is made of the owner's own words, so it is encrypted like note text.
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(EncryptedString("conversations.title", 200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class Message(Base):
    """One turn of a conversation: something the owner asked, or the answer it got."""

    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("conversation_id", "position", name="uq_messages_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Deleting a conversation deletes its messages in the database itself.
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    # 0, 1, 2, ... in the order the turns were spoken.
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(10))  # "user" or "assistant"
    mode: Mapped[str] = mapped_column(String(10))  # "general" or "notes"
    content: Mapped[str] = mapped_column(EncryptedText("messages.content"))
    # What the interface shows beside an answer (its sources, whether it was grounded...), as JSON.
    payload: Mapped[str | None] = mapped_column(EncryptedText("messages.payload"), default=None)
    # Numbers of the library documents this answer quoted, as ",12,40,". Plain on purpose: removing
    # a document must find the answers that quoted it without decrypting every message.
    cited_documents: Mapped[str] = mapped_column(String(2000), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
