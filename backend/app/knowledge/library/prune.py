"""Remove documents whose files are gone, once the owner has said so.

A sync only *marks* a document as missing (its file was not found in two scans in a row) or
superseded (its file was edited). Nothing is deleted until `prune` is run and confirmed. A prune
removes the document, its chunks, its vectors and its stored copy, and says exactly what went.
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.library.documents import Removed, delete_document
from app.storage.models import DOC_MISSING, DOC_SUPERSEDED, Document
from app.storage.vector_store import VectorStore


@dataclass
class PruneCandidates:
    missing: list[Document]
    superseded: list[Document]

    @property
    def total(self) -> int:
        return len(self.missing) + len(self.superseded)


def find_candidates(session: Session, *, include_superseded: bool) -> PruneCandidates:
    """The documents a prune would remove. Looking never changes anything."""

    def with_status(status: str) -> list[Document]:
        return list(
            session.scalars(
                select(Document).where(Document.status == status).order_by(Document.id)
            ).all()
        )

    return PruneCandidates(
        missing=with_status(DOC_MISSING),
        superseded=with_status(DOC_SUPERSEDED) if include_superseded else [],
    )


def prune(
    session: Session, data_dir: Path, candidates: PruneCandidates, store: VectorStore | None
) -> Removed:
    """Remove the candidates. A superseded version already removed with a newer one is skipped."""
    removed = Removed()
    ids = [document.id for document in [*candidates.missing, *candidates.superseded]]
    for document_id in ids:
        document = session.get(Document, document_id)
        if document is None:
            continue  # taken away together with a newer version of the same file
        removed.add(delete_document(session, data_dir, document, exclude=False, store=store))
    return removed
