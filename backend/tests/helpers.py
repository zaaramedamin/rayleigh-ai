from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingProvider
from app.knowledge.chunking.service import chunk_document
from app.knowledge.indexing.service import index_pending
from app.storage.files import save_file
from app.storage.models import Document
from app.storage.vector_store import VectorStore


def add_and_index(
    session: Session,
    data_dir: Path,
    embedder: EmbeddingProvider,
    store: VectorStore,
    notes: dict[str, str],
    chunk_size: int = 1000,
) -> dict[str, Document]:
    """Store, chunk and index notes given as {file name: content}."""
    documents = {}
    for name, content in notes.items():
        document = save_file(session, data_dir, content.encode(), name)
        chunk_document(session, data_dir, document, chunk_size, 50)
        documents[name] = document
    index_pending(session, embedder, store)
    return documents
