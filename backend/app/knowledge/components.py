"""Builds the concrete embedder and vector store from settings, for the CLI and the API."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from app.ai.embeddings.base import EmbeddingProvider
from app.core.config import Settings
from app.storage.vector_store import QdrantVectorStore, collection_name


@lru_cache(maxsize=2)
def load_embedder(model_name: str, models_dir: Path) -> EmbeddingProvider:
    """Load the local model once per process. Raises ModelNotAvailableError if not downloaded."""
    from app.ai.embeddings.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(model_name, models_dir)


def vector_store_path(settings: Settings) -> Path:
    return settings.data_dir / "qdrant"


def create_vector_store(settings: Settings, embedder: EmbeddingProvider) -> QdrantVectorStore:
    """Open the embedded Qdrant store for the embedder's model. The caller must close it."""
    return QdrantVectorStore.open_local(
        vector_store_path(settings), collection_name(embedder.model_name), embedder.dimension
    )


@contextmanager
def open_vector_store(
    settings: Settings, embedder: EmbeddingProvider
) -> Iterator[QdrantVectorStore]:
    """Open the embedded Qdrant store for the embedder's model, and always close it again."""
    store = create_vector_store(settings, embedder)
    try:
        yield store
    finally:
        store.close()
