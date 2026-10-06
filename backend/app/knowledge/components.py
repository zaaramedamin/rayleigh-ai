"""Builds the concrete embedder and vector store from settings, for the CLI and the API."""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.llm.ollama import OllamaProvider
from app.ai.speech.base import SpeechRecognizer
from app.core.config import Settings
from app.storage.vector_store import QdrantVectorStore, VectorStoreBusyError, collection_name


@lru_cache(maxsize=2)
def load_embedder(model_name: str, models_dir: Path) -> EmbeddingProvider:
    """Load the local model once per process. Raises ModelNotAvailableError if not downloaded."""
    from app.ai.embeddings.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(model_name, models_dir)


_RECOGNIZER_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_recognizer(model_name: str, models_dir: Path) -> SpeechRecognizer:
    from app.ai.speech.whisper import WhisperRecognizer

    return WhisperRecognizer(model_name, models_dir)


def load_recognizer(model_name: str, models_dir: Path) -> SpeechRecognizer:
    """Load the local speech model once per process.

    Raises SpeechModelNotAvailableError if it is not downloaded. Two requests arriving while it
    loads wait for the same copy instead of each loading their own.
    """
    with _RECOGNIZER_LOCK:
        return _load_recognizer(model_name, models_dir)


def vector_store_path(settings: Settings) -> Path:
    return settings.data_dir / "qdrant"


def create_vector_store(settings: Settings, embedder: EmbeddingProvider) -> QdrantVectorStore:
    """Open the embedded Qdrant store for the embedder's model. The caller must close it."""
    return QdrantVectorStore.open_local(
        vector_store_path(settings), collection_name(embedder.model_name), embedder.dimension
    )


@contextmanager
def open_vector_store(
    settings: Settings,
    embedder: EmbeddingProvider,
    *,
    wait_seconds: float = 0.0,
    on_wait: Callable[[], None] | None = None,
) -> Iterator[QdrantVectorStore]:
    """Open the embedded Qdrant store for the embedder's model, and always close it again.

    Only one process can have the store open. If another one (typically the server, in the middle
    of a request) has it, wait up to `wait_seconds` for it to finish instead of failing at once;
    `on_wait` is called once, the first time it has to wait.
    """
    deadline = time.monotonic() + wait_seconds
    waited = False
    while True:
        try:
            store = create_vector_store(settings, embedder)
            break
        except VectorStoreBusyError:
            if time.monotonic() >= deadline:
                raise
            if not waited and on_wait is not None:
                on_wait()
            waited = True
            time.sleep(0.25)
    try:
        yield store
    finally:
        store.close()


def create_llm(settings: Settings, timeout_seconds: float | None = None) -> OllamaProvider:
    """The local LLM configured in settings. Nothing is contacted until it is used."""
    return OllamaProvider(
        settings.ollama_url,
        settings.llm_model,
        timeout_seconds=timeout_seconds or settings.llm_timeout_seconds,
        think=settings.llm_think,
    )
