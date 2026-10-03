"""FastAPI dependencies: database session, embedding model, vector store and LLM per request."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.ai.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRuntimeError,
    ModelNotAvailableError,
)
from app.ai.llm.base import LLMProvider
from app.core.config import Settings, get_settings
from app.knowledge.components import create_llm, create_vector_store, load_embedder
from app.storage.database import create_db_engine
from app.storage.migrations import database_is_up_to_date
from app.storage.vector_store import VectorStore, VectorStoreError

# The embedded vector store can only be open once per process, so requests take turns.
_STORE_LOCK = threading.Lock()
_STORE_WAIT_SECONDS = 30

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def _engine(data_dir: Path) -> Engine:
    return create_db_engine(data_dir)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def get_session(settings: SettingsDep) -> Iterator[Session]:
    engine = _engine(settings.data_dir)
    if not database_is_up_to_date(engine):
        raise _unavailable("Database missing or out of date. Run `alembic upgrade head`.")
    with Session(engine) as session:
        yield session


def get_embedder(settings: SettingsDep) -> EmbeddingProvider:
    try:
        return load_embedder(settings.embedding_model, settings.models_dir)
    except ModelNotAvailableError as exc:
        raise _unavailable(
            "Embedding model not downloaded. Run `python -m app download-model`."
        ) from exc
    except EmbeddingRuntimeError as exc:
        raise _unavailable(str(exc)) from exc


@contextmanager
def locked_vector_store(settings: Settings, embedder: EmbeddingProvider) -> Iterator[VectorStore]:
    """Open the vector store for the duration of the block, one request at a time.

    Hold it only while searching: never while waiting for the LLM, which can take minutes.
    """
    if not _STORE_LOCK.acquire(timeout=_STORE_WAIT_SECONDS):
        raise _unavailable("The search index is busy. Try again.")
    try:
        try:
            store = create_vector_store(settings, embedder)
        except VectorStoreError as exc:
            raise _unavailable(str(exc)) from exc
        try:
            yield store
        finally:
            store.close()
    finally:
        _STORE_LOCK.release()


def get_vector_store(
    settings: SettingsDep, embedder: Annotated[EmbeddingProvider, Depends(get_embedder)]
) -> Iterator[VectorStore]:
    with locked_vector_store(settings, embedder) as store:
        yield store


def get_llm(settings: SettingsDep) -> LLMProvider:
    return create_llm(settings)


SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
