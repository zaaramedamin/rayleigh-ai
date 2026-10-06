"""Background work for the library: scan the folders, then make new documents searchable.

One update runs at a time, in a thread, so the interface can show progress instead of waiting.
Nothing here logs or reports note text or file names; messages are fixed sentences.
"""

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.ai.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRuntimeError,
    ModelNotAvailableError,
    is_model_downloaded,
)
from app.api.deps import _engine, locked_vector_store
from app.core.config import Settings
from app.knowledge.components import load_embedder
from app.knowledge.indexing.service import (
    IndexProgress,
    count_pending,
    count_pending_chunks,
    count_stale_vectors,
    index_pending,
    purge_inactive_vectors,
)
from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.library.documents import clear_legacy_sources
from app.knowledge.library.folders import allowed_folders
from app.knowledge.library.jobs import finish_job, start_job
from app.knowledge.library.state import load_state
from app.storage.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

INDEX_BATCH_DOCUMENTS = 2  # the search index is released between batches so questions can run

SyncState = Literal["idle", "running", "done", "failed"]


@dataclass
class SyncStatus:
    state: SyncState = "idle"
    phase: str = ""
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    folders: int = 0
    added: int = 0
    unchanged: int = 0
    skipped_excluded: int = 0
    failed_files: int = 0
    # Documents replaced by an edited file, and documents whose file could not be found again.
    replaced: int = 0
    missing: int = 0
    indexing_total: int = 0
    indexed: int = 0
    # The same progress in chunks, which moves even while one large document is indexed.
    chunks_total: int = 0
    chunks_done: int = 0
    # The record of this run in the jobs table, once it exists.
    job_id: int | None = None


_lock = threading.Lock()
_status = SyncStatus()


def current_status() -> SyncStatus:
    with _lock:
        return SyncStatus(**asdict(_status))


def _update(**changes: object) -> None:
    with _lock:
        for key, value in changes.items():
            setattr(_status, key, value)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def start_sync(settings: Settings) -> bool:
    """Start an update unless one is already running. Returns whether it was started."""
    with _lock:
        if _status.state == "running":
            return False
        for key, value in asdict(
            SyncStatus(state="running", phase="scanning", started_at=_now())
        ).items():
            setattr(_status, key, value)
    threading.Thread(target=_run, args=(settings,), name="library-sync", daemon=True).start()
    return True


def _reason(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if isinstance(exc, ModelNotAvailableError):
        return "The embedding model is not downloaded. Run python -m app download-model."
    if isinstance(exc, (EmbeddingRuntimeError, VectorStoreError)):
        return str(exc)
    return "The update stopped because of an unexpected error. Check the server log."


def _record_start(settings: Settings) -> int | None:
    """Note in the database that an update began. Bookkeeping never stops an update."""
    try:
        with Session(_engine(settings.data_dir)) as session:
            job_id = start_job(session)
    except Exception:
        logger.warning("the update could not be recorded")
        return None
    _update(job_id=job_id)
    return job_id


def _record_finish(settings: Settings, job_id: int | None) -> None:
    if job_id is None:
        return
    status = current_status()
    state = status.state if status.state in ("done", "failed") else "failed"
    try:
        with Session(_engine(settings.data_dir)) as session:
            finish_job(session, job_id, state, status.message, asdict(status))
    except Exception:
        logger.warning("the update could not be recorded")


def _run(settings: Settings) -> None:
    job_id = _record_start(settings)
    try:
        _execute(settings)
    finally:
        _record_finish(settings, job_id)


def _execute(settings: Settings) -> None:
    try:
        engine = _engine(settings.data_dir)
        with Session(engine) as session:
            folders = [entry.path for entry in allowed_folders(settings)]
            _update(folders=len(folders))
            state = load_state(settings.data_dir)
            summary = ingest_folders(
                session,
                settings.data_dir,
                folders,
                settings.max_file_size_mb * 1024 * 1024,
                chunk_size=settings.chunk_size_chars,
                chunk_overlap=settings.chunk_overlap_chars,
                excluded_hashes=frozenset(state.excluded),
                legacy_sources=state.sources,
            )
            clear_legacy_sources(settings.data_dir)  # now recorded, encrypted, in the database
            _update(
                added=summary.added,
                unchanged=summary.unchanged,
                skipped_excluded=summary.skipped_excluded,
                failed_files=sum(summary.failed.values()),
                replaced=summary.sync.superseded,
                missing=summary.sync.newly_missing,
            )

            pending = count_pending(session, settings.embedding_model)
            stale = count_stale_vectors(session)
            if pending == 0 and stale == 0:
                _finish("done", None)
                return
            if not is_model_downloaded(settings.models_dir, settings.embedding_model):
                _finish(
                    "done",
                    f"{pending} document(s) are not searchable yet: the embedding model is not "
                    "downloaded. Run python -m app download-model, then update again."
                    if pending
                    else None,
                )
                return

            embedder = load_embedder(settings.embedding_model, settings.models_dir)
            if stale:
                with locked_vector_store(settings, embedder) as store:
                    purge_inactive_vectors(session, store)
            if pending == 0:
                _finish("done", None)
                return
            _update(
                phase="indexing",
                indexing_total=pending,
                chunks_total=count_pending_chunks(session, settings.embedding_model),
            )
            chunks_before = 0
            while True:
                with locked_vector_store(settings, embedder) as store:
                    done = index_pending(
                        session,
                        embedder,
                        store,
                        limit=INDEX_BATCH_DOCUMENTS,
                        on_progress=_chunk_progress(chunks_before),
                    )
                if done.documents_indexed == 0:
                    break
                chunks_before += done.chunks_embedded
                with _lock:
                    _status.indexed += done.documents_indexed
        _finish("done", None)
    except Exception as exc:
        logger.warning("library update failed: %s", type(exc).__name__)
        _finish("failed", _reason(exc))


def _chunk_progress(already_done: int) -> Callable[[IndexProgress], None]:
    """Report chunk progress of one index call, on top of what earlier calls finished."""

    def report(progress: IndexProgress) -> None:
        _update(chunks_done=already_done + progress.chunks_done)

    return report


def _finish(state: SyncState, message: str | None) -> None:
    _update(state=state, phase="", message=message, finished_at=_now())


def make_searchable(settings: Settings, session: Session, document_ids: Sequence[int]) -> bool:
    """Index just these documents now. Returns False if they could not be made searchable."""
    if not document_ids or not is_model_downloaded(settings.models_dir, settings.embedding_model):
        return False
    try:
        embedder = load_embedder(settings.embedding_model, settings.models_dir)
        with locked_vector_store(settings, embedder) as store:
            index_pending(session, embedder, store, only=list(document_ids))
    except (HTTPException, EmbeddingRuntimeError, ModelNotAvailableError, VectorStoreError):
        return False
    return True


def delete_vectors(settings: Settings, document_ids: Sequence[int]) -> bool:
    """Remove these documents' vectors if the index can be opened (leftovers are harmless)."""
    if not document_ids or not is_model_downloaded(settings.models_dir, settings.embedding_model):
        return False
    try:
        embedder: EmbeddingProvider = load_embedder(settings.embedding_model, settings.models_dir)
        with locked_vector_store(settings, embedder) as store:
            for document_id in document_ids:
                store.delete_document(document_id)
    except (HTTPException, EmbeddingRuntimeError, ModelNotAvailableError, VectorStoreError):
        return False
    return True
