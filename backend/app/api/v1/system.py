from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app import __version__
from app.ai.embeddings.base import is_model_downloaded
from app.ai.llm.base import LLMError, LLMTimeoutError, LLMUnavailableError
from app.api.deps import SessionDep, SettingsDep
from app.core.config import Settings
from app.knowledge.components import create_llm
from app.knowledge.indexing.service import library_counts
from app.security import keystore

router = APIRouter(prefix="/system", tags=["system"])

# Short, because this is a status check. Not shorter: Windows takes about two seconds to
# refuse a connection to a port nothing listens on, and that must read as "not running".
LLM_PROBE_TIMEOUT_SECONDS = 5


class EmbeddingStatus(BaseModel):
    model: str
    downloaded: bool


class LibraryStatus(BaseModel):
    documents: int
    chunks: int
    searchable_documents: int = Field(description="Documents whose chunks are in the search index.")
    pending_documents: int = Field(description="Documents that still need indexing.")
    missing_documents: int = Field(
        default=0, description="Documents whose file was not found in the last two updates."
    )


class LLMStatus(BaseModel):
    model: str
    state: Literal["ready", "model_missing", "not_running", "error"]
    hint: str | None = Field(default=None, description="What to do about it, if anything.")


class SystemStatus(BaseModel):
    embedding: EmbeddingStatus
    library: LibraryStatus
    llm: LLMStatus


def _llm_status(settings: Settings) -> LLMStatus:
    """Ask Ollama which models it has. Messages are fixed text: no exception detail is relayed."""
    model = settings.llm_model
    try:
        installed = create_llm(settings, timeout_seconds=LLM_PROBE_TIMEOUT_SECONDS).list_models()
    except (LLMUnavailableError, LLMTimeoutError):
        return LLMStatus(
            model=model,
            state="not_running",
            hint="Open the Ollama app or run `ollama serve`.",
        )
    except LLMError:
        return LLMStatus(model=model, state="error", hint="Ollama answered with a problem.")
    wanted = model if ":" in model else f"{model}:latest"
    if wanted in installed:
        return LLMStatus(model=model, state="ready")
    return LLMStatus(model=model, state="model_missing", hint=f"Run `ollama pull {model}`.")


@router.get("/status")
def system_status(session: SessionDep, settings: SettingsDep) -> SystemStatus:
    """What is installed and indexed, and whether the local model is reachable.

    Counts and model names only: no note text, file names or questions.
    """
    counts = library_counts(session, settings.embedding_model)
    return SystemStatus(
        embedding=EmbeddingStatus(
            model=settings.embedding_model,
            downloaded=is_model_downloaded(settings.models_dir, settings.embedding_model),
        ),
        library=LibraryStatus(
            documents=counts.active,
            chunks=counts.chunks,
            searchable_documents=counts.searchable,
            pending_documents=counts.pending,
            missing_documents=counts.missing,
        ),
        llm=_llm_status(settings),
    )


class SettingsOut(BaseModel):
    """What the program is set to do. No folders, paths, passwords or keys."""

    version: str
    embedding_model: str
    llm_model: str
    ollama_url: str
    speech_model: str
    search_mode: Literal["vector", "keyword", "hybrid"]
    retrieval_top_k: int
    answer_min_score: float
    chunk_size_chars: int
    chunk_overlap_chars: int
    max_file_size_mb: int
    parser_timeout_seconds: int
    pdf_max_pages: int
    pdf_keep_headers_footers: bool
    max_request_mb: int
    rate_limit_per_minute: int
    access_required: bool
    library_encrypted: bool


@router.get("/settings")
def system_settings(settings: SettingsDep) -> SettingsOut:
    """The current settings, read-only. Change them in the .env file and restart."""
    return SettingsOut(
        version=__version__,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        ollama_url=settings.ollama_url,
        speech_model=settings.speech_model,
        search_mode=settings.search_mode,
        retrieval_top_k=settings.retrieval_top_k,
        answer_min_score=settings.answer_min_score,
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        max_file_size_mb=settings.max_file_size_mb,
        parser_timeout_seconds=settings.parser_timeout_seconds,
        pdf_max_pages=settings.pdf_max_pages,
        pdf_keep_headers_footers=settings.pdf_keep_headers_footers,
        max_request_mb=settings.max_request_mb,
        rate_limit_per_minute=settings.rate_limit_per_minute,
        access_required=settings.access_required,
        library_encrypted=keystore.library_state(settings.data_dir) == "encrypted",
    )
