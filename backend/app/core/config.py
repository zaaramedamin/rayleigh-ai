import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

# "organisation/model-name", as on Hugging Face. Also used to build a folder name under MODELS_DIR.
MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")

# An Ollama model tag, e.g. "qwen3.5:4b" or "llama3.2".
LLM_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*(:[A-Za-z0-9_.-]+)?")

# A spoken-language code as Whisper names them, e.g. "en", "fr", "ar".
SPEECH_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}")

# Upper bound for the number of search results per query.
MAX_TOP_K = 50

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_loopback_url(value: str) -> str:
    """Accept only a plain http(s) URL on this machine, so prompts can never leave it."""
    url = value.strip()
    parts = urlsplit(url)
    _ = parts.port  # raises ValueError for a malformed port
    if (
        parts.scheme not in ("http", "https")
        or parts.hostname not in LOOPBACK_HOSTS
        or parts.username
        or parts.password
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "must be a plain http(s) URL on this machine, such as http://127.0.0.1:11434; "
            "remote servers are not allowed"
        )
    return url.rstrip("/")


class Settings(BaseSettings):
    """Application settings, read from environment variables and the repo-root .env file."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    # Allow-list of folders that may be indexed. Empty means nothing can be indexed.
    allowed_folders: Annotated[list[Path], NoDecode] = []
    # Files larger than this are skipped during ingestion.
    max_file_size_mb: int = Field(default=5, gt=0)
    # Chunking, in characters. Overlap must be smaller than the chunk size.
    chunk_size_chars: int = Field(default=1000, ge=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    # Local embedding model (sentence-transformers). Downloaded once into MODELS_DIR.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Number of chunks a search returns by default.
    retrieval_top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    # How notes are found: by meaning (vector), by the words in the question (keyword), or both
    # merged (hybrid). Whatever the mode, the relevance gate still uses meaning similarity.
    search_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    # Minimum similarity for a note to be used to answer. Weaker matches are ignored, and if
    # none is left the assistant says it does not have enough information. Tune with the
    # evaluation set.
    answer_min_score: float = Field(default=0.30, ge=0.0, le=1.0)
    # Local LLM, served by Ollama on this machine. Remote URLs are rejected.
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:4b"
    llm_timeout_seconds: int = Field(default=120, ge=5, le=900)
    # Let the model reason before answering: much slower, sometimes better.
    llm_think: bool = False
    # Files that need a real parser (PDF) are read in a separate process with these limits.
    parser_timeout_seconds: int = Field(default=120, ge=5, le=900)
    pdf_max_pages: int = Field(default=2000, ge=1, le=20000)
    # Lines that repeat at the top or bottom of most pages (running headers, page numbers) are
    # dropped from PDFs, because they would match nearly every question. Keep them if they
    # hold real content.
    pdf_keep_headers_footers: bool = False
    # Names besides 127.0.0.1, localhost and ::1 that the API answers to (comma separated).
    # Any other Host header is refused, which is what stops DNS rebinding.
    allowed_hosts: Annotated[list[str], NoDecode] = []
    # Other web pages allowed to read the API's answers, e.g. http://127.0.0.1:5173. Empty means
    # none: the interface is served from the same address and needs no cross-site access.
    cors_origins: Annotated[list[str], NoDecode] = []
    # The largest request the API reads, and how many it accepts per minute.
    max_request_mb: int = Field(default=8, ge=1, le=256)
    rate_limit_per_minute: int = Field(default=1200, ge=10, le=100000)
    # Stored conversations that were not touched for this many days are deleted. 0 keeps them
    # until you delete them.
    conversation_retention_days: int = Field(default=0, ge=0, le=3650)
    # Require the access password for every API route except /health and /auth.
    access_required: bool = True
    # Local speech recognition (Whisper) for voice orders. Downloaded once into MODELS_DIR.
    speech_model: str = "openai/whisper-base"
    # The language you speak to it, as a code such as "en" or "fr". Empty: detected each time.
    speech_language: str = ""

    @field_validator("allowed_hosts", "cors_origins", mode="before")
    @classmethod
    def _split_names(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("allowed_folders", mode="before")
    @classmethod
    def _split_allowed_folders(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("embedding_model")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        if not MODEL_NAME_PATTERN.fullmatch(value):
            raise ValueError("EMBEDDING_MODEL must look like 'organisation/model-name'")
        return value

    @field_validator("speech_model")
    @classmethod
    def _validate_speech_model(cls, value: str) -> str:
        if not MODEL_NAME_PATTERN.fullmatch(value):
            raise ValueError("SPEECH_MODEL must look like 'organisation/model-name'")
        return value

    @field_validator("speech_language")
    @classmethod
    def _validate_speech_language(cls, value: str) -> str:
        code = value.strip().lower()
        if code and not SPEECH_LANGUAGE_PATTERN.fullmatch(code):
            raise ValueError("SPEECH_LANGUAGE must be a language code such as 'en', or empty")
        return code

    @field_validator("ollama_url")
    @classmethod
    def _validate_ollama_url(cls, value: str) -> str:
        return validate_loopback_url(value)

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model(cls, value: str) -> str:
        if not LLM_MODEL_PATTERN.fullmatch(value):
            raise ValueError("LLM_MODEL must be an Ollama model tag such as 'qwen3.5:4b'")
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
        return self

    @field_validator("data_dir", "models_dir")
    @classmethod
    def _anchor_to_repo_root(cls, value: Path) -> Path:
        # Relative paths resolve against the repo root, not whatever directory the server ran from.
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
