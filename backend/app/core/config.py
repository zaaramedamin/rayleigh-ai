from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


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

    @field_validator("allowed_folders", mode="before")
    @classmethod
    def _split_allowed_folders(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("data_dir", "models_dir")
    @classmethod
    def _anchor_to_repo_root(cls, value: Path) -> Path:
        # Relative paths resolve against the repo root, not whatever directory the server ran from.
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
