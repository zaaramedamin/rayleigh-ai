from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.storage.files import resolve_inside

# Present in every model folder saved in sentence-transformers format.
MODEL_MARKER_FILE = "modules.json"


class ModelNotAvailableError(RuntimeError):
    """The embedding model has not been downloaded into MODELS_DIR."""


class EmbeddingRuntimeError(RuntimeError):
    """The embedding library is installed but cannot be loaded on this machine."""


class EmbeddingProvider(Protocol):
    """Turns text into vectors. Implementations must run entirely on this machine."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts to be stored and searched. Vectors are L2-normalised."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Vectors are L2-normalised."""
        ...


def model_dir_for(models_dir: Path, model_name: str) -> Path:
    """Folder that holds `model_name`, e.g. models/sentence-transformers__all-MiniLM-L6-v2.

    The name is validated by settings; resolve_inside is a second guard against escaping.
    """
    return resolve_inside(models_dir, model_name.replace("/", "__"))


def is_model_downloaded(models_dir: Path, model_name: str) -> bool:
    return (model_dir_for(models_dir, model_name) / MODEL_MARKER_FILE).is_file()
