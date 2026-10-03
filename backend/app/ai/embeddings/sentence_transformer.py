import os
from collections.abc import Sequence
from pathlib import Path

from app.ai.embeddings.base import (
    EmbeddingRuntimeError,
    ModelNotAvailableError,
    is_model_downloaded,
    model_dir_for,
)
from app.ai.embeddings.compat import import_sentence_transformer


def _force_offline() -> None:
    """Make sure the Hugging Face libraries never reach the network from this process.

    The model is always loaded from a local folder; these flags turn any accidental
    download attempt into an error instead of a silent network call.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class SentenceTransformerProvider:
    """Local embeddings with a sentence-transformers model stored under MODELS_DIR."""

    def __init__(self, model_name: str, models_dir: Path, batch_size: int = 32) -> None:
        if not is_model_downloaded(models_dir, model_name):
            raise ModelNotAvailableError(
                f"embedding model {model_name!r} is not downloaded; "
                "run `python -m app download-model`"
            )
        _force_offline()
        # Deferred: importing torch takes seconds and is only needed once a model is used.
        try:
            SentenceTransformer = import_sentence_transformer()
        except ImportError as exc:
            hint = ""
            if "Application Control" in str(exc):
                hint = (
                    " Windows (Smart App Control) blocked a library file; re-running once "
                    "while online sometimes clears it."
                )
            raise EmbeddingRuntimeError(
                f"the embedding library could not be loaded: {str(exc).rstrip('.')}.{hint}"
            ) from exc

        self._model = SentenceTransformer(
            str(model_dir_for(models_dir, model_name)),
            device="cpu",
            trust_remote_code=False,  # never run code shipped with a model
            local_files_only=True,
        )
        self.model_name = model_name
        self.dimension = int(self._model.get_embedding_dimension())
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
