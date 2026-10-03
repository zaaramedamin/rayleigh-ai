import hashlib
import math
import re
from collections.abc import Sequence


class HashingEmbedder:
    """Deterministic stand-in for a real embedding model, for fast offline tests.

    Each word is hashed into one vector slot, so texts that share words get similar vectors.
    """

    def __init__(self, dimension: int = 256, model_name: str = "test/hashing-embedder") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.texts_embedded = 0

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Scripted stand-in for a local LLM. Records every prompt it is given."""

    def __init__(
        self,
        reply: str = "ready",
        *,
        error: Exception | None = None,
        model_name: str = "test/fake-llm",
        installed: Sequence[str] = ("test/fake-llm",),
    ) -> None:
        self.reply = reply
        self.error = error
        self.model_name = model_name
        self.installed = list(installed)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.reply

    def list_models(self) -> list[str]:
        if self.error is not None:
            raise self.error
        return self.installed
