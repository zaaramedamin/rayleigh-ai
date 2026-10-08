import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from typing import Any

from app.ai.llm.base import ChatMessage, ChatReply, ToolCall, ToolSpec
from app.ai.speech.base import Samples


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
        self.chats: list[tuple[str, list[ChatMessage], float]] = []
        self.truncated = False
        # Replies given to the next `generate` calls, one each, before `reply` is used again.
        self.script: list[str] = []
        # For `stream`: how many characters each piece holds, an error to raise once that many
        # pieces were sent, and how many streams were closed before they ended.
        self.piece_chars = 4
        self.stream_error_after: tuple[int, Exception] | None = None
        self.streams_closed_early = 0
        # What the model asks the application to do, and the tools it was offered each time.
        self.tool_calls: list[ToolCall] = []
        self.tools_offered: list[list[ToolSpec]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.script.pop(0) if self.script else self.reply

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        text = self.script.pop(0) if self.script else self.reply
        sent = 0
        finished = False
        try:
            for start in range(0, len(text), self.piece_chars):
                if self.stream_error_after is not None and sent >= self.stream_error_after[0]:
                    raise self.stream_error_after[1]
                yield text[start : start + self.piece_chars]
                sent += 1
            finished = True
        finally:
            if not finished:
                self.streams_closed_early += 1

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        self.chats.append((system, list(messages), temperature))
        self.tools_offered.append(list(tools))
        if self.error is not None:
            raise self.error
        return ChatReply(
            text=self.reply, truncated=self.truncated, tool_calls=tuple(self.tool_calls)
        )

    def list_models(self) -> list[str]:
        if self.error is not None:
            raise self.error
        return self.installed


class FakeRecognizer:
    """Scripted stand-in for the speech model. Records what it was asked to transcribe."""

    def __init__(self, text: str = "open the settings page", *, error: Exception | None = None):
        self.text = text
        self.error = error
        self.model_name = "test/fake-speech"
        self.heard: list[tuple[int, str | None]] = []  # (number of samples, language)

    def transcribe(self, samples: Samples, language: str | None = None) -> str:
        self.heard.append((int(samples.size), language))
        if self.error is not None:
            raise self.error
        return self.text


class ScriptedModel:
    """A model that does what the test says, one turn at a time, and remembers what it was given."""

    model_name = "test/scripted"

    def __init__(self, *turns: ChatReply | Exception) -> None:
        self.turns = list(turns)
        self.chats: list[tuple[str, list[ChatMessage], list[ToolSpec]]] = []
        # What the model says when the agent first reads the request (see app.agent.understand),
        # and what it was shown. Nothing scripted means a reply that cannot be read.
        self.readings: list[str] = []
        self.read: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.read.append((system, user))
        return self.readings.pop(0) if self.readings else ""

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        self.chats.append((system, list(messages), list(tools)))
        if not self.turns:
            raise AssertionError("the model was asked for more turns than the test scripted")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def says(text: str) -> ChatReply:
    return ChatReply(text=text)


def asks(name: str, **arguments: Any) -> ChatReply:
    return ChatReply(text="", tool_calls=(ToolCall(name, arguments),))
