from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["user", "assistant"]


class LLMError(RuntimeError):
    """The local LLM could not produce a usable response."""


class LLMUnavailableError(LLMError):
    """The LLM server is not running or not reachable."""


class LLMModelNotFoundError(LLMError):
    """The configured model is not installed on the LLM server."""


class LLMTimeoutError(LLMError):
    """The LLM did not answer within the configured time."""


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a conversation: something the user said, or something the model replied."""

    role: Role
    content: str


@dataclass(frozen=True)
class ChatReply:
    text: str
    truncated: bool = False  # the model reached its length limit before it finished


class LLMProvider(Protocol):
    """Generates text from a prompt. Implementations must run entirely on this machine."""

    model_name: str

    def generate(self, system: str, user: str) -> str:
        """Return the model's reply to `user`, following the instructions in `system`."""
        ...

    def chat(
        self, system: str, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> ChatReply:
        """Continue a conversation. `messages` is oldest first and ends with the user's turn."""
        ...
