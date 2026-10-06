from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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
class ToolSpec:
    """Something the model may ask the application to do. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to use a tool. Untrusted: the application validates it."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ChatReply:
    text: str
    truncated: bool = False  # the model reached its length limit before it finished
    # What the model asked the application to do. `text` may be empty when this is not.
    tool_calls: tuple[ToolCall, ...] = ()


class LLMProvider(Protocol):
    """Generates text from a prompt. Implementations must run entirely on this machine."""

    model_name: str

    def generate(self, system: str, user: str) -> str:
        """Return the model's reply to `user`, following the instructions in `system`."""
        ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        """The same reply as `generate`, as the model writes it, piece by piece.

        Closing the iterator early stops the model. Errors are raised from the first `next()`
        (the server is down, the model is missing) or in the middle of the stream.
        """
        ...

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        """Continue a conversation. `messages` is oldest first and ends with the user's turn.

        With `tools`, the model may ask for some of them instead of (or as well as) replying in
        words. A model that cannot use tools simply replies in words.
        """
        ...
