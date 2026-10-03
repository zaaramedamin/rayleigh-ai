from typing import Protocol


class LLMError(RuntimeError):
    """The local LLM could not produce a usable response."""


class LLMUnavailableError(LLMError):
    """The LLM server is not running or not reachable."""


class LLMModelNotFoundError(LLMError):
    """The configured model is not installed on the LLM server."""


class LLMTimeoutError(LLMError):
    """The LLM did not answer within the configured time."""


class LLMProvider(Protocol):
    """Generates text from a prompt. Implementations must run entirely on this machine."""

    model_name: str

    def generate(self, system: str, user: str) -> str:
        """Return the model's reply to `user`, following the instructions in `system`."""
        ...
