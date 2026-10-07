"""LLM provider for a local Ollama server (https://ollama.com), using only the standard library."""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from app.ai.llm.base import (
    ChatMessage,
    ChatReply,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
    ToolCall,
    ToolSpec,
)
from app.ai.llm.budget import (
    CONTEXT_TOKENS,
    REPLY_TOKENS,
    THINKING_REPLY_TOKENS,
    input_chars,
)
from app.core.config import validate_loopback_url

logger = logging.getLogger(__name__)

# Context window requested from Ollama, in tokens. The prompt builder keeps the notes under
# 6000 characters, so the system prompt can never be pushed out of the window.
DEFAULT_NUM_CTX = CONTEXT_TOKENS
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
# Fields a model may not support. Ollama names the feature in its refusal, and the request is
# then sent again without that field: (field in the request, word in the refusal).
_OPTIONAL_FIELDS = (("think", "think"), ("tools", "tool"))
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


class _RejectedError(LLMError):
    """Ollama refused the request itself (HTTP 4xx), as opposed to being down or slow."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a prompt must never be re-sent to another address."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        timeout_seconds: float = 120,
        think: bool = False,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self._base_url = validate_loopback_url(base_url)
        self.model_name = model_name
        self._timeout = timeout_seconds
        self._think = think
        self._num_ctx = num_ctx
        # No proxies (a proxy would see the prompt) and no redirects.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @property
    def reply_tokens(self) -> int:
        """The room kept free for the reply (the most the model may write)."""
        return THINKING_REPLY_TOKENS if self._think else REPLY_TOKENS

    @property
    def input_chars(self) -> int:
        """The most characters a prompt may hold so that the reply still fits in the window."""
        return input_chars(self._num_ctx, self.reply_tokens)

    # --- HTTP ---------------------------------------------------------------------------------

    def _open(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """Send a request and return the open HTTP response. Failures become LLM errors."""
        request = urllib.request.Request(
            self._base_url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            return self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except TimeoutError as exc:
            raise self._timeout_error() from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timeout_error() from exc
            raise LLMUnavailableError(
                f"Ollama is not reachable at {self._base_url}. "
                "Start it (open the Ollama app or run `ollama serve`) and try again."
            ) from exc
        except OSError as exc:  # connection reset while connecting, etc.
            raise self._connection_error(exc) from exc

    def _connection_error(self, exc: OSError) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"connection to Ollama at {self._base_url} failed ({type(exc).__name__})"
        )

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._open(path, body)
        try:
            with response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError as exc:
            raise self._timeout_error() from exc
        except OSError as exc:  # connection reset while reading, etc.
            raise self._connection_error(exc) from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMError("Ollama returned an unexpectedly large response")
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise LLMError("Ollama returned a response that is not valid JSON") from exc
        if not isinstance(data, dict):
            raise LLMError("Ollama returned an unexpected response")
        return data

    def _timeout_error(self) -> LLMTimeoutError:
        return LLMTimeoutError(
            f"the model did not answer within {self._timeout:g} seconds "
            "(raise LLM_TIMEOUT_SECONDS, or use a smaller model)"
        )

    def _http_error(self, exc: urllib.error.HTTPError) -> LLMError:
        detail = _error_detail(exc)
        if exc.code == 404:
            return LLMModelNotFoundError(
                f"model {self.model_name!r} is not installed in Ollama. "
                f"Run `ollama pull {self.model_name}`."
            )
        if 300 <= exc.code < 400:
            return LLMError(f"Ollama answered with a redirect (HTTP {exc.code}), which is refused")
        if exc.code >= 500:
            return LLMUnavailableError(
                f"Ollama is not ready (HTTP {exc.code}{detail}). Try again in a moment."
            )
        return _RejectedError(f"Ollama rejected the request (HTTP {exc.code}{detail})")

    # --- API ----------------------------------------------------------------------------------

    def generate(self, system: str, user: str) -> str:
        # Temperature 0: the same notes and question should give the same answer.
        return self._complete(system, [ChatMessage("user", user)], temperature=0).text

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        return self._complete(system, messages, temperature=temperature, tools=tools)

    def _chat_body(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        tools: Sequence[ToolSpec] = (),
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                *({"role": message.role, "content": message.content} for message in messages),
            ],
            "stream": stream,
            "think": self._think,
            "options": {
                "temperature": temperature,
                "num_ctx": self._num_ctx,
                "num_predict": self.reply_tokens,  # bound a runaway answer
            },
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        return body

    def _complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        body = self._chat_body(system, messages, temperature=temperature, tools=tools)
        started = time.monotonic()
        data = self._request_without_unsupported_fields(body)

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("Ollama returned a response without a message")
        content = _THINK_BLOCK.sub("", content).strip()
        truncated = data.get("done_reason") == "length"
        # Only a request that offered tools can be answered with tool calls.
        tool_calls = _tool_calls(message) if "tools" in body else ()
        if not content and not tool_calls:
            if truncated:
                raise LLMError(
                    "the model used up its whole answer budget before writing an answer"
                    + (" while reasoning; set LLM_THINK=false" if self._think else "")
                )
            raise LLMError("the model returned an empty answer")
        # Log timing and sizes only, never the prompt or the answer.
        logger.info(
            "llm finished model=%s seconds=%.1f answer_chars=%d tool_calls=%d",
            self.model_name,
            time.monotonic() - started,
            len(content),
            len(tool_calls),
        )
        return ChatReply(text=content, truncated=truncated, tool_calls=tool_calls)

    def _request_without_unsupported_fields(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send a chat request; drop `think` or `tools` and ask again if the model refuses them."""
        while True:
            try:
                return self._request("/api/chat", body)
            except _RejectedError as exc:
                self._drop_refused_fields(body, exc)

    @staticmethod
    def _drop_refused_fields(body: dict[str, Any], exc: _RejectedError) -> None:
        """Remove the optional fields Ollama named in its refusal, or raise it again."""
        reason = str(exc).lower()
        refused = [name for name, word in _OPTIONAL_FIELDS if name in body and word in reason]
        if not refused:
            raise exc
        for name in refused:
            del body[name]

    def stream(self, system: str, user: str) -> Iterator[str]:
        body = self._chat_body(system, [ChatMessage("user", user)], temperature=0, stream=True)
        started = time.monotonic()
        while True:
            try:
                response = self._open("/api/chat", body)
                break
            except _RejectedError as exc:
                self._drop_refused_fields(body, exc)

        hide_thinking = _ThinkFilter()
        received = 0
        shown = 0
        try:
            with response:
                for line in response:
                    received += len(line)
                    if received > MAX_RESPONSE_BYTES:
                        raise LLMError("Ollama returned an unexpectedly large response")
                    data = _stream_line(line)
                    message = data.get("message")
                    piece = message.get("content") if isinstance(message, dict) else None
                    if isinstance(piece, str) and piece:
                        text = hide_thinking.feed(piece)
                        if text:
                            shown += len(text)
                            yield text
                    if data.get("done"):
                        break
                text = hide_thinking.flush()
                if text:
                    shown += len(text)
                    yield text
        except TimeoutError as exc:
            raise self._timeout_error() from exc
        except OSError as exc:
            raise self._connection_error(exc) from exc
        if shown == 0:
            raise LLMError("the model returned an empty answer")
        # Timing and sizes only, never the prompt or the answer.
        logger.info(
            "llm streamed model=%s seconds=%.1f answer_chars=%d",
            self.model_name,
            time.monotonic() - started,
            shown,
        )

    def list_models(self) -> list[str]:
        """Names of the models installed in Ollama. Raises LLMUnavailableError if it is down."""
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list):
            raise LLMError("Ollama returned an unexpected model list")
        return [str(m["name"]) for m in models if isinstance(m, dict) and "name" in m]


def _stream_line(line: bytes) -> dict[str, Any]:
    """One line of Ollama's streamed reply. An error reported in the middle of it is raised."""
    try:
        data = json.loads(line)
    except ValueError as exc:
        raise LLMError("Ollama returned a response that is not valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("Ollama returned an unexpected response")
    if "error" in data:
        # Its own wording can quote the request, so it is not repeated.
        raise LLMError("Ollama reported an error while the model was answering")
    return data


class _ThinkFilter:
    """Removes <think>...</think> from a stream, even when a tag is split between two pieces."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        """How many characters at the end of `text` could be the start of `tag`."""
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def feed(self, piece: str) -> str:
        self._buffer += piece
        out: list[str] = []
        while True:
            if self._inside:
                end = self._buffer.find(_THINK_CLOSE)
                if end < 0:
                    # Keep only what might be the start of the closing tag.
                    keep = self._partial_tail(self._buffer, _THINK_CLOSE)
                    self._buffer = self._buffer[len(self._buffer) - keep :]
                    break
                self._buffer = self._buffer[end + len(_THINK_CLOSE) :]
                self._inside = False
                continue
            start = self._buffer.find(_THINK_OPEN)
            if start < 0:
                keep = self._partial_tail(self._buffer, _THINK_OPEN)
                out.append(self._buffer[: len(self._buffer) - keep])
                self._buffer = self._buffer[len(self._buffer) - keep :]
                break
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(_THINK_OPEN) :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        """What is left when the stream ends. Text still inside an open <think> is dropped."""
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        return rest


def _tool_calls(message: object) -> tuple[ToolCall, ...]:
    """The tool requests in a reply, in order. Anything malformed is left out."""
    raw = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for item in raw:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):  # some servers send the arguments as JSON text
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        calls.append(ToolCall(function["name"], arguments if isinstance(arguments, dict) else {}))
    return tuple(calls)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Ollama's own error message, e.g. "model 'x' not found". Never contains the prompt."""
    try:
        payload = json.loads(exc.read(4096))
        message = payload.get("error") if isinstance(payload, dict) else None
    except (ValueError, OSError):
        message = None
    return f": {str(message)[:200]}" if message else ""
