"""LLM provider for a local Ollama server (https://ollama.com), using only the standard library."""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

from app.ai.llm.base import (
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.config import validate_loopback_url

logger = logging.getLogger(__name__)

# Context window requested from Ollama, in tokens. The prompt builder keeps the notes under
# 6000 characters, so the system prompt can never be pushed out of the window.
DEFAULT_NUM_CTX = 8192
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


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

    # --- HTTP ---------------------------------------------------------------------------------

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
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
        except OSError as exc:  # connection reset while reading, etc.
            raise LLMUnavailableError(
                f"connection to Ollama at {self._base_url} failed ({type(exc).__name__})"
            ) from exc

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
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self._think,
            "options": {
                "temperature": 0,  # the same notes and question should give the same answer
                "num_ctx": self._num_ctx,
                "num_predict": 4096 if self._think else 1024,  # bound a runaway answer
            },
        }
        started = time.monotonic()
        try:
            data = self._request("/api/chat", body)
        except _RejectedError as exc:
            # Models without a reasoning mode can reject the `think` field; ask again without it.
            if "think" not in str(exc).lower():
                raise
            del body["think"]
            data = self._request("/api/chat", body)

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("Ollama returned a response without a message")
        content = _THINK_BLOCK.sub("", content).strip()
        if not content:
            raise LLMError("the model returned an empty answer")
        # Log timing and sizes only, never the prompt or the answer.
        logger.info(
            "llm finished model=%s seconds=%.1f answer_chars=%d",
            self.model_name,
            time.monotonic() - started,
            len(content),
        )
        return content

    def list_models(self) -> list[str]:
        """Names of the models installed in Ollama. Raises LLMUnavailableError if it is down."""
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list):
            raise LLMError("Ollama returned an unexpected model list")
        return [str(m["name"]) for m in models if isinstance(m, dict) and "name" in m]


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Ollama's own error message, e.g. "model 'x' not found". Never contains the prompt."""
    try:
        payload = json.loads(exc.read(4096))
        message = payload.get("error") if isinstance(payload, dict) else None
    except (ValueError, OSError):
        message = None
    return f": {str(message)[:200]}" if message else ""
