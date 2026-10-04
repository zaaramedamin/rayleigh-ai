"""OllamaProvider against a scripted local HTTP server, so the tests need no real Ollama."""

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from app.ai.llm.base import (
    ChatMessage,
    ChatReply,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.ai.llm.ollama import OllamaProvider

Reply = tuple[int, Any]  # (HTTP status, JSON-able body or raw bytes)


def _chat(content: str, **message_extra: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": content, **message_extra}, "done": True}


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []  # method, path, body
        self.respond: Callable[[str, dict[str, Any] | None], Reply] = lambda _p, _b: (
            200,
            _chat("ok"),
        )
        self.delay = 0.0
        self.extra_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                outer.requests.append((self.command, self.path, body))
                time.sleep(outer.delay)
                status, payload = outer.respond(self.path, body)
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for name, value in outer.extra_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _handle  # noqa: N815
            do_POST = _handle  # noqa: N815

            def log_message(self, *_args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, *_args: object) -> None:
                pass  # the client gave up first (timeout tests)

        self._server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def ollama() -> Iterator[FakeOllama]:
    server = FakeOllama()
    yield server
    server.close()


def _provider(server: FakeOllama, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(server.url, "test-model:1b", **{"timeout_seconds": 5, **kwargs})


def test_generate_returns_the_reply_and_sends_a_deterministic_chat_request(
    ollama: FakeOllama,
) -> None:
    ollama.respond = lambda _p, _b: (200, _chat("  Bananas are yellow.  "))

    reply = _provider(ollama).generate("Be brief.", "What colour is a banana?")

    assert reply == "Bananas are yellow."
    method, path, body = ollama.requests[0]
    assert (method, path) == ("POST", "/api/chat")
    assert body is not None
    assert body["model"] == "test-model:1b"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What colour is a banana?"},
    ]
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0


def test_chat_sends_the_whole_conversation_at_the_requested_temperature(
    ollama: FakeOllama,
) -> None:
    ollama.respond = lambda _p, _b: (200, {**_chat(" In France. "), "done_reason": "stop"})
    conversation = [
        ChatMessage("user", "Hello"),
        ChatMessage("assistant", "Hi."),
        ChatMessage("user", "Where is Paris?"),
    ]

    reply = _provider(ollama).chat("Be brief.", conversation, temperature=0.6)

    assert reply == ChatReply("In France.", truncated=False)
    method, path, body = ollama.requests[0]
    assert (method, path) == ("POST", "/api/chat")
    assert body is not None
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "Where is Paris?"},
    ]
    assert body["options"]["temperature"] == 0.6
    assert body["stream"] is False


def test_a_reply_cut_off_at_the_length_limit_is_flagged(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (200, {**_chat("The first half of"), "done_reason": "length"})
    provider = _provider(ollama)

    assert provider.chat("s", [ChatMessage("user", "u")]) == ChatReply(
        "The first half of", truncated=True
    )
    assert provider.generate("s", "u") == "The first half of"


def test_reasoning_text_is_never_part_of_the_answer(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (
        200,
        _chat("<think>inline reasoning</think>Yellow.", thinking="separate reasoning"),
    )

    assert _provider(ollama).generate("s", "u") == "Yellow."


def test_think_setting_is_passed_on(ollama: FakeOllama) -> None:
    _provider(ollama, think=True).generate("s", "u")

    assert ollama.requests[0][2]["think"] is True


def test_models_without_a_reasoning_mode_are_retried_without_the_think_field(
    ollama: FakeOllama,
) -> None:
    def respond(_path: str, body: dict[str, Any] | None) -> Reply:
        if body is not None and "think" in body:
            return 400, {"error": '"test-model:1b" does not support thinking'}
        return 200, _chat("fine")

    ollama.respond = respond

    assert _provider(ollama).generate("s", "u") == "fine"
    assert len(ollama.requests) == 2
    assert "think" not in ollama.requests[1][2]


def test_other_rejections_are_reported_and_not_retried(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (400, {"error": "invalid options"})

    with pytest.raises(LLMError, match="HTTP 400: invalid options"):
        _provider(ollama).generate("s", "u")
    assert len(ollama.requests) == 1


def test_server_not_running_gives_a_clear_error() -> None:
    with socket.socket() as sock:  # find a port that nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = OllamaProvider(f"http://127.0.0.1:{port}", "m", timeout_seconds=5)

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        provider.generate("s", "u")


def test_missing_model_says_how_to_install_it(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (404, {"error": "model 'test-model:1b' not found"})

    with pytest.raises(LLMModelNotFoundError, match="ollama pull test-model:1b"):
        _provider(ollama).generate("s", "u")


def test_slow_model_times_out_with_a_clear_error(ollama: FakeOllama) -> None:
    ollama.delay = 1.5

    with pytest.raises(LLMTimeoutError, match="LLM_TIMEOUT_SECONDS"):
        _provider(ollama, timeout_seconds=0.3).generate("s", "u")


def test_server_errors_mean_not_ready(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (500, {"error": "loading"})

    with pytest.raises(LLMUnavailableError, match="not ready"):
        _provider(ollama).generate("s", "u")


@pytest.mark.parametrize(
    "payload",
    [b"this is not json", [1, 2, 3], {"done": True}, {"message": {"role": "assistant"}}],
)
def test_malformed_responses_are_errors(ollama: FakeOllama, payload: Any) -> None:
    ollama.respond = lambda _p, _b: (200, payload)

    with pytest.raises(LLMError):
        _provider(ollama).generate("s", "u")


@pytest.mark.parametrize("content", ["", "   ", "<think>only reasoning</think>"])
def test_empty_answers_are_errors(ollama: FakeOllama, content: str) -> None:
    ollama.respond = lambda _p, _b: (200, _chat(content))

    with pytest.raises(LLMError, match="empty"):
        _provider(ollama).generate("s", "u")


def test_running_out_of_budget_while_reasoning_is_explained(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (
        200,
        {**_chat("", thinking="long reasoning ..."), "done_reason": "length"},
    )

    with pytest.raises(LLMError, match="LLM_THINK=false"):
        _provider(ollama, think=True).generate("s", "u")


def test_running_out_of_budget_without_reasoning_is_reported_plainly(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (200, {**_chat(""), "done_reason": "length"})

    with pytest.raises(LLMError, match="answer budget") as raised:
        _provider(ollama, think=False).generate("s", "u")
    assert "LLM_THINK" not in str(raised.value)


def test_redirects_are_refused_so_the_prompt_goes_nowhere_else(ollama: FakeOllama) -> None:
    elsewhere = FakeOllama()
    try:
        ollama.respond = lambda _p, _b: (307, {})
        ollama.extra_headers = {"Location": f"{elsewhere.url}/api/chat"}

        with pytest.raises(LLMError, match="redirect"):
            _provider(ollama).generate("s", "secret question")

        assert elsewhere.requests == []
    finally:
        elsewhere.close()


def test_system_proxy_settings_are_ignored(
    ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # a proxy that does not exist
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    assert _provider(ollama).generate("s", "u") == "ok"


def test_list_models(ollama: FakeOllama) -> None:
    ollama.respond = lambda _p, _b: (
        200,
        {"models": [{"name": "qwen3.5:4b"}, {"name": "llama3.2:latest"}]},
    )

    assert _provider(ollama).list_models() == ["qwen3.5:4b", "llama3.2:latest"]
    assert ollama.requests[0][:2] == ("GET", "/api/tags")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://192.168.1.20:11434",
        "http://localhost.evil.com:11434",
        "http://user:pw@localhost:11434",
        "ftp://localhost:11434",
        "http://localhost:11434/v1?x=1",
        "localhost:11434",
    ],
)
def test_only_this_machine_is_accepted_as_a_server(url: str) -> None:
    with pytest.raises(ValueError, match="this machine"):
        OllamaProvider(url, "m")
