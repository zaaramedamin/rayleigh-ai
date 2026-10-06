"""OllamaProvider.stream against a scripted local server that can pause between lines."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from app.ai.llm.base import LLMError, LLMModelNotFoundError, LLMTimeoutError, LLMUnavailableError
from app.ai.llm.ollama import OllamaProvider, _ThinkFilter

Line = dict[str, Any] | bytes | threading.Event | float  # a JSON line, raw bytes, a gate, a pause


def piece(text: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": text}, "done": False}


DONE: dict[str, Any] = {"message": {"role": "assistant", "content": ""}, "done": True}


class StreamingOllama:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.lines: list[Line] = [piece("ok"), DONE]
        # What to answer before streaming: (status, body), per request, last one repeats.
        self.refusals: list[tuple[int, dict[str, Any]]] = []
        self.client_left = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(length)))
                if outer.refusals:
                    status, body = outer.refusals.pop(0)
                    raw = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    for line in outer.lines:
                        if isinstance(line, threading.Event):
                            line.wait(10)
                        elif isinstance(line, float):
                            time.sleep(line)
                        else:
                            raw = line if isinstance(line, bytes) else json.dumps(line).encode()
                            self.wfile.write(raw + b"\n")
                            self.wfile.flush()
                except OSError:
                    outer.client_left.set()

            def log_message(self, *_args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, *_args: object) -> None:
                outer.client_left.set()

        self._server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server() -> Iterator[StreamingOllama]:
    fake = StreamingOllama()
    yield fake
    fake.close()


def provider(server: StreamingOllama, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(server.url, "test-model:1b", **{"timeout_seconds": 5, **kwargs})


def test_the_pieces_come_back_in_order_and_the_request_asks_for_a_stream(
    server: StreamingOllama,
) -> None:
    server.lines = [piece("Sim"), piece("mer the "), piece("oats."), DONE]

    pieces = list(provider(server).stream("Be brief.", "How do I cook oats?"))

    assert pieces == ["Sim", "mer the ", "oats."]
    body = server.requests[0]
    assert body["stream"] is True
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "How do I cook oats?"},
    ]


def test_the_first_piece_arrives_while_the_model_is_still_writing(server: StreamingOllama) -> None:
    gate = threading.Event()
    server.lines = [piece("First "), gate, piece("second."), DONE]
    tokens = provider(server).stream("s", "u")

    started = time.monotonic()
    first = next(tokens)
    waited = time.monotonic() - started

    assert first == "First "
    assert waited < 2  # the server is still blocked on the gate: nothing waited for the whole reply
    gate.set()
    assert list(tokens) == ["second."]


def test_closing_the_stream_early_stops_the_server_from_going_on(server: StreamingOllama) -> None:
    gate = threading.Event()
    server.lines = [piece("one "), gate, *[piece("x" * 2000)] * 400, DONE]
    tokens = provider(server).stream("s", "u")
    assert next(tokens) == "one "

    tokens.close()
    gate.set()

    assert server.client_left.wait(5), "the connection stayed open after the stream was closed"


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["<think>plan</think>The answer."], "The answer."),
        (["<thi", "nk>plan the ", "answer</th", "ink>The ans", "wer."], "The answer."),
        (["The answer.<think>after</think>"], "The answer."),
        (["a < b and <thing> c"], "a < b and <thing> c"),
        (["a <", "b"], "a <b"),
        (["end <thi"], "end <thi"),
    ],
)
def test_thinking_is_removed_even_when_a_tag_is_split_between_pieces(
    server: StreamingOllama, pieces: list[str], expected: str
) -> None:
    server.lines = [*(piece(text) for text in pieces), DONE]

    assert "".join(provider(server).stream("s", "u")) == expected


def test_the_filter_drops_text_left_inside_an_unclosed_think_block() -> None:
    filtering = _ThinkFilter()

    shown = filtering.feed("Visible <think>never closed") + filtering.flush()

    assert shown == "Visible "


# --- failures ---------------------------------------------------------------------------------


def test_a_model_that_is_not_installed_is_reported_on_the_first_piece(
    server: StreamingOllama,
) -> None:
    server.refusals = [(404, {"error": "model not found"})]
    tokens = provider(server).stream("s", "u")

    with pytest.raises(LLMModelNotFoundError, match="ollama pull test-model:1b"):
        next(tokens)


def test_a_server_that_is_not_running_is_reported() -> None:
    with socket.socket() as sock:  # find a port that nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dead = OllamaProvider(f"http://127.0.0.1:{port}", "m", timeout_seconds=5)

    with pytest.raises(LLMUnavailableError, match="not reachable"):
        next(dead.stream("s", "u"))


def test_a_model_without_thinking_support_is_asked_again_without_it(
    server: StreamingOllama,
) -> None:
    server.refusals = [(400, {"error": '"test-model:1b" does not support thinking'})]
    server.lines = [piece("fine"), DONE]

    assert list(provider(server).stream("s", "u")) == ["fine"]

    assert "think" in server.requests[0] and "think" not in server.requests[1]


def test_an_error_in_the_middle_of_the_stream_is_raised_without_repeating_its_text(
    server: StreamingOllama,
) -> None:
    server.lines = [piece("Start "), {"error": "secret detail from the model"}]
    tokens = provider(server).stream("s", "u")
    assert next(tokens) == "Start "

    with pytest.raises(LLMError) as raised:
        next(tokens)

    assert "secret detail" not in str(raised.value)


@pytest.mark.parametrize("bad", [b"not json", b"[1, 2]"])
def test_a_line_that_is_not_a_json_object_is_refused(server: StreamingOllama, bad: bytes) -> None:
    server.lines = [bad]

    with pytest.raises(LLMError):
        list(provider(server).stream("s", "u"))


def test_a_stream_with_no_text_is_an_error(server: StreamingOllama) -> None:
    server.lines = [DONE]

    with pytest.raises(LLMError, match="empty answer"):
        list(provider(server).stream("s", "u"))


def test_a_model_that_stops_talking_times_out(server: StreamingOllama) -> None:
    server.lines = [piece("Start "), 3.0, piece("never")]
    tokens = provider(server, timeout_seconds=0.5).stream("s", "u")
    assert next(tokens) == "Start "

    with pytest.raises(LLMTimeoutError):
        next(tokens)


def test_nothing_of_the_prompt_or_the_answer_is_logged(
    server: StreamingOllama, caplog: pytest.LogCaptureFixture
) -> None:
    server.lines = [piece("the garage code is 4821"), DONE]

    with caplog.at_level("DEBUG"):
        list(provider(server).stream("system secret 1", "question secret 2"))

    assert "4821" not in caplog.text
    assert "secret" not in caplog.text
    assert "llm streamed" in caplog.text


def test_the_provider_still_answers_in_one_piece_as_before(server: StreamingOllama) -> None:
    server.refusals = [(200, {"message": {"content": " whole answer "}, "done": True})]

    assert provider(server).generate("s", "u") == "whole answer"
    assert server.requests[0]["stream"] is False
