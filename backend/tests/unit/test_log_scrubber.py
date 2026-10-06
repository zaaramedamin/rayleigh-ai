"""Nothing the owner wrote, asked or was answered ends up in a log.

The rule (docs/security.md): logs say what happened in counts and kinds, never in words. This test
puts a distinctive marker into everything that is the owner's own text, drives the main routes and
services with it, and fails if any marker appears in anything the application logged. It protects
the rule as the code grows: a new `logger.info(f"... {question}")` fails here.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_embedder, get_llm, get_session, get_vector_store
from app.api.v1 import ask as ask_module
from app.core.config import Settings, get_settings
from app.knowledge.indexing.service import index_pending
from app.knowledge.ingestion.service import ingest_folders
from app.main import app
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import FakeLLM, HashingEmbedder

MARKERS = {
    "note text": "Quillon-Marmalade-4821",
    "file name": "diary-Zeppelin-7733",
    "folder name": "Folder-Narwhal-5512",
    "question": "Walrus-Basement-9150",
    "earlier turn": "Heron-Lantern-3067",
    "rewritten question": "Ibex-Compass-2294",
    "answer": "Sentinel-Parrot-3318",
    "chat message": "Cobalt-Meadow-6410",
    "chat reply": "Saffron-Glacier-8825",
    "conversation title": "Maple-Orbit-1176",
    "conversation text": "Falcon-Ledger-9043",
    "profile": "Tundra-Violet-5589",
    "memory": "Pebble-Harbor-7102",
    "error message": "Oracle-Quartz-4417",
}


@pytest.fixture
def library(
    session: Session, data_dir: Path, tmp_path: Path, make_settings: Callable[..., Settings]
) -> Iterator[FakeLLM]:
    """A small library of notes that hold markers, behind the real routes, with a scripted model."""
    folder = tmp_path / MARKERS["folder name"]
    folder.mkdir()
    name = MARKERS["file name"]
    (folder / f"{name}.md").write_text(
        f"# Garage\n\nThe garage code is {MARKERS['note text']} and oats need five minutes.\n"
    )
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("test", embedder.dimension)
    ingest_folders(session, data_dir, [folder], 100_000)
    index_pending(session, embedder, store)
    llm = FakeLLM(f"The garage code is {MARKERS['answer']} [1].")

    @contextmanager
    def locked(*_args: object) -> Iterator[QdrantVectorStore]:
        yield store

    original = ask_module.locked_vector_store
    ask_module.locked_vector_store = locked  # type: ignore[assignment]
    app.dependency_overrides[get_settings] = lambda: make_settings(retrieval_top_k=3)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    yield llm
    ask_module.locked_vector_store = original  # type: ignore[assignment]
    app.dependency_overrides.clear()
    store.close()


def drive_everything(client: TestClient, llm: FakeLLM) -> None:
    """The main things a person does, each carrying their own words."""
    question = f"What is the garage code {MARKERS['question']} for {MARKERS['note text']}?"
    history = [
        {"role": "user", "content": f"Tell me about the garage {MARKERS['earlier turn']}"},
        {"role": "assistant", "content": f"It has a code {MARKERS['note text']} [1]."},
    ]
    client.post("/api/v1/search", json={"query": question, "mode": "hybrid"})
    client.post("/api/v1/ask", json={"question": question})
    # A follow-up: the model rewrites it, then answers.
    llm.script = [f"What is the garage code {MARKERS['rewritten question']}?"]
    client.post("/api/v1/ask", json={"question": "and its number?", "history": history})
    llm.script = [f"What is the garage code {MARKERS['rewritten question']}?"]
    client.post("/api/v1/ask/stream", json={"question": "and its number?", "history": history})
    client.post("/api/v1/ask/stream", json={"question": question})

    llm.reply = f"Of course: {MARKERS['chat reply']}."
    client.post(
        "/api/v1/chat", json={"message": f"Remember {MARKERS['chat message']}", "history": history}
    )
    llm.reply = f"The garage code is {MARKERS['answer']} [1]."

    created = client.post("/api/v1/conversations", json={"title": MARKERS["conversation title"]})
    conversation_id = created.json()["id"]
    client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "messages": [
                {"role": "user", "mode": "notes", "content": MARKERS["conversation text"]},
                {
                    "role": "assistant",
                    "mode": "notes",
                    "content": f"{MARKERS['answer']} [1]",
                    "payload": {"sources": [{"document_id": 1, "text": MARKERS["note text"]}]},
                },
            ]
        },
    )
    client.put(f"/api/v1/conversations/{conversation_id}", json={"title": MARKERS["profile"]})
    client.get(f"/api/v1/conversations/{conversation_id}")
    client.get("/api/v1/conversations")
    client.delete(f"/api/v1/conversations/{conversation_id}")

    client.put("/api/v1/profile", json={"name": MARKERS["profile"], "about": MARKERS["memory"]})
    client.post("/api/v1/assistant/memories", json={"text": MARKERS["memory"]})

    client.get("/api/v1/library/documents")
    client.get("/api/v1/library/documents/1")
    client.get("/api/v1/library/documents/1/chunks")
    client.get("/api/v1/system/status")
    client.get("/api/v1/system/settings")


def test_nothing_the_owner_wrote_is_in_any_log(
    library: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    client = TestClient(app)

    with caplog.at_level(logging.DEBUG):
        drive_everything(client, library)

    logged = caplog.text
    assert "search finished" in logged and "answer finished" in logged  # real logs were captured
    found = {kind: marker for kind, marker in MARKERS.items() if marker in logged}
    assert found == {}, f"these appear in the logs: {found}"


def test_an_error_does_not_log_what_it_said(
    library: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    library.stream_error_after = (1, RuntimeError(f"it said {MARKERS['error message']}"))
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.DEBUG):
        response = client.post("/api/v1/ask/stream", json={"question": "What is the garage code?"})

    assert MARKERS["error message"] not in response.text
    assert "streamed answer failed error=RuntimeError" in caplog.text
    assert MARKERS["error message"] not in caplog.text
    assert "Traceback" not in caplog.text


def test_an_unexpected_failure_in_a_route_does_not_log_what_it_said(
    library: FakeLLM, caplog: pytest.LogCaptureFixture
) -> None:
    @app.get("/api/v1/_scrubber_probe")
    def probe() -> None:
        raise RuntimeError(f"the question was {MARKERS['question']}")

    try:
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.DEBUG):
            response = client.get("/api/v1/_scrubber_probe")
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", "") != "/api/v1/_scrubber_probe"
        ]

    assert response.status_code == 500
    assert MARKERS["question"] not in caplog.text and MARKERS["question"] not in response.text


def test_the_server_does_not_log_each_request_with_its_query_string() -> None:
    """A request line holds the query string, which can be a folder path; `serve` turns them off."""
    import inspect

    from app import cli

    assert "access_log=False" in inspect.getsource(cli._cmd_serve)
