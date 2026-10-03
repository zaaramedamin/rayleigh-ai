from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.llm.base import (
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.api import deps
from app.api.deps import get_embedder, get_llm, get_session
from app.api.v1 import ask as ask_module
from app.core.config import Settings, get_settings
from app.main import app
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import FakeLLM, HashingEmbedder
from tests.helpers import add_and_index

NOTES = {
    "oats.md": "# Oats\n\nOats are high in fibre.\n\n## Cooking\n\nSimmer the oats in milk.\n",
    "rice.txt": "Rice needs twice its volume of water and eighteen minutes.",
}


class Harness:
    """A small indexed corpus plus a fake LLM, wired into the API."""

    def __init__(self) -> None:
        self.llm = FakeLLM("Simmer the oats in milk [1].")
        self.store_open = False
        self.store_was_open_during_llm: list[bool] = []


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def harness(
    session: Session,
    data_dir: Path,
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Harness]:
    harness = Harness()
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("test", embedder.dimension)
    add_and_index(session, data_dir, embedder, store, NOTES)

    @contextmanager
    def fake_locked_store(*_args: object) -> Iterator[QdrantVectorStore]:
        harness.store_open = True
        try:
            yield store
        finally:
            harness.store_open = False

    original_generate = harness.llm.generate

    def generate(system: str, user: str) -> str:
        harness.store_was_open_during_llm.append(harness.store_open)
        return original_generate(system, user)

    harness.llm.generate = generate  # type: ignore[method-assign]
    monkeypatch.setattr(ask_module, "locked_vector_store", fake_locked_store)
    app.dependency_overrides[get_settings] = lambda: make_settings(
        retrieval_top_k=3, answer_min_score=0.3
    )
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: harness.llm
    yield harness
    store.close()


def test_a_cited_answer_comes_back_with_sources(client: TestClient, harness: Harness) -> None:
    response = client.post("/api/v1/ask", json={"question": "How do I simmer oats in milk?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Simmer the oats in milk [1]."
    assert body["grounded"] is True
    assert body["reason"] == "answered"
    assert body["notes_considered"] >= 1
    source = body["sources"][0]
    assert source["marker"] == 1
    assert (source["source"], source["heading_path"]) == ("oats.md", "Oats > Cooking")
    assert source["text"].startswith("## Cooking")
    assert {"citation_id", "document_id", "start_line", "end_line", "score"} <= set(source)


def test_an_unrelated_question_gets_an_explicit_refusal_and_no_model_call(
    client: TestClient, harness: Harness
) -> None:
    response = client.post("/api/v1/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["reason"] == "no_relevant_notes"
    assert "don't have enough information" in body["answer"]
    assert body["sources"] == []
    assert harness.llm.calls == []


def test_the_model_declining_is_reported(client: TestClient, harness: Harness) -> None:
    harness.llm.reply = "INSUFFICIENT"

    body = client.post("/api/v1/ask", json={"question": "How long do I simmer oats?"}).json()

    assert (body["grounded"], body["reason"]) == (False, "model_declined")
    assert body["sources"] == []


def test_an_uncited_answer_is_not_returned(client: TestClient, harness: Harness) -> None:
    harness.llm.reply = "Oats take five minutes."

    body = client.post("/api/v1/ask", json={"question": "How long do I simmer oats?"}).json()

    assert (body["grounded"], body["reason"]) == (False, "no_valid_citation")
    assert "five minutes" not in body["answer"]


def test_the_search_index_is_not_held_while_the_model_thinks(
    client: TestClient, harness: Harness
) -> None:
    client.post("/api/v1/ask", json={"question": "How do I simmer oats in milk?"})

    assert harness.store_was_open_during_llm == [False]


def test_filters_and_top_k_are_honoured(client: TestClient, harness: Harness) -> None:
    body = client.post(
        "/api/v1/ask",
        json={"question": "water minutes", "file_types": [".txt"], "top_k": 1},
    ).json()

    assert body["notes_considered"] == 1
    assert "Source: rice.txt" in harness.llm.calls[0][1]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (LLMTimeoutError("too slow"), 504),
        (LLMUnavailableError("Ollama is not reachable"), 503),
        (LLMModelNotFoundError("ollama pull x"), 503),
        (LLMError("odd response"), 502),
    ],
)
def test_llm_failures_map_to_clear_http_errors(
    client: TestClient, harness: Harness, error: Exception, status_code: int
) -> None:
    harness.llm.error = error

    response = client.post("/api/v1/ask", json={"question": "How do I simmer oats in milk?"})

    assert response.status_code == status_code
    assert str(error) in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "oats", "top_k": 0},
        {"question": "oats", "top_k": 51},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, harness: Harness, body: dict) -> None:
    assert client.post("/api/v1/ask", json=body).status_code == 422
    assert harness.llm.calls == []


def test_the_store_lock_is_released_after_a_request(client: TestClient, harness: Harness) -> None:
    client.post("/api/v1/ask", json={"question": "How do I simmer oats in milk?"})

    assert not deps._STORE_LOCK.locked()


def test_ask_is_listed_in_the_api_docs(client: TestClient) -> None:
    assert "/api/v1/ask" in client.get("/openapi.json").json()["paths"]
