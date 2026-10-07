"""The /tasks API: summarize, compare, extract."""

import json
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
from app.api.access import require_access
from app.api.deps import get_embedder, get_llm, get_session
from app.api.v1 import tasks as tasks_module
from app.core.config import Settings, get_settings
from app.main import app
from app.storage.models import DOC_SUPERSEDED, Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import FakeLLM, HashingEmbedder
from tests.helpers import add_and_index

URL = "/api/v1/tasks"
NOTES = {
    "oats.md": (
        "# Oats\n\nOats are high in fibre.\n\n"
        "## Cooking\n\nSimmer the oats in milk for five minutes.\n"
    ),
    "rice.txt": "Rice needs twice its volume of water and eighteen minutes.",
    "invoices.md": "Invoice INV-1 totals 100 euros.\n\nInvoice INV-2 totals 250 euros.\n",
}


class Harness:
    def __init__(self) -> None:
        self.llm = FakeLLM("A short summary.")
        self.documents: dict[str, Document] = {}


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
    harness.documents = add_and_index(session, data_dir, embedder, store, NOTES)

    @contextmanager
    def fake_locked_store(*_args: object) -> Iterator[QdrantVectorStore]:
        yield store

    monkeypatch.setattr(tasks_module, "locked_vector_store", fake_locked_store)
    app.dependency_overrides[get_settings] = lambda: make_settings(
        retrieval_top_k=3, answer_min_score=0.3
    )
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: harness.llm
    yield harness
    store.close()


def doc(harness: Harness, name: str) -> int:
    return harness.documents[name].id


ERRORS = [
    (LLMTimeoutError("too slow"), 504),
    (LLMUnavailableError("Ollama is not reachable"), 503),
    (LLMModelNotFoundError("ollama pull x"), 503),
    (LLMError("odd response"), 502),
]


# --- summarize -------------------------------------------------------------------------------


def test_a_document_is_summarized(client: TestClient, harness: Harness) -> None:
    response = client.post(f"{URL}/summarize", json={"document_id": doc(harness, "oats.md")})

    assert response.status_code == 200
    assert response.json() == {
        "text": "A short summary.",
        "document_id": doc(harness, "oats.md"),
        "name": "oats.md",
        "parts": 1,
        "covered_parts": 1,
        "truncated": False,
    }
    assert "Simmer the oats in milk" in harness.llm.calls[0][1]  # the document's own text


def test_a_document_that_does_not_exist_is_a_404(client: TestClient, harness: Harness) -> None:
    response = client.post(f"{URL}/summarize", json={"document_id": 999})

    assert response.status_code == 404 and "not in the library" in response.json()["detail"]
    assert harness.llm.calls == []


def test_an_older_version_is_refused_with_a_reason(
    client: TestClient, harness: Harness, session: Session
) -> None:
    harness.documents["rice.txt"].status = DOC_SUPERSEDED
    session.commit()

    response = client.post(f"{URL}/summarize", json={"document_id": doc(harness, "rice.txt")})

    assert response.status_code == 422 and "not current" in response.json()["detail"]


def test_a_document_the_model_finds_nothing_in_is_a_422(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "INSUFFICIENT"

    response = client.post(f"{URL}/summarize", json={"document_id": doc(harness, "oats.md")})

    assert response.status_code == 422 and "no readable content" in response.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"document_id": "one"}, {"document_id": None}])
def test_a_bad_summarize_request_is_refused(
    client: TestClient, harness: Harness, body: dict
) -> None:
    assert client.post(f"{URL}/summarize", json=body).status_code == 422
    assert harness.llm.calls == []


@pytest.mark.parametrize(("error", "status_code"), ERRORS)
def test_summarize_maps_model_failures_like_ask(
    client: TestClient, harness: Harness, error: Exception, status_code: int
) -> None:
    harness.llm.error = error

    response = client.post(f"{URL}/summarize", json={"document_id": doc(harness, "oats.md")})

    assert response.status_code == status_code and str(error) in response.json()["detail"]


# --- compare ---------------------------------------------------------------------------------


def test_two_documents_are_compared_and_the_cited_ones_are_listed(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "Oats take five minutes [1]; rice takes eighteen [2]."
    ids = [doc(harness, "oats.md"), doc(harness, "rice.txt")]

    body = client.post(f"{URL}/compare", json={"document_ids": ids}).json()

    assert body["grounded"] is True and body["reason"] == "compared"
    assert body["text"] == "Oats take five minutes [1]; rice takes eighteen [2]."
    assert [(d["marker"], d["name"]) for d in body["documents"]] == [
        (1, "oats.md"),
        (2, "rice.txt"),
    ]
    assert [(d["marker"], d["document_id"]) for d in body["sources"]] == [(1, ids[0]), (2, ids[1])]
    assert body["truncated_documents"] == []


def test_a_comparison_with_no_valid_citation_is_withheld_but_still_answered(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "They differ in cooking time."
    ids = [doc(harness, "oats.md"), doc(harness, "rice.txt")]

    response = client.post(f"{URL}/compare", json={"document_ids": ids})

    body = response.json()
    assert response.status_code == 200
    assert (body["grounded"], body["reason"], body["sources"]) == (False, "no_valid_citation", [])
    assert "cooking time" not in body["text"] and len(body["documents"]) == 2


def test_a_missing_document_in_a_comparison_is_a_404(client: TestClient, harness: Harness) -> None:
    response = client.post(f"{URL}/compare", json={"document_ids": [doc(harness, "oats.md"), 999]})

    assert response.status_code == 404 and harness.llm.calls == []


@pytest.mark.parametrize("count", [0, 1, 5])
def test_only_two_to_four_documents_may_be_compared(
    client: TestClient, harness: Harness, count: int
) -> None:
    ids = list(range(1, count + 1))

    assert client.post(f"{URL}/compare", json={"document_ids": ids}).status_code == 422
    assert harness.llm.calls == []


def test_a_document_cannot_be_compared_with_itself(client: TestClient, harness: Harness) -> None:
    same = doc(harness, "oats.md")

    response = client.post(f"{URL}/compare", json={"document_ids": [same, same]})

    assert response.status_code == 422 and "with itself" in response.json()["detail"]


@pytest.mark.parametrize(("error", "status_code"), ERRORS)
def test_compare_maps_model_failures_like_ask(
    client: TestClient, harness: Harness, error: Exception, status_code: int
) -> None:
    harness.llm.error = error
    ids = [doc(harness, "oats.md"), doc(harness, "rice.txt")]

    assert client.post(f"{URL}/compare", json={"document_ids": ids}).status_code == status_code


# --- extract ---------------------------------------------------------------------------------


def test_facts_are_extracted_into_rows_with_their_sources(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = json.dumps(
        [
            {"item": "INV-1", "value": "100 euros", "note": 1},
            {"item": "INV-2", "value": "250 euros", "note": 1},
        ]
    )

    body = client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"}).json()

    assert body["reason"] == "extracted" and body["dropped"] == 0
    assert body["rows"] == [
        {"item": "INV-1", "value": "100 euros", "marker": 1},
        {"item": "INV-2", "value": "250 euros", "marker": 1},
    ]
    (source,) = body["sources"]
    assert source["source"] == "invoices.md" and source["marker"] == 1
    assert {"citation_id", "document_id", "start_line", "score", "text"} <= set(source)
    assert body["notes_considered"] >= 1


def test_rows_for_notes_that_do_not_exist_are_refused_and_counted(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = json.dumps(
        [{"item": "a", "value": "b", "note": 1}, {"item": "c", "value": "d", "note": 9}]
    )

    body = client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"}).json()

    assert [r["item"] for r in body["rows"]] == ["a"] and body["dropped"] == 1


def test_a_request_no_note_is_relevant_to_never_reaches_the_model(
    client: TestClient, harness: Harness
) -> None:
    body = client.post(f"{URL}/extract", json={"request": "What is the capital of France?"}).json()

    assert body["reason"] == "no_relevant_notes" and body["rows"] == [] and body["sources"] == []
    assert harness.llm.calls == []


def test_the_model_answering_with_no_facts_is_nothing_found_not_an_error(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "[]"

    response = client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"})

    assert response.status_code == 200 and response.json()["reason"] == "nothing_found"


def test_a_reply_that_is_not_a_table_is_reported_as_unreadable(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "I found two invoices."

    body = client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"}).json()

    assert body["reason"] == "unreadable" and body["rows"] == []


def test_the_search_can_be_limited_to_some_documents(client: TestClient, harness: Harness) -> None:
    harness.llm.reply = "[]"

    client.post(
        f"{URL}/extract",
        json={"request": "water minutes", "document_ids": [doc(harness, "rice.txt")], "top_k": 1},
    )

    assert (
        "Source: rice.txt" in harness.llm.calls[0][1] and "oats.md" not in harness.llm.calls[0][1]
    )


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"request": ""},
        {"request": "   "},
        {"request": "x" * 2001},
        {"request": "a", "top_k": 0},
        {"request": "a", "mode": "x"},
    ],
)
def test_a_bad_extract_request_is_refused(client: TestClient, harness: Harness, body: dict) -> None:
    assert client.post(f"{URL}/extract", json=body).status_code == 422
    assert harness.llm.calls == []


@pytest.mark.parametrize(("error", "status_code"), ERRORS)
def test_extract_maps_model_failures_like_ask(
    client: TestClient, harness: Harness, error: Exception, status_code: int
) -> None:
    harness.llm.error = error

    response = client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"})

    assert response.status_code == status_code


def test_the_search_index_is_released_after_an_extraction(
    client: TestClient, harness: Harness
) -> None:
    harness.llm.reply = "[]"

    client.post(f"{URL}/extract", json={"request": "Invoice INV totals euros"})

    assert not deps._STORE_LOCK.locked()


# --- all three -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("summarize", {"document_id": 1}),
        ("compare", {"document_ids": [1, 2]}),
        ("extract", {"request": "anything"}),
    ],
)
def test_every_task_needs_the_access_password(
    harness: Harness, make_settings: Callable[..., Settings], route: str, body: dict
) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=True)
    app.dependency_overrides.pop(require_access, None)  # the real check, as in production

    assert TestClient(app).post(f"{URL}/{route}", json=body).status_code == 401
    assert harness.llm.calls == []


def test_the_task_routes_are_listed_in_the_api_docs(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {f"{URL}/summarize", f"{URL}/compare", f"{URL}/extract"} <= set(paths)
