from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_embedder, get_session, get_vector_store
from app.core.config import Settings, get_settings
from app.main import app
from app.storage.vector_store import QdrantVectorStore, collection_name
from tests.fakes import HashingEmbedder
from tests.helpers import add_and_index

NOTES = {
    "oats.md": "# Oats\n\nOats are high in fibre.\n\n## Cooking\n\nSimmer the oats in milk.\n",
    "rice.txt": "Rice needs twice its volume of water and eighteen minutes.",
    "travel.md": "# Lisbon trip\n\nThe train to Lisbon leaves from platform four.\n",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def indexed(
    session: Session, data_dir: Path, make_settings: Callable[..., Settings]
) -> Iterator[None]:
    """Serve the API from a small indexed corpus with a fake embedder."""
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("test", embedder.dimension)
    add_and_index(session, data_dir, embedder, store, NOTES)
    app.dependency_overrides[get_settings] = lambda: make_settings(retrieval_top_k=2)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    yield
    store.close()


def test_search_returns_ranked_results_with_sources(client: TestClient, indexed: None) -> None:
    response = client.post("/api/v1/search", json={"query": "simmer oats in milk"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2  # RETRIEVAL_TOP_K from settings
    assert results[0]["source"] == "oats.md"
    assert results[0]["heading_path"] == "Oats > Cooking"
    assert set(results[0]) == {
        "citation_id",
        "document_id",
        "chunk_index",
        "score",
        "source",
        "heading_path",
        "start_line",
        "end_line",
        "text",
    }


def test_top_k_and_filters_in_the_request(client: TestClient, indexed: None) -> None:
    response = client.post(
        "/api/v1/search", json={"query": "water", "top_k": 5, "file_types": [".txt"]}
    )

    assert response.status_code == 200
    assert [r["source"] for r in response.json()["results"]] == ["rice.txt"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "oats", "top_k": 0},
        {"query": "oats", "top_k": 51},
        {"query": "oats", "file_types": ["x" * 17]},
        {},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, indexed: None, body: dict) -> None:
    assert client.post("/api/v1/search", json=body).status_code == 422


def test_missing_model_gives_a_clear_503(
    client: TestClient, session: Session, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings()  # no model in models_dir
    app.dependency_overrides[get_session] = lambda: session

    response = client.post("/api/v1/search", json={"query": "oats"})

    assert response.status_code == 503
    assert "download-model" in response.json()["detail"]


def test_unmigrated_database_gives_a_clear_503(
    client: TestClient, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings()

    response = client.post("/api/v1/search", json={"query": "oats"})

    assert response.status_code == 503
    assert "alembic upgrade head" in response.json()["detail"]


def test_busy_vector_store_gives_a_clear_503(
    client: TestClient,
    session: Session,
    data_dir: Path,
    make_settings: Callable[..., Settings],
) -> None:
    embedder = HashingEmbedder()
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: embedder

    with QdrantVectorStore.open_local(
        data_dir / "qdrant", collection_name(embedder.model_name), embedder.dimension
    ):
        response = client.post("/api/v1/search", json={"query": "oats"})

    assert response.status_code == 503
    assert "in use by another process" in response.json()["detail"]


def test_search_is_listed_in_the_api_docs(client: TestClient) -> None:
    assert "/api/v1/search" in client.get("/openapi.json").json()["paths"]
