from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.retrieval.service import MAX_QUERY_CHARS, retrieve
from app.storage.models import Chunk, Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder
from tests.helpers import add_and_index

NOTES = {
    "oats.md": "# Oats\n\nOats are high in fibre.\n\n## Cooking\n\nSimmer the oats in milk.\n",
    "rice.txt": "Rice needs twice its volume of water and eighteen minutes.",
    "travel.md": "# Lisbon trip\n\nThe train to Lisbon leaves from platform four.\n",
}


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder: HashingEmbedder) -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore.in_memory("test", embedder.dimension)
    yield vector_store
    vector_store.close()


@pytest.fixture
def documents(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> dict[str, Document]:
    return add_and_index(session, data_dir, embedder, store, NOTES)


def test_known_query_returns_the_expected_chunk_first(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    results = retrieve(session, embedder, store, "simmer oats in milk", top_k=3)

    assert results[0].source == "oats.md"
    assert results[0].heading_path == "Oats > Cooking"


def test_results_carry_provenance_from_the_database(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    result = retrieve(session, embedder, store, "train to Lisbon platform", top_k=1)[0]

    document = documents["travel.md"]
    chunk = session.scalars(
        select(Chunk).where(Chunk.document_id == document.id, Chunk.chunk_index == 0)
    ).one()
    assert result.source == "travel.md"
    assert result.citation_id == f"{document.id}:0"
    assert result.text == chunk.text
    assert (result.start_line, result.end_line) == (chunk.start_line, chunk.end_line)
    assert result.heading_path == "Lisbon trip"
    assert 0 < result.score <= 1.0001


def test_results_are_best_first_and_limited_by_top_k(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    results = retrieve(session, embedder, store, "oats rice train", top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_filter_by_file_type(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    results = retrieve(session, embedder, store, "water", top_k=10, file_types=["md"])

    assert results
    assert all(r.source.endswith(".md") for r in results)


def test_filter_by_document(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    rice = documents["rice.txt"]

    results = retrieve(session, embedder, store, "oats", top_k=10, document_ids=[rice.id])

    assert [r.source for r in results] == ["rice.txt"]


def test_min_score_drops_weak_matches(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    everything = retrieve(session, embedder, store, "oats", top_k=10)
    strong_only = retrieve(session, embedder, store, "oats", top_k=10, min_score=0.3)

    assert len(strong_only) < len(everything)
    assert all(r.score >= 0.3 for r in strong_only)


def test_changed_chunk_text_is_never_returned_with_an_old_vector(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    rice_chunk = session.scalars(
        select(Chunk).where(Chunk.document_id == documents["rice.txt"].id)
    ).one()
    rice_chunk.text = "something else entirely"  # changed without re-indexing
    session.commit()

    results = retrieve(session, embedder, store, "rice water minutes", top_k=10)

    assert all(r.source != "rice.txt" for r in results)


def test_vectors_of_deleted_documents_are_ignored(
    session: Session,
    embedder: HashingEmbedder,
    store: QdrantVectorStore,
    documents: dict[str, Document],
) -> None:
    session.delete(documents["rice.txt"])  # vectors stay in the store
    session.commit()

    results = retrieve(session, embedder, store, "rice water minutes", top_k=10)

    assert results
    assert all(r.source != "rice.txt" for r in results)


def test_an_empty_index_returns_no_results(
    session: Session, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    assert retrieve(session, embedder, store, "anything", top_k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_queries_are_rejected(
    session: Session, embedder: HashingEmbedder, store: QdrantVectorStore, query: str
) -> None:
    with pytest.raises(ValueError, match="empty"):
        retrieve(session, embedder, store, query, top_k=5)


def test_overlong_queries_are_rejected(
    session: Session, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    with pytest.raises(ValueError, match="longer than"):
        retrieve(session, embedder, store, "x" * (MAX_QUERY_CHARS + 1), top_k=5)


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_out_of_range_top_k_is_rejected(
    session: Session, embedder: HashingEmbedder, store: QdrantVectorStore, top_k: int
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve(session, embedder, store, "oats", top_k=top_k)
