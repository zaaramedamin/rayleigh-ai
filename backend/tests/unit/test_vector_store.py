from collections.abc import Iterator
from pathlib import Path

import pytest

from app.storage.vector_store import (
    QdrantVectorStore,
    VectorPoint,
    VectorStoreBusyError,
    VectorStoreError,
    collection_name,
    count_local_vectors,
    normalize_file_type,
    point_id,
)

DIM = 4
X = [1.0, 0.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0, 0.0]
NEAR_X = [0.9, 0.1, 0.0, 0.0]


def _point(
    document_id: int,
    chunk_index: int,
    vector: list[float],
    *,
    file_type: str = ".md",
    text_hash: str = "hash",
) -> VectorPoint:
    return VectorPoint(
        document_id=document_id,
        chunk_index=chunk_index,
        vector=vector,
        text_hash=text_hash,
        file_type=file_type,
    )


@pytest.fixture
def store() -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore.in_memory("test", DIM)
    yield vector_store
    vector_store.close()


def _keys(hits: list) -> list[tuple[int, int]]:
    return [(hit.document_id, hit.chunk_index) for hit in hits]


def test_search_returns_most_similar_first(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X), _point(1, 1, Y), _point(2, 0, NEAR_X)])

    hits = store.search(X, top_k=3)

    assert _keys(hits) == [(1, 0), (2, 0), (1, 1)]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_top_k_limits_the_number_of_hits(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, i, X) for i in range(5)])

    assert len(store.search(X, top_k=2)) == 2


def test_hits_carry_the_stored_metadata(store: QdrantVectorStore) -> None:
    store.upsert([_point(7, 3, X, text_hash="abc123")])

    hit = store.search(X, top_k=1)[0]

    assert (hit.document_id, hit.chunk_index, hit.text_hash) == (7, 3, "abc123")


def test_upserting_the_same_chunk_replaces_it(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X, text_hash="old")])
    store.upsert([_point(1, 0, Y, text_hash="new")])

    assert store.count() == 1
    assert store.search(Y, top_k=1)[0].text_hash == "new"


def test_delete_document_removes_only_its_vectors(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X), _point(1, 1, Y), _point(2, 0, X)])

    store.delete_document(1)

    assert store.count() == 1
    assert store.count(document_id=1) == 0
    assert store.count(document_id=2) == 1


def test_filter_by_document_ids(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X), _point(2, 0, NEAR_X), _point(3, 0, Y)])

    hits = store.search(X, top_k=5, document_ids=[2, 3])

    assert {hit.document_id for hit in hits} == {2, 3}


@pytest.mark.parametrize("spelling", [".md", "md", "MD", " .Md "])
def test_filter_by_file_type_ignores_case_and_dot(store: QdrantVectorStore, spelling: str) -> None:
    store.upsert([_point(1, 0, X, file_type=".md"), _point(2, 0, X, file_type=".txt")])

    hits = store.search(X, top_k=5, file_types=[spelling])

    assert {hit.document_id for hit in hits} == {1}


def test_empty_filters_mean_no_filter(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X), _point(2, 0, Y)])

    assert len(store.search(X, top_k=5, document_ids=[], file_types=[])) == 2


def test_reset_removes_every_vector(store: QdrantVectorStore) -> None:
    store.upsert([_point(1, 0, X), _point(2, 0, Y)])

    store.reset()

    assert store.count() == 0
    store.upsert([_point(1, 0, X)])  # still usable afterwards
    assert store.count() == 1


def test_upserting_nothing_is_a_no_op(store: QdrantVectorStore) -> None:
    store.upsert([])

    assert store.count() == 0


def test_local_store_persists_between_opens(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    with QdrantVectorStore.open_local(path, "c", DIM) as first:
        first.upsert([_point(1, 0, X), _point(1, 1, Y)])

    with QdrantVectorStore.open_local(path, "c", DIM) as second:
        assert second.count() == 2
        assert _keys(second.search(X, top_k=1)) == [(1, 0)]
    assert count_local_vectors(path, "c") == 2


def test_reopening_with_another_dimension_is_refused_and_releases_the_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qdrant"
    with QdrantVectorStore.open_local(path, "c", DIM):
        pass

    with pytest.raises(VectorStoreError, match="--rebuild"):
        QdrantVectorStore.open_local(path, "c", DIM * 2)

    with QdrantVectorStore.open_local(path, "c", DIM) as reopened:
        assert reopened.count() == 0


def test_a_second_opener_gets_a_clear_busy_error(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    with QdrantVectorStore.open_local(path, "c", DIM):
        with pytest.raises(VectorStoreBusyError, match="another process"):
            QdrantVectorStore.open_local(path, "c", DIM)


def test_counting_a_missing_store_is_zero_and_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert count_local_vectors(missing, "c") == 0
    assert not missing.exists()


def test_each_model_gets_its_own_collection() -> None:
    assert (
        collection_name("sentence-transformers/all-MiniLM-L6-v2")
        == "chunks_sentence_transformers_all_minilm_l6_v2"
    )
    assert collection_name("org/model-a") != collection_name("org/model-b")


def test_point_ids_are_stable_per_chunk() -> None:
    assert point_id(1, 2) == point_id(1, 2)
    assert point_id(1, 2) != point_id(2, 1)


@pytest.mark.parametrize(("raw", "expected"), [(".md", ".md"), ("MD", ".md"), (" txt ", ".txt")])
def test_normalize_file_type(raw: str, expected: str) -> None:
    assert normalize_file_type(raw) == expected
