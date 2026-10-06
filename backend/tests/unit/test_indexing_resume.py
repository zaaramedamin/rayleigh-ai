"""Slice 2.2: indexing that shows progress and continues where an interrupted run stopped."""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.knowledge.chunking.service import chunk_document
from app.knowledge.indexing.service import (
    EMBED_BATCH_SIZE,
    IndexProgress,
    count_pending,
    index_pending,
    reset_index,
)
from app.knowledge.ingestion.service import ingest_folders
from app.storage.files import save_file
from app.storage.models import Chunk, Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

MODEL = "test/model-a"


class Interrupted(RuntimeError):
    """Stands in for a crash, a closed window or a model error in the middle of a run."""


class FlakyEmbedder(HashingEmbedder):
    """Fails on one chosen call to `embed_documents`, then behaves again."""

    def __init__(self, fail_on_call: int | None, model_name: str = MODEL) -> None:
        super().__init__(model_name=model_name)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise Interrupted("the embedding model stopped")
        return super().embed_documents(texts)


def long_note(paragraphs: int = 5 * EMBED_BATCH_SIZE) -> str:
    """Many short paragraphs, so that small chunks make many of them."""
    return "\n\n".join(
        f"Paragraph {number} talks about subject{number} only." for number in range(paragraphs)
    )


@pytest.fixture
def big_document(session: Session, data_dir: Path) -> Document:
    document = save_file(session, data_dir, long_note().encode(), "big.txt")
    chunk_document(session, data_dir, document, 60, 0)
    return document


def chunk_total(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Chunk)) or 0


def indexed_chunks(session: Session, model: str = MODEL) -> int:
    return (
        session.scalar(select(func.count()).select_from(Chunk).where(Chunk.indexed_model == model))
        or 0
    )


def test_the_test_document_really_needs_several_batches(
    session: Session, big_document: Document
) -> None:
    assert chunk_total(session) > 3 * EMBED_BATCH_SIZE


def test_an_interrupted_run_continues_at_the_first_chunk_without_a_vector(
    session: Session, big_document: Document
) -> None:
    store = QdrantVectorStore.in_memory("resume", HashingEmbedder().dimension)
    total = chunk_total(session)
    flaky = FlakyEmbedder(fail_on_call=3)  # two batches are saved, the third one fails

    with pytest.raises(Interrupted):
        index_pending(session, flaky, store)

    session.expire_all()
    document = session.get_one(Document, big_document.id)
    assert document.indexed_model is None  # not complete, so still pending
    assert count_pending(session, MODEL) == 1
    assert indexed_chunks(session) == 2 * EMBED_BATCH_SIZE
    assert store.count(document.id) == 2 * EMBED_BATCH_SIZE  # the saved work is really there

    healthy = HashingEmbedder(model_name=MODEL)
    summary = index_pending(session, healthy, store)

    assert summary.chunks_embedded == total - 2 * EMBED_BATCH_SIZE  # only what was left
    assert healthy.texts_embedded == total - 2 * EMBED_BATCH_SIZE
    session.expire_all()
    assert session.get_one(Document, document.id).indexed_model == MODEL
    assert store.count(document.id) == total  # one vector per chunk: nothing duplicated
    assert indexed_chunks(session) == total
    store.close()


def test_a_resumed_index_is_identical_to_an_uninterrupted_one(
    session: Session, big_document: Document
) -> None:
    dimension = HashingEmbedder().dimension
    interrupted = QdrantVectorStore.in_memory("interrupted", dimension)
    straight = QdrantVectorStore.in_memory("straight", dimension)
    with pytest.raises(Interrupted):
        index_pending(session, FlakyEmbedder(fail_on_call=2), interrupted)
    index_pending(session, HashingEmbedder(model_name=MODEL), interrupted)
    reset_index(session, straight)
    index_pending(session, HashingEmbedder(model_name=MODEL), straight)

    query = HashingEmbedder().embed_query("subject7 paragraph")
    resumed_hits = [
        (h.document_id, h.chunk_index, h.text_hash) for h in interrupted.search(query, top_k=20)
    ]
    straight_hits = [
        (h.document_id, h.chunk_index, h.text_hash) for h in straight.search(query, top_k=20)
    ]

    assert interrupted.count() == straight.count()
    assert resumed_hits == straight_hits
    interrupted.close()
    straight.close()


def test_a_failure_before_anything_was_saved_leaves_the_document_pending(
    session: Session, big_document: Document
) -> None:
    store = QdrantVectorStore.in_memory("early", HashingEmbedder().dimension)

    with pytest.raises(Interrupted):
        index_pending(session, FlakyEmbedder(fail_on_call=1), store)

    assert indexed_chunks(session) == 0
    assert store.count() == 0
    assert count_pending(session, MODEL) == 1
    store.close()


def test_a_run_that_is_interrupted_twice_still_finishes(
    session: Session, big_document: Document
) -> None:
    store = QdrantVectorStore.in_memory("twice", HashingEmbedder().dimension)
    for _ in range(2):
        with pytest.raises(Interrupted):
            index_pending(session, FlakyEmbedder(fail_on_call=2), store)

    index_pending(session, HashingEmbedder(model_name=MODEL), store)

    assert store.count() == chunk_total(session)
    assert count_pending(session, MODEL) == 0
    store.close()


def test_progress_only_moves_forward_and_ends_complete(
    session: Session, data_dir: Path, big_document: Document
) -> None:
    save_file(session, data_dir, b"A second, small note.", "small.txt")
    store = QdrantVectorStore.in_memory("progress", HashingEmbedder().dimension)
    seen: list[IndexProgress] = []

    index_pending(session, HashingEmbedder(model_name=MODEL), store, on_progress=seen.append)

    assert seen, "progress must be reported"
    assert [p.chunks_done for p in seen] == sorted(p.chunks_done for p in seen)
    assert [p.documents_done for p in seen] == sorted(p.documents_done for p in seen)
    assert [p.seconds for p in seen] == sorted(p.seconds for p in seen)
    assert {p.chunks_total for p in seen} == {seen[-1].chunks_total}
    last = seen[-1]
    assert (last.documents_done, last.documents_total) == (2, 2)
    assert last.chunks_done == last.chunks_total == chunk_total(session) - 0
    assert last.seconds_left == 0
    store.close()


def test_the_estimate_needs_a_speed_first() -> None:
    nothing_yet = IndexProgress(0, 1, 0, 100, 0.0)
    halfway = IndexProgress(0, 1, 50, 100, 10.0)

    assert nothing_yet.seconds_left is None
    assert halfway.chunks_per_second == 5
    assert halfway.seconds_left == 10


def test_progress_is_not_reported_when_there_is_nothing_to_embed(session: Session) -> None:
    store = QdrantVectorStore.in_memory("idle", HashingEmbedder().dimension)
    seen: list[IndexProgress] = []

    index_pending(session, HashingEmbedder(model_name=MODEL), store, on_progress=seen.append)

    assert seen == []
    store.close()


def test_switching_models_in_the_middle_starts_the_new_model_from_the_beginning(
    session: Session, big_document: Document
) -> None:
    dimension = HashingEmbedder().dimension
    store_a = QdrantVectorStore.in_memory("model-a", dimension)
    store_b = QdrantVectorStore.in_memory("model-b", dimension)
    with pytest.raises(Interrupted):
        index_pending(session, FlakyEmbedder(fail_on_call=3), store_a)

    summary = index_pending(session, HashingEmbedder(model_name="test/model-b"), store_b)

    assert summary.chunks_embedded == chunk_total(session)
    assert store_b.count() == chunk_total(session)
    store_a.close()
    store_b.close()


def test_rebuilding_forgets_every_chunk_so_all_of_them_are_embedded_again(
    session: Session, big_document: Document
) -> None:
    store = QdrantVectorStore.in_memory("rebuild", HashingEmbedder().dimension)
    index_pending(session, HashingEmbedder(model_name=MODEL), store)
    reset_index(session, store)
    assert indexed_chunks(session) == 0

    again = HashingEmbedder(model_name=MODEL)
    summary = index_pending(session, again, store)

    assert summary.chunks_embedded == again.texts_embedded == chunk_total(session)
    store.close()


def test_chunking_a_document_again_leaves_no_vector_of_the_old_chunks(
    session: Session, data_dir: Path, big_document: Document
) -> None:
    store = QdrantVectorStore.in_memory("rechunk", HashingEmbedder().dimension)
    index_pending(session, HashingEmbedder(model_name=MODEL), store)
    before = store.count(big_document.id)

    chunk_document(session, data_dir, big_document, 600, 0)  # far fewer, larger chunks
    new_total = chunk_total(session)
    assert new_total < before
    index_pending(session, HashingEmbedder(model_name=MODEL), store)

    assert store.count(big_document.id) == new_total
    store.close()


def test_a_document_that_comes_back_after_being_replaced_is_embedded_again(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "a.txt"
    store = QdrantVectorStore.in_memory("return", HashingEmbedder().dimension)
    embedder = HashingEmbedder(model_name=MODEL)

    note.write_text("first words of the note")
    ingest_folders(session, data_dir, [notes], 100_000)
    index_pending(session, embedder, store)
    first = session.scalars(select(Document)).one()
    note.write_text("different words entirely")
    ingest_folders(session, data_dir, [notes], 100_000)
    index_pending(session, embedder, store)  # purges the first version
    assert store.count(first.id) == 0
    note.write_text("first words of the note")
    ingest_folders(session, data_dir, [notes], 100_000)

    index_pending(session, embedder, store)

    assert store.count(first.id) > 0  # it must not be marked indexed without vectors
    store.close()
