from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.chunking.service import chunk_document
from app.knowledge.indexing.service import (
    count_pending,
    embedding_input,
    index_pending,
    reset_index,
    text_hash,
)
from app.storage.files import save_file
from app.storage.models import Chunk, Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

OATS = "# Oats\n\nOats are high in fibre.\n\n## Cooking\n\nBoil the oats in milk.\n"
RICE = "Rice needs twice its volume of water."
LONG = "\n\n".join(f"paragraph {i} about lentils " + "word " * 30 for i in range(12))


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder: HashingEmbedder) -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore.in_memory("test", embedder.dimension)
    yield vector_store
    vector_store.close()


def _add(
    session: Session, data_dir: Path, content: str, name: str, chunk_size: int = 1000
) -> Document:
    document = save_file(session, data_dir, content.encode(), name)
    chunk_document(session, data_dir, document, chunk_size, 50)
    return document


def test_index_pending_embeds_every_chunk(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    _add(session, data_dir, RICE, "rice.txt")

    summary = index_pending(session, embedder, store)

    assert summary.documents_indexed == 2
    assert summary.chunks_embedded == 3
    assert store.count() == 3
    assert count_pending(session, embedder.model_name) == 0
    assert {d.indexed_model for d in session.scalars(select(Document))} == {embedder.model_name}


def test_indexing_again_embeds_nothing(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    index_pending(session, embedder, store)
    embedded_before = embedder.texts_embedded

    summary = index_pending(session, embedder, store)

    assert summary.documents_indexed == 0
    assert embedder.texts_embedded == embedded_before


def test_rechunked_document_is_reindexed_and_its_old_vectors_removed(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    document = _add(session, data_dir, LONG, "lentils.txt", chunk_size=200)
    index_pending(session, embedder, store)
    assert store.count(document_id=document.id) > 1

    chunk_document(session, data_dir, document, 5000, 50)  # now a single chunk
    assert count_pending(session, embedder.model_name) == 1
    index_pending(session, embedder, store)

    assert store.count(document_id=document.id) == 1


def test_switching_model_makes_every_document_pending(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    _add(session, data_dir, RICE, "rice.txt")
    index_pending(session, embedder, store)

    assert count_pending(session, "test/another-model") == 2


def test_reset_index_drops_vectors_and_marks_documents_pending(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    index_pending(session, embedder, store)

    reset_index(session, store)

    assert store.count() == 0
    assert count_pending(session, embedder.model_name) == 1


def test_vectors_carry_the_chunk_fingerprint_and_file_type(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    _add(session, data_dir, RICE, "rice.txt")
    index_pending(session, embedder, store)

    hit = store.search(embedder.embed_query("rice water"), top_k=1, file_types=[".txt"])[0]

    chunk = session.scalars(
        select(Chunk).where(
            Chunk.document_id == hit.document_id, Chunk.chunk_index == hit.chunk_index
        )
    ).one()
    assert chunk.text == RICE
    assert hit.text_hash == text_hash(chunk.text)


def test_embedding_input_adds_the_heading_path() -> None:
    with_heading = Chunk(text="Boil the oats.", heading_path="Oats > Cooking")
    without_heading = Chunk(text="Plain note.", heading_path="")

    assert embedding_input(with_heading) == "Oats > Cooking\n\nBoil the oats."
    assert embedding_input(without_heading) == "Plain note."


def test_document_without_chunks_is_marked_indexed(
    session: Session, data_dir: Path, embedder: HashingEmbedder, store: QdrantVectorStore
) -> None:
    save_file(session, data_dir, b"not chunked yet", "raw.txt")

    summary = index_pending(session, embedder, store)

    assert summary.documents_indexed == 1
    assert store.count() == 0


class _BrokenEmbedder(HashingEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("model crashed")


def test_a_failed_run_leaves_the_document_pending(
    session: Session, data_dir: Path, store: QdrantVectorStore
) -> None:
    _add(session, data_dir, OATS, "oats.md")
    broken = _BrokenEmbedder()

    with pytest.raises(RuntimeError):
        index_pending(session, broken, store)

    assert count_pending(session, broken.model_name) == 1
