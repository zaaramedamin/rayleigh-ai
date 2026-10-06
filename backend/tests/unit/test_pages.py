"""Slice 3.1: page numbers travel from a reader to the chunks and to the citations."""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.chunking.chunker import chunk_text
from app.knowledge.indexing.service import index_pending
from app.knowledge.ingestion import file_types
from app.knowledge.ingestion.file_types import FileType, extract_document, extract_text
from app.knowledge.ingestion.parsers import Extracted
from app.knowledge.ingestion.sandbox import run_in_sandbox
from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.retrieval.service import describe_location, retrieve
from app.storage.models import Chunk
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

PAGE_ONE = "Page one talks about apples.\nIt has two lines."
PAGE_TWO = "Page two talks about bananas."
PAGE_THREE = "Page three talks about cherries."


def pages(*texts: str, separator: str = "\n") -> Extracted:
    """The text of several pages, and where each begins."""
    starts, text = [], ""
    for page in texts:
        starts.append(len(text))
        text += page + separator
    return Extracted(text, tuple(starts))


def chunks_of(extracted: Extracted, size: int = 40) -> list[tuple[int | None, int | None]]:
    return [
        (c.start_page, c.end_page)
        for c in chunk_text(
            extracted.text,
            markdown=False,
            chunk_size=size,
            overlap=0,
            page_starts=extracted.page_starts,
        )
    ]


# --- the chunker -------------------------------------------------------------------------------


def test_each_chunk_knows_its_page() -> None:
    document = pages(PAGE_ONE, PAGE_TWO, PAGE_THREE, separator="\n\n")

    assert chunks_of(document, size=60) == [(1, 1), (2, 2), (3, 3)]


def test_a_chunk_that_spans_two_pages_says_so() -> None:
    document = pages("short one", "short two", "short three", separator="\n\n")

    spans = chunks_of(document, size=200)  # everything fits in one chunk

    assert spans == [(1, 3)]


def test_text_without_pages_has_no_page_numbers() -> None:
    assert chunks_of(Extracted("just some text\n\nand more text")) == [(None, None)]


def test_windows_line_endings_do_not_shift_the_pages() -> None:
    text = "first page line\r\nsecond line\r\n\r\nsecond page here\r\n"
    start_of_page_two = text.index("second page")

    result = chunks_of(Extracted(text, (0, start_of_page_two)), size=30)

    assert result[0][0] == 1 and result[-1][1] == 2


def test_a_page_that_starts_in_the_middle_of_a_line_is_reported_as_both() -> None:
    text = "alpha beta\ngamma delta"
    result = chunks_of(Extracted(text, (0, 6)), size=100)  # page 2 begins at "beta"

    assert result == [(1, 2)]  # the first line touches page 1 and page 2


def test_the_chunker_still_gives_the_same_chunks_without_page_information() -> None:
    plain = chunk_text("a\n\nb\n\nc", markdown=False, chunk_size=10, overlap=0)
    with_pages = chunk_text(
        "a\n\nb\n\nc", markdown=False, chunk_size=10, overlap=0, page_starts=(0, 3, 6)
    )

    assert [(c.text, c.start_line, c.end_line) for c in plain] == [
        (c.text, c.start_line, c.end_line) for c in with_pages
    ]


# --- readers return text plus pages --------------------------------------------------------------


def test_the_existing_formats_have_no_pages_and_unchanged_text() -> None:
    assert extract_document(".md", b"# Title\n\nbody") == Extracted("# Title\n\nbody", None)
    assert extract_text(".csv", b"a,b\n1,2\n") == "a,b\n1,2\n"
    assert extract_document(".html", b"<p>one</p>").page_starts is None


@pytest.fixture
def paged_format(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Register `.pgs`, a made-up format read in the sandbox, with a chosen reader."""

    def register(reader_name: str) -> None:
        def read(data: bytes) -> Extracted:
            return run_in_sandbox(f"tests.sandbox_samples:{reader_name}", data, timeout=3)

        kind = FileType("Paged test", (".pgs",), "a made-up paged format", read, sandboxed=True)
        monkeypatch.setitem(file_types._BY_EXTENSION, ".pgs", kind)

    return register


def test_a_paged_file_is_cited_with_its_page(
    session: Session, data_dir: Path, tmp_path: Path, paged_format: Callable[[str], None]
) -> None:
    paged_format("two_pages")
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "report.pgs").write_bytes(
        b"The first page is about apples and oranges.|The second page is about bananas."
    )
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("pages", embedder.dimension)

    summary = ingest_folders(session, data_dir, [folder], 100_000, chunk_size=50, chunk_overlap=0)
    index_pending(session, embedder, store)

    assert summary.added == 1 and not summary.failed
    rows = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert [(c.start_page, c.end_page) for c in rows] == [(1, 1), (2, 2)]
    found = retrieve(session, embedder, store, "bananas on the second page", top_k=1)[0]
    assert (found.source, found.start_page, found.end_page) == ("report.pgs", 2, 2)
    assert (
        describe_location(found.start_line, found.end_line, found.start_page, found.end_page)
        == "page 2"
    )
    store.close()


@pytest.mark.parametrize(
    ("reader", "reason"),
    [("hangs", "timeout"), ("dies", "crashed"), ("refuses", "encrypted"), ("huge", "crashed")],
)
def test_a_file_whose_reader_fails_is_reported_and_the_others_are_still_read(
    session: Session,
    data_dir: Path,
    tmp_path: Path,
    paged_format: Callable[[str], None],
    reader: str,
    reason: str,
) -> None:
    paged_format(reader)
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "bad.pgs").write_bytes(b"whatever")
    (folder / "good.txt").write_text("An ordinary note.")

    summary = ingest_folders(session, data_dir, [folder], 100_000)

    if reader == "huge":
        # The sandbox cap is generous, so this one is simply read; it is not a failure.
        assert summary.added == 2
        return
    assert dict(summary.failed) == {reason: 1}
    assert summary.added == 1  # the ordinary note was not held up


def test_where_a_chunk_comes_from() -> None:
    assert describe_location(3, 9, None, None) == "lines 3-9"
    assert describe_location(3, 9, 12, 12) == "page 12"
    assert describe_location(3, 9, 12, None) == "page 12"
    assert describe_location(3, 9, 12, 14) == "pages 12-14"
