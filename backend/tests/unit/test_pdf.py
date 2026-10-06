"""Slice 3.2: PDFs are read in the sandbox, cited by page, and fail with a clear reason."""

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.indexing.service import index_pending
from app.knowledge.ingestion.file_types import extract_document, file_type_for
from app.knowledge.ingestion.parsers import Extracted, ParseError
from app.knowledge.ingestion.readers import (
    OPTIONS_ENV,
    clean_page_text,
    drop_repeated_lines,
    read_pdf,
)
from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.retrieval.service import describe_location, retrieve
from app.storage.models import Chunk
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder
from tests.pdf_factory import blank_pdf, encrypted, make_pdf

REPORT = [
    ["Quarterly report", "Revenue grew by twelve percent in the north region."],
    ["Costs fell after the warehouse moved.", "The new warehouse is in Rotterdam."],
    ["The outlook for next year is cautious.", "Hiring is paused until March."],
]


def read(data: bytes) -> Extracted:
    return read_pdf(data)


# --- the reader ----------------------------------------------------------------------------------


def test_the_text_comes_back_page_by_page_with_the_page_starts() -> None:
    extracted = read(make_pdf(REPORT))

    pages = extracted.text.split("\n\n")
    assert len(pages) == 3
    assert "Revenue grew by twelve percent" in pages[0]
    assert "Rotterdam" in pages[1]
    assert "Hiring is paused" in pages[2]
    assert extracted.page_starts is not None
    assert [extracted.text[start : start + 7] for start in extracted.page_starts] == [
        "Quarter",
        "Costs f",
        "The out",
    ]


def test_special_characters_in_the_text_survive() -> None:
    extracted = read(make_pdf([["Total (net) = 5 \\ 6 and the (nested (brackets))"]]))

    assert "Total (net) = 5 \\ 6" in extracted.text
    assert "(nested (brackets))" in extracted.text


def test_a_pdf_with_no_text_says_so() -> None:
    with pytest.raises(ParseError, match="^no_text$"):
        read(blank_pdf(3))


@pytest.mark.parametrize("data", [b"", b"this is not a pdf", b"%PDF-1.4\n garbage \n%%EOF"])
def test_damaged_files_are_reported_as_corrupt(data: bytes) -> None:
    with pytest.raises(ParseError, match="^corrupt$"):
        read(data)


def test_a_truncated_pdf_does_not_crash_the_reader() -> None:
    data = make_pdf(REPORT)

    try:
        extracted = read(data[: len(data) // 2])
    except ParseError as exc:
        assert str(exc) in {"corrupt", "no_text"}
    else:
        assert isinstance(extracted, Extracted)  # pypdf may recover some of it


def test_a_pdf_that_needs_a_password_is_refused() -> None:
    with pytest.raises(ParseError, match="^encrypted$"):
        read(encrypted(make_pdf(REPORT), "a real password"))


def test_a_pdf_encrypted_with_an_empty_password_opens() -> None:
    extracted = read(encrypted(make_pdf(REPORT), ""))

    assert "Rotterdam" in extracted.text


def test_too_many_pages_is_refused_before_reading_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPTIONS_ENV, json.dumps({"pdf_max_pages": 2}))

    with pytest.raises(ParseError, match="^too_many_pages$"):
        read(make_pdf(REPORT))
    assert read(make_pdf(REPORT[:2])).page_starts is not None  # two pages are fine


# --- cleaning ------------------------------------------------------------------------------------


TOPICS = ["apples", "bananas", "cherries", "dates", "elderberries", "figs"]


def with_header_and_footer(count: int = 6) -> list[list[str]]:
    """Pages with a running header, a page number footer, and a body that is different each time."""
    return [
        [
            "ACME CONFIDENTIAL",
            f"This page is about {TOPICS[number - 1]}.",
            "The second line of the body says something else.",
            f"A third line, about {TOPICS[number - 1]} again.",
            "And a fourth line to finish the page.",
            f"Page {number} of {count}",
        ]
        for number in range(1, count + 1)
    ]


def test_running_headers_and_page_numbers_are_dropped() -> None:
    extracted = read(make_pdf(with_header_and_footer()))

    assert "ACME" not in extracted.text
    assert "Page 3 of 6" not in extracted.text
    assert "This page is about cherries." in extracted.text
    assert len(extracted.page_starts or ()) == 6  # the pages are still all there


def test_they_can_be_kept_when_they_hold_real_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPTIONS_ENV, json.dumps({"pdf_keep_headers_footers": True}))

    extracted = read(make_pdf(with_header_and_footer()))

    assert "ACME CONFIDENTIAL" in extracted.text and "Page 3 of 6" in extracted.text


def test_a_line_that_repeats_only_a_little_is_kept() -> None:
    pages = [
        "Intro\nA unique first body.",
        "Intro\nAnother body.",
        "Different\nThird body.",
        "Last\nFourth.",
    ]

    assert drop_repeated_lines(pages) == pages  # "Intro" is on 2 of 4 pages: not a header


def test_a_short_document_is_left_alone() -> None:
    pages = ["Same title\nbody one", "Same title\nbody two"]

    assert drop_repeated_lines(pages) == pages


def test_hyphenated_line_ends_are_rejoined_but_real_hyphens_stay() -> None:
    text = (
        "an informa-\ntion system, a well-known\nidea, Anglo-\nSaxon words, a trail-\n  ing space"
    )

    cleaned = clean_page_text(text)

    assert "information system" in cleaned
    assert "well-known" in cleaned
    assert "Anglo-\nSaxon" in cleaned  # a capital after the hyphen: keep it
    assert "trailing space" in cleaned


def test_ligatures_become_plain_letters() -> None:
    assert clean_page_text("oﬃce ﬁle ﬂow") == "office file flow"


# --- the whole way through -----------------------------------------------------------------------


def test_the_pdf_type_is_registered_and_runs_in_the_sandbox() -> None:
    kind = file_type_for(".PDF")

    assert kind is not None and kind.sandboxed
    extracted = extract_document(".pdf", make_pdf(REPORT))
    assert "Rotterdam" in extracted.text


def test_a_pdf_question_is_answered_with_the_right_page(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "report.pdf").write_bytes(make_pdf(REPORT))
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("pdf", embedder.dimension)

    summary = ingest_folders(session, data_dir, [folder], 1_000_000, chunk_size=80, chunk_overlap=0)
    index_pending(session, embedder, store)

    assert summary.added == 1 and not summary.failed
    found = retrieve(session, embedder, store, "where is the new warehouse", top_k=1)[0]
    assert found.source == "report.pdf"
    assert "Rotterdam" in found.text
    assert (found.start_page, found.end_page) == (2, 2)
    assert (
        describe_location(found.start_line, found.end_line, found.start_page, found.end_page)
        == "page 2"
    )
    assert {c.start_page for c in session.scalars(select(Chunk))} == {1, 2, 3}
    store.close()


def test_pdf_failures_are_counted_by_reason_and_never_stop_the_rest(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "scan.pdf").write_bytes(blank_pdf())
    (folder / "locked.pdf").write_bytes(encrypted(make_pdf(REPORT), "secret"))
    (folder / "broken.pdf").write_bytes(b"not really a pdf")
    (folder / "good.pdf").write_bytes(make_pdf(REPORT))
    (folder / "note.txt").write_text("An ordinary note.")

    summary = ingest_folders(session, data_dir, [folder], 1_000_000)

    assert dict(summary.failed) == {"no_text": 1, "encrypted": 1, "corrupt": 1}
    assert summary.added == 2  # the good PDF and the note
