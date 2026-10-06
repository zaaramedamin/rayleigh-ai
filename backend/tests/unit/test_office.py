"""Slice 3.3: Word, Excel and PowerPoint files, read safely with the standard library."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.indexing.service import index_pending
from app.knowledge.ingestion import office
from app.knowledge.ingestion.file_types import extract_document, file_type_for, is_markdown
from app.knowledge.ingestion.office import read_docx, read_pptx, read_xlsx
from app.knowledge.ingestion.parsers import ParseError
from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.retrieval.service import retrieve
from app.storage.models import Chunk
from app.storage.vector_store import QdrantVectorStore
from tests import office_factory as make
from tests.fakes import HashingEmbedder

# --- Word -----------------------------------------------------------------------------------------


def test_headings_lists_and_text_keep_their_structure() -> None:
    data = make.docx(
        make.para("Cooking oats", "Title"),
        make.para("Preparation", "Heading1"),
        make.para("Simmer the oats in milk for five minutes."),
        make.para("Variations", "Heading2"),
        make.para("Add fruit", bullet=True),
        make.para("Add nuts", bullet=True),
        make.para("Hidden depth", "Titre3"),  # a style named "heading 3" in another language
    )

    text = read_docx(data).text

    assert text == (
        "# Cooking oats\n\n## Preparation\n\nSimmer the oats in milk for five minutes.\n\n"
        "### Variations\n\n- Add fruit\n\n- Add nuts\n\n#### Hidden depth"
    )


def test_a_table_becomes_lines_of_column_and_value_pairs() -> None:
    data = make.docx(
        make.para("Nutrition", "Heading1"),
        make.table(
            [["Food", "kcal", "Protein"], ["Oats", "389", "17 g"], ["Rice", "130", "2.7 g"]]
        ),
    )

    text = read_docx(data).text

    assert "Food: Oats; kcal: 389; Protein: 17 g" in text
    assert "Food: Rice; kcal: 130; Protein: 2.7 g" in text  # a row is understood on its own


def test_tabs_and_line_breaks_and_tracked_changes() -> None:
    inserted = "<w:ins><w:r><w:t>new words</w:t></w:r></w:ins>"
    deleted = "<w:del><w:r><w:delText>old words</w:delText></w:r></w:del>"
    data = make.docx(
        make.para("", raw=make.run("one") + "<w:r><w:tab/></w:r>" + make.run("two")),
        make.para("", raw=make.run("first") + "<w:r><w:br/></w:r>" + make.run("second")),
        make.para("", raw=make.run("kept ") + inserted + deleted),
    )

    text = read_docx(data).text

    assert "one two" in text  # the tab is whitespace, collapsed
    assert (
        "first\nsecond" in text.replace("first second", "first\nsecond") or "first second" in text
    )
    assert "new words" in text and "old words" not in text


def test_a_text_box_stored_twice_is_read_once() -> None:
    twice = (
        "<mc:AlternateContent><mc:Choice>" + make.run("boxed text") + "</mc:Choice>"
        "<mc:Fallback>" + make.run("boxed text") + "</mc:Fallback></mc:AlternateContent>"
    )

    text = read_docx(make.docx(make.para("", raw=twice))).text

    assert text == "boxed text"


def test_a_word_file_without_styles_still_reads() -> None:
    assert read_docx(make.docx(make.para("Plain", "Heading1"), styles=None)).text == "Plain"


def test_without_a_title_the_first_heading_is_the_top_level() -> None:
    text = read_docx(make.docx(make.para("Part", "Heading1"), make.para("Sub", "Heading2"))).text

    assert text == "# Part\n\n## Sub"


def test_word_chunks_get_their_heading_paths(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.docx").write_bytes(
        make.docx(
            make.para("Kitchen", "Heading1"),
            make.para("Oats", "Heading2"),
            make.para("Simmer rolled oats in milk for five minutes."),
        )
    )

    summary = ingest_folders(session, data_dir, [folder], 1_000_000)

    assert summary.added == 1 and not summary.failed
    chunk = session.scalars(select(Chunk)).one()
    assert chunk.heading_path == "Kitchen > Oats"
    assert chunk.start_page is None  # Word files have no page numbers


# --- Excel ----------------------------------------------------------------------------------------


def test_each_sheet_is_a_section_and_each_row_reads_with_its_headings() -> None:
    data = make.xlsx(
        {
            "Food": [["Name", "kcal", "Vegan"], ["Oats", 389, True], ["Honey", 304, False]],
            "Drinks": [["Name", "Price"], ["Tea", 2.5]],
        }
    )

    text = read_xlsx(data).text

    assert "## Food" in text and "## Drinks" in text
    assert "Name: Oats; kcal: 389; Vegan: yes" in text
    assert "Name: Honey; kcal: 304; Vegan: no" in text
    assert "Name: Tea; Price: 2.5" in text


def test_empty_cells_do_not_shift_the_columns() -> None:
    data = make.xlsx({"S": [["A", "B", "C"], ["x", None, "z"], [None, "y", None]]})

    text = read_xlsx(data).text

    assert "A: x; C: z" in text and "B: y" in text


def test_formulas_are_read_as_their_saved_value_never_evaluated() -> None:
    data = make.xlsx({"S": [["Total"], [("f", 42)], [("i", "inline text")]]})

    text = read_xlsx(data).text

    assert "Total: 42" in text and "Total: inline text" in text
    assert "SUM" not in text


def test_many_rows_are_cut_at_the_limit_and_it_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(office, "MAX_SHEET_ROWS", 3)
    rows = [["id"]] + [[str(n)] for n in range(10)]

    text = read_xlsx(make.xlsx({"Big": rows})).text

    assert "(only the first 3 rows are read)" in text
    assert "id: 1" in text and "id: 2" not in text  # the header row counts toward the three


def test_a_very_hidden_sheet_is_skipped_but_a_hidden_one_is_read() -> None:
    data = make.xlsx(
        {"Open": [["a"], ["1"]], "Hidden": [["b"], ["2"]], "Secret": [["c"], ["3"]]},
        states={"Hidden": "hidden", "Secret": "veryHidden"},
    )

    text = read_xlsx(data).text

    assert "## Open" in text and "## Hidden" in text and "## Secret" not in text


def test_column_names_beyond_z_are_placed_correctly() -> None:
    row = ["h"] * 30
    data = make.xlsx({"S": [row, ["v"] * 30]})

    text = read_xlsx(data).text

    assert text.count("h: v") == 30


def test_a_workbook_with_nothing_in_it_has_no_text() -> None:
    with pytest.raises(ParseError, match="^no_text$"):
        read_xlsx(make.xlsx({"Empty": [[None, None]]}))


# --- PowerPoint -----------------------------------------------------------------------------------


def test_slides_have_titles_text_tables_and_notes() -> None:
    data = make.pptx(
        [
            {"title": "Welcome", "body": ["We make oats.", "Since 1990."]},
            {
                "title": "Prices",
                "table": [["Item", "Price"], ["Oats", "2 euro"]],
                "notes": "Mention the discount for large orders.",
            },
        ]
    )

    text = read_pptx(data).text

    assert text.startswith("# Slide 1\n\n## Welcome\n\nWe make oats.\n\nSince 1990.")
    assert "# Slide 2\n\n## Prices" in text
    assert "Item: Oats; Price: 2 euro" in text
    assert "Notes: Mention the discount for large orders." in text
    assert (
        "Notes: Mention the discount for large orders. 2" not in text
    )  # the slide number is left out


def test_slides_follow_the_order_of_the_presentation_not_the_file_numbers() -> None:
    data = make.pptx(
        [{"title": "First file"}, {"title": "Second file"}, {"title": "Third file"}],
        order=[3, 1, 2],
    )

    text = read_pptx(data).text

    assert text.index("Third file") < text.index("First file") < text.index("Second file")
    assert "# Slide 1\n\n## Third file" in text


def test_a_deck_without_slides_is_corrupt() -> None:
    with pytest.raises(ParseError, match="^corrupt$"):
        read_pptx(make.package({"ppt/presentation.xml": "<x/>"}))


# --- hostile and broken files ---------------------------------------------------------------------


@pytest.mark.parametrize("reader", [read_docx, read_xlsx, read_pptx])
def test_all_three_refuse_what_is_not_a_package(reader) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ParseError, match="^corrupt$"):
        reader(b"this is not a zip file")
    with pytest.raises(ParseError, match="^corrupt$"):
        reader(b"")
    with pytest.raises(ParseError, match="^old_format$"):
        reader(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100)


@pytest.mark.parametrize("reader", [read_docx, read_xlsx, read_pptx])
def test_a_zip_bomb_is_refused_before_it_is_unpacked(reader) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ParseError, match="^unsafe_archive$"):
        reader(make.zip_bomb())


@pytest.mark.parametrize(
    "name",
    [
        "../outside.xml",
        "word/../../outside.xml",
        "/etc/passwd",
        "C:/Windows/x.xml",
        "word\\..\\x.xml",
    ],
)
def test_an_entry_that_could_leave_its_folder_makes_the_archive_unsafe(name: str) -> None:
    with pytest.raises(ParseError, match="^unsafe_archive$"):
        read_docx(make.with_entry(name))


def test_a_part_bigger_than_the_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(office, "MAX_PART_BYTES", 1000)

    with pytest.raises(ParseError, match="^unsafe_archive$"):
        read_docx(make.docx(make.para("x" * 5000)))


def test_too_many_entries_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(office, "MAX_ENTRIES", 1)

    with pytest.raises(ParseError, match="^unsafe_archive$"):
        read_docx(make.docx(make.para("a")))  # the document and its styles: two entries


def test_xml_with_a_dtd_is_refused_instead_of_expanded() -> None:
    with pytest.raises(ParseError, match="^unsafe_archive$"):
        read_docx(make.with_doctype())


def test_a_password_protected_archive_entry_is_refused() -> None:
    with pytest.raises(ParseError, match="^encrypted$"):
        read_docx(make.marked_encrypted())


def test_a_package_missing_its_main_part_is_corrupt() -> None:
    with pytest.raises(ParseError, match="^corrupt$"):
        read_docx(make.package({"word/other.xml": "<x/>"}))
    with pytest.raises(ParseError, match="^corrupt$"):
        read_docx(make.package({"word/document.xml": "<w:broken"}))


def test_a_document_with_no_words_has_no_text() -> None:
    with pytest.raises(ParseError, match="^no_text$"):
        read_docx(make.docx(make.para("   ")))


def test_links_are_never_followed() -> None:
    external = (
        f'{make.XML}<Relationships xmlns="{make.REL}">'
        '<Relationship Id="rId1" Type="x" Target="http://example.com/evil.xml"'
        ' TargetMode="External"/>'
        "</Relationships>"
    )
    data = make.package(
        {
            "xl/workbook.xml": f'{make.XML}<workbook xmlns="{make.S}" xmlns:r="{make.R}"><sheets>'
            '<sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": external,
        }
    )

    with pytest.raises(ParseError, match="^no_text$"):
        read_xlsx(data)  # the external target is ignored, so there is no sheet to read


# --- the whole way through ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "maker"),
    [
        (".docx", lambda: make.docx(make.para("A Word note about rhubarb."))),
        (".xlsx", lambda: make.xlsx({"S": [["Plant"], ["rhubarb"]]})),
        (".pptx", lambda: make.pptx([{"title": "Rhubarb", "body": ["Grows in spring."]}])),
    ],
)
def test_each_type_is_registered_and_runs_in_the_sandbox(suffix: str, maker) -> None:  # type: ignore[no-untyped-def]
    kind = file_type_for(suffix.upper())

    assert kind is not None and kind.sandboxed and is_markdown(suffix)
    assert "rhubarb" in extract_document(suffix, maker()).text.lower()


def test_office_files_are_found_by_a_question_and_cited(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "trip.docx").write_bytes(
        make.docx(
            make.para("Lisbon", "Heading1"), make.para("The flight leaves at seven on Friday.")
        )
    )
    (folder / "costs.xlsx").write_bytes(
        make.xlsx({"Budget": [["Item", "Cost"], ["Hotel", 480], ["Flights", 220]]})
    )
    (folder / "pitch.pptx").write_bytes(
        make.pptx([{"title": "Launch plan", "body": ["Ship the beta in October."]}])
    )
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("office", embedder.dimension)

    summary = ingest_folders(session, data_dir, [folder], 1_000_000)
    index_pending(session, embedder, store)

    assert summary.added == 3 and not summary.failed
    top = {
        question: retrieve(session, embedder, store, question, top_k=1)[0]
        for question in (
            "flight leaves at seven on friday",
            "hotel cost item",
            "ship the beta in october",
        )
    }
    assert top["flight leaves at seven on friday"].source == "trip.docx"
    assert top["hotel cost item"].source == "costs.xlsx"
    assert "Hotel" in top["hotel cost item"].text
    assert top["ship the beta in october"].source == "pitch.pptx"
    assert top["ship the beta in october"].heading_path == "Slide 1 > Launch plan"
    store.close()


def test_hostile_office_files_are_reported_and_the_rest_is_still_read(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "bomb.docx").write_bytes(make.zip_bomb())
    (folder / "escape.docx").write_bytes(make.with_entry("../evil.xml"))
    (folder / "old.docx").write_bytes(b"\xd0\xcf\x11\xe0" + b"\0" * 64)
    (folder / "good.docx").write_bytes(make.docx(make.para("A perfectly ordinary note.")))

    summary = ingest_folders(session, data_dir, [folder], 100_000_000)

    assert dict(summary.failed) == {"unsafe_archive": 2, "old_format": 1}
    assert summary.added == 1
