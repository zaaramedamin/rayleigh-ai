"""`python -m app summarize | compare | extract`, end to end through the command line."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app import cli
from app.ai.llm.base import LLMUnavailableError
from app.cli import main
from app.core.config import Settings
from app.knowledge.answering import tasks
from app.storage.database import create_db_engine
from app.storage.models import DOC_SUPERSEDED, Document
from tests.fakes import FakeLLM, HashingEmbedder

MakeSettings = Callable[..., Settings]

OATS = (
    "# Oats\n\nOats are high in fibre.\n\n## Cooking\n\nSimmer the oats in milk for five minutes.\n"
)
RICE = "# Rice\n\nRice needs twice its volume of water and eighteen minutes.\n"
INVOICES = "Invoice INV-1 totals 100 euros.\n\nInvoice INV-2 totals 250 euros.\n"


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """No CLI test may talk to a real Ollama server."""
    llm = FakeLLM("A short summary.", model_name="qwen3.5:4b", installed=["qwen3.5:4b"])
    monkeypatch.setattr(cli, "create_llm", lambda *_args, **_kwargs: llm)
    return llm


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> HashingEmbedder:
    embedder = HashingEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    folder = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")
    monkeypatch.setattr(cli, "load_embedder", lambda *_args: embedder)
    return embedder


@pytest.fixture
def settings(
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_model: HashingEmbedder,
    tmp_path: Path,
) -> Settings:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "oats.md").write_text(OATS)
    (folder / "rice.md").write_text(RICE)
    (folder / "invoices.md").write_text(INVOICES)
    result = make_settings(allowed_folders=[folder], retrieval_top_k=3, answer_min_score=0.3)
    main(["ingest"], settings=result)
    return result


def run(capsys: pytest.CaptureFixture[str], settings: Settings, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv), settings=settings)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def number_of(settings: Settings, name: str) -> int:
    engine = create_db_engine(settings.data_dir)
    try:
        with Session(engine) as session:
            return next(d.id for d in session.query(Document) if d.original_filename == name)
    finally:
        engine.dispose()


# --- summarize -------------------------------------------------------------------------------


def test_a_document_is_summarized_by_its_file_name(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    code, out, _err = run(capsys, settings, "summarize", "OATS.md")  # the case does not matter

    assert code == 0
    assert out.splitlines() == ["Summary of oats.md:", "A short summary."]
    assert "Simmer the oats in milk" in fake_llm.calls[0][1]


def test_a_document_is_summarized_by_its_number(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, out, _err = run(capsys, settings, "summarize", str(number_of(settings, "rice.md")))

    assert code == 0 and out.startswith("Summary of rice.md:")


def test_a_long_document_says_that_only_its_first_parts_were_read(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "MAX_PART_CHARS", 40)
    monkeypatch.setattr(tasks, "MAX_PARTS", 1)

    code, out, _err = run(capsys, settings, "summarize", "oats.md")

    assert code == 0 and "note: the document is long. Only the first 1 of its" in out


def test_a_name_no_document_has_is_refused_with_a_hint(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    code, _out, err = run(capsys, settings, "summarize", "nothing.md")

    assert code == 1 and "no current document is named 'nothing.md'" in err and "id 4:0" in err
    assert fake_llm.calls == []


def test_two_documents_with_the_same_name_must_be_told_apart_by_number(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_model: HashingEmbedder,
    tmp_path: Path,
) -> None:
    folders = []
    for name, text in (("one", "First note."), ("two", "Second note.")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "same.md").write_text(text)
        folders.append(folder)
    settings = make_settings(allowed_folders=folders)
    main(["ingest"], settings=settings)

    code, _out, err = run(capsys, settings, "summarize", "same.md")

    assert code == 1 and "2 documents are named 'same.md'" in err and "use a number" in err


def test_a_number_that_is_not_in_the_library_is_refused(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, _out, err = run(capsys, settings, "summarize", "999")

    assert code == 1 and "not in the library" in err


def test_an_older_version_is_refused(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    engine = create_db_engine(settings.data_dir)
    with Session(engine) as session:
        for document in session.query(Document):
            document.status = DOC_SUPERSEDED
        session.commit()
    engine.dispose()

    code, _out, err = run(capsys, settings, "summarize", "1")

    assert code == 1 and "not current" in err


def test_a_model_that_is_down_is_reported(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    fake_llm.error = LLMUnavailableError("Ollama is not reachable")

    code, _out, err = run(capsys, settings, "summarize", "oats.md")

    assert code == 1 and "Ollama is not reachable" in err


def test_summarize_needs_a_document(settings: Settings) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["summarize"], settings=settings)

    assert stopped.value.code == 2


# --- compare ---------------------------------------------------------------------------------


def test_documents_are_compared_with_their_sources(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = "Oats take five minutes [1]; rice takes eighteen [2]."

    code, out, _err = run(capsys, settings, "compare", "oats.md", "rice.md")

    assert code == 0 and out.startswith("Oats take five minutes [1]; rice takes eighteen [2].")
    assert "Sources:" in out
    assert f"  [1] oats.md  (document {number_of(settings, 'oats.md')})" in out
    assert f"  [2] rice.md  (document {number_of(settings, 'rice.md')})" in out


def test_a_comparison_with_no_valid_citation_prints_the_refusal_and_no_sources(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = "They differ."

    code, out, _err = run(capsys, settings, "compare", "oats.md", "rice.md")

    assert code == 0 and "I'm not going to guess" in out and "Sources:" not in out


def test_a_document_cannot_be_compared_with_itself_or_alone(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    code, _out, err = run(capsys, settings, "compare", "oats.md", "oats.md")
    assert code == 1 and "cannot be compared with itself" in err

    code, _out, err = run(capsys, settings, "compare", "oats.md")
    assert code == 1 and "from 2 to 4 documents" in err
    assert fake_llm.calls == []


def test_a_document_that_was_only_partly_read_is_named(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "MAX_COMPARE_CHARS", 120)
    fake_llm.reply = "They differ [1][2]."

    code, out, _err = run(capsys, settings, "compare", "oats.md", "rice.md")

    assert (
        code == 0 and "note: these are long, so only their start was read: oats.md, rice.md." in out
    )


# --- extract ---------------------------------------------------------------------------------


def test_facts_are_printed_as_a_table_with_their_sources(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = json.dumps(
        [
            {"item": "INV-1", "value": "100 euros", "note": 1},
            {"item": "Invoice INV-2", "value": "250 euros", "note": 1},
        ]
    )

    code, out, _err = run(capsys, settings, "extract", "Invoice", "INV", "totals", "euros")

    assert code == 0
    assert "  INV-1          100 euros  [1]" in out and "  Invoice INV-2  250 euros  [1]" in out
    assert "Sources:" in out and "[1] invoices.md" in out


def test_a_request_nothing_matches_says_so_and_never_calls_the_model(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    code, out, _err = run(capsys, settings, "extract", "zebra", "quantum", "giraffe")

    assert code == 0 and "don't have enough information in your notes" in out
    assert fake_llm.calls == []


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("[]", "No note holds a fact that matches the request."),
        ("Two invoices.", "could not be read as a table"),
        (json.dumps([{"item": "a", "value": "b", "note": 9}]), "None of the rows the model wrote"),
    ],
)
def test_each_way_an_extraction_can_come_up_empty_is_explained(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    fake_llm: FakeLLM,
    reply: str,
    message: str,
) -> None:
    fake_llm.reply = reply

    code, out, _err = run(capsys, settings, "extract", "Invoice", "INV", "totals", "euros")

    assert code == 0 and message in out and "Sources:" not in out


def test_rows_for_notes_that_do_not_exist_are_dropped_and_the_table_says_so(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = json.dumps(
        [{"item": "a", "value": "b", "note": 1}, {"item": "c", "value": "d", "note": 9}]
    )

    code, out, _err = run(capsys, settings, "extract", "Invoice", "INV", "totals", "euros")

    assert code == 0 and "  a  b  [1]" in out and "1 row(s) the model wrote were refused" in out


def test_a_blank_request_is_refused(capsys: pytest.CaptureFixture[str], settings: Settings) -> None:
    code, _out, err = run(capsys, settings, "extract", "   ")

    assert code == 1 and "query is empty" in err


def test_extract_without_the_embedding_model_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
) -> None:
    code, _out, err = run(capsys, make_settings(), "extract", "invoices")

    assert code == 1 and "download-model" in err
