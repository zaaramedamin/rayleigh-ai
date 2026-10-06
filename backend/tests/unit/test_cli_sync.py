"""The sync, status and prune commands, end to end through the command line."""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import cli
from app.cli import main
from app.core.config import Settings
from app.storage.database import create_db_engine
from app.storage.models import Document
from tests.fakes import FakeLLM, HashingEmbedder

MakeSettings = Callable[..., Settings]


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """No CLI test may talk to a real Ollama server."""
    llm = FakeLLM(model_name="qwen3.5:4b", installed=["qwen3.5:4b"])
    monkeypatch.setattr(cli, "create_llm", lambda *_args, **_kwargs: llm)
    return llm


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "keep.md").write_text("# Keep\n\nThis note stays.\n")
    (folder / "gone.txt").write_text("This note will be deleted from the folder.")
    return folder


@pytest.fixture
def settings(make_settings: MakeSettings, migrated_data_dir: Path, notes: Path) -> Settings:
    return make_settings(allowed_folders=[notes])


def run(capsys: pytest.CaptureFixture[str], settings: Settings, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv), settings=settings)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def document_count(settings: Settings) -> int:
    engine = create_db_engine(settings.data_dir)
    try:
        with Session(engine) as session:
            return session.scalar(select(func.count()).select_from(Document)) or 0
    finally:
        engine.dispose()


def test_a_first_sync_has_nothing_to_report_about_edits_or_deletions(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, out, _err = run(capsys, settings, "sync")

    assert code == 0
    assert "added:                     2" in out
    assert "replaced by an edit" not in out
    assert "not found again" not in out


def test_an_edit_is_reported_and_the_old_version_is_kept(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    run(capsys, settings, "ingest")
    (notes / "keep.md").write_text("# Keep\n\nThis note was changed.\n")

    code, out, _err = run(capsys, settings, "ingest")
    _code, status, _err = run(capsys, settings, "status")

    assert code == 0
    assert "replaced by an edit:       1" in out
    assert "documents:       2" in status  # the two current notes
    assert "older versions:  1" in status


def test_a_deleted_file_is_reported_after_the_second_sync_and_status_says_prune(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    run(capsys, settings, "ingest")
    (notes / "gone.txt").unlink()

    _code, first, _err = run(capsys, settings, "ingest")
    _code, second, _err = run(capsys, settings, "ingest")
    _code, status, _err = run(capsys, settings, "status")

    assert "not found again" not in first
    assert "files not found again:     1 document(s) now marked missing" in second
    assert "missing:         1" in status
    assert "python -m app prune" in status
    assert "documents:       1" in status  # only what is still found counts


def _one_missing(capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path) -> None:
    run(capsys, settings, "ingest")
    (notes / "gone.txt").unlink()
    run(capsys, settings, "ingest")
    run(capsys, settings, "ingest")


def test_prune_dry_run_lists_and_removes_nothing(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    _one_missing(capsys, settings, notes)

    code, out, _err = run(capsys, settings, "prune", "--dry-run")

    assert code == 0
    assert "gone.txt" in out
    assert "dry run: nothing was removed" in out
    assert document_count(settings) == 2


def test_prune_without_confirmation_removes_nothing(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    _one_missing(capsys, settings, notes)

    code, out, err = run(capsys, settings, "prune")  # no terminal to ask on, and no --yes

    assert code == 1
    assert "gone.txt" in out
    assert "run again with --yes" in err
    assert document_count(settings) == 2


def test_prune_with_yes_removes_the_document_and_says_what_went(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    _one_missing(capsys, settings, notes)
    stored = list((settings.data_dir / "files").rglob("*"))
    assert len([p for p in stored if p.is_file()]) == 2

    code, out, _err = run(capsys, settings, "prune", "--yes")

    assert code == 0
    assert "removed 1 document(s), 1 chunks and 1 stored file(s)" in out
    assert "compacted" in out
    assert document_count(settings) == 1
    assert len([p for p in (settings.data_dir / "files").rglob("*") if p.is_file()]) == 1
    assert (notes / "keep.md").exists()  # the original files are never touched
    _code, again, _err = run(capsys, settings, "prune")
    assert "nothing to remove" in again


def test_prune_only_removes_old_versions_when_asked(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    run(capsys, settings, "ingest")
    (notes / "keep.md").write_text("# Keep\n\nChanged.\n")
    run(capsys, settings, "ingest")

    _code, plain, _err = run(capsys, settings, "prune", "--yes")
    assert "nothing to remove" in plain
    assert "--superseded" in plain
    code, out, _err = run(capsys, settings, "prune", "--superseded", "--yes")

    assert code == 0
    assert "1 replaced version(s)" in out
    assert document_count(settings) == 2


def test_the_folders_added_in_the_interface_are_synced_by_the_command_line_too(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    tmp_path: Path,
) -> None:
    from app.knowledge.library.state import UpdateState

    extra = tmp_path / "from-the-interface"
    extra.mkdir()
    (extra / "x.txt").write_text("a note in a folder added through the interface")
    with UpdateState(migrated_data_dir) as state:
        state.folders.append(str(extra))

    code, out, _err = run(capsys, make_settings(), "ingest")

    assert code == 0
    assert "added:                     1" in out


# --- progress while indexing --------------------------------------------------------------------


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> HashingEmbedder:
    """Pretend the default model is downloaded, and embed with a fast deterministic fake."""
    embedder = HashingEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    folder = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")
    monkeypatch.setattr(cli, "load_embedder", lambda *_args: embedder)
    return embedder


def test_indexing_shows_how_far_it_has_come_on_stderr_only(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_model: HashingEmbedder
) -> None:
    code, out, err = run(capsys, settings, "ingest")

    assert code == 0
    assert "2 document(s) indexed, 2 chunks embedded" in out
    assert "indexing: 2 of 2 chunks (100%)" in err
    assert "indexing:" not in out  # the output stays easy to read and to pipe
    assert "note" not in err  # counts only: never what the notes say


def test_nothing_to_index_prints_no_progress(
    capsys: pytest.CaptureFixture[str], settings: Settings, fake_model: HashingEmbedder
) -> None:
    run(capsys, settings, "ingest")

    _code, out, err = run(capsys, settings, "ingest")

    assert "up to date" in out
    assert "indexing:" not in err


def test_pressing_ctrl_c_says_that_progress_was_saved(
    capsys: pytest.CaptureFixture[str], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_cmd_index", interrupted)

    code, _out, err = run(capsys, settings, "index")

    assert code == 130
    assert "Progress is saved" in err
    assert "Traceback" not in err


# --- files that cannot be read ------------------------------------------------------------------


def test_a_file_that_cannot_be_read_is_explained_in_plain_words(
    capsys: pytest.CaptureFixture[str], settings: Settings, notes: Path
) -> None:
    from tests.pdf_factory import blank_pdf, encrypted, make_pdf

    (notes / "scan.pdf").write_bytes(blank_pdf())
    (notes / "locked.pdf").write_bytes(encrypted(make_pdf([["Some text on a page of text."]]), "x"))
    (notes / "fine.pdf").write_bytes(make_pdf([["Plenty of ordinary words on this page."]]))

    code, out, _err = run(capsys, settings, "ingest")

    assert code == 0
    assert "failed:                    2" in out
    assert "no_text:" in out and "probably a scan" in out
    assert "encrypted:" in out and "protected by a password" in out
    assert "added:                     3" in out  # the two notes and the readable PDF
