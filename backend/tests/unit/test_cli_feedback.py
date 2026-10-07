"""`python -m app feedback`: see the marks, or turn the failures into evaluation questions."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app import cli
from app.assistant.feedback import add_feedback
from app.cli import main
from app.core.config import Settings
from app.storage.database import create_db_engine

MakeSettings = Callable[..., Settings]


@pytest.fixture
def settings(make_settings: MakeSettings, migrated_data_dir: Path) -> Settings:
    return make_settings()


@pytest.fixture(autouse=True)
def default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where a bare `feedback export` writes. Never the real eval-private folder."""
    target = tmp_path / "private" / "feedback-candidates.json"
    monkeypatch.setattr(cli, "DEFAULT_FILE", target)
    return target


def run(capsys: pytest.CaptureFixture[str], settings: Settings, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv), settings=settings)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def mark(settings: Settings, kind: str, question: str, mode: str = "notes") -> None:
    engine = create_db_engine(settings.data_dir)
    with Session(engine) as session:
        add_feedback(session, kind=kind, mode=mode, question=question, answer="An answer [1].")
    engine.dispose()


def test_with_no_marks_it_says_how_to_make_one(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, out, _err = run(capsys, settings, "feedback")

    assert code == 0 and "no marks yet" in out and "wrong source" in out


def test_the_list_counts_every_kind_and_shows_the_failures_but_not_the_thumbs_up(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    mark(settings, "helpful", "a happy question")
    mark(settings, "not_helpful", "an unhappy question")
    mark(settings, "wrong_source", "x" * 200)

    code, out, _err = run(capsys, settings, "feedback")
    _same_code, same, _err = run(capsys, settings, "feedback", "list")

    assert code == 0 and same == out  # list is the default
    assert "marks: 1 helpful, 1 not helpful, 1 wrong source, 0 missing information" in out
    assert "an unhappy question" in out and "wrong source" in out
    assert "a happy question" not in out
    assert "x" * 67 + "..." in out and "x" * 68 not in out  # long questions are cut short
    assert "feedback export" in out


def test_export_writes_the_failures_and_says_what_to_do_next(
    capsys: pytest.CaptureFixture[str], settings: Settings, tmp_path: Path
) -> None:
    mark(settings, "not_helpful", "first failure")
    mark(settings, "missing_info", "second failure")
    mark(settings, "helpful", "fine")
    target = tmp_path / "out" / "candidates.json"

    code, out, _err = run(capsys, settings, "feedback", "export", "--to", str(target))

    assert code == 0 and "added 2 evaluation question(s) to review" in out and str(target) in out
    assert "expected_sources" in out and "answer_contains" in out and "eval --set" in out
    questions = [
        e["question"] for e in json.loads(target.read_text(encoding="utf-8"))["candidates"]
    ]
    assert questions == ["first failure", "second failure"]


def test_a_bare_export_goes_to_the_private_folder(
    capsys: pytest.CaptureFixture[str], settings: Settings, default_file: Path
) -> None:
    mark(settings, "wrong_source", "a failure")

    code, out, _err = run(capsys, settings, "feedback", "export")

    assert code == 0 and default_file.exists() and str(default_file) in out


def test_exporting_again_adds_nothing_and_says_so(
    capsys: pytest.CaptureFixture[str], settings: Settings, tmp_path: Path
) -> None:
    mark(settings, "not_helpful", "a failure")
    target = tmp_path / "candidates.json"
    run(capsys, settings, "feedback", "export", "--to", str(target))

    code, out, _err = run(capsys, settings, "feedback", "export", "--to", str(target))

    assert (
        code == 0
        and "nothing new to add" in out
        and "already in the file, left as they were: 1" in out
    )
    assert "Open the file" not in out  # nothing to review that was not already there


def test_failures_on_general_chat_are_reported_as_left_out(
    capsys: pytest.CaptureFixture[str], settings: Settings, tmp_path: Path
) -> None:
    mark(settings, "not_helpful", "a chat question", mode="general")

    code, out, _err = run(capsys, settings, "feedback", "export", "--to", str(tmp_path / "c.json"))

    assert code == 0 and "1 failure(s) on general chat were not added" in out
    assert not (tmp_path / "c.json").exists()


def test_a_file_that_is_not_a_candidates_file_is_refused_and_left_alone(
    capsys: pytest.CaptureFixture[str], settings: Settings, tmp_path: Path
) -> None:
    mark(settings, "not_helpful", "a failure")
    target = tmp_path / "notes.json"
    target.write_text('{"something": "else"}', encoding="utf-8")

    code, _out, err = run(capsys, settings, "feedback", "export", "--to", str(target))

    assert code == 1 and "not a candidates file" in err
    assert target.read_text(encoding="utf-8") == '{"something": "else"}'


def test_an_action_that_does_not_exist_is_refused(settings: Settings) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["feedback", "delete-everything"], settings=settings)

    assert stopped.value.code == 2
