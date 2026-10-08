"""`python -m app agent ...`, end to end through the command line."""

import builtins
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app import cli
from app.agent.permissions import Grants, load_grants, save_grants
from app.cli import main
from app.core.config import Settings
from tests.fakes import HashingEmbedder, ScriptedModel, asks, says

MakeSettings = Callable[..., Settings]

OATS = "# Oats\n\nSimmer the oats in milk for five minutes.\n"


@pytest.fixture
def settings(make_settings: MakeSettings, migrated_data_dir: Path) -> Settings:
    return make_settings(answer_min_score=0.3)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedModel:
    """No CLI test may talk to a real Ollama server."""
    scripted = ScriptedModel()
    monkeypatch.setattr(cli, "create_llm", lambda *_args, **_kwargs: scripted)
    return scripted


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What the owner types at the keyboard, in order."""
    answers: list[str] = []

    def read(prompt: str = "") -> str:
        if not answers:
            raise EOFError
        return answers.pop(0)

    monkeypatch.setattr(builtins, "input", read)
    return answers


def run(capsys: pytest.CaptureFixture[str], settings: Settings, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(["agent", *argv], settings=settings)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_fresh_install_has_the_agent_off(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, out, _err = run(capsys, settings, "status")

    assert code == 0 and "agent: OFF" in out and "web access: off" in out
    assert load_grants(settings.data_dir) == Grants()  # looking changes nothing


def test_the_switches_are_changed_from_the_command_line(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    assert run(capsys, settings, "enable")[0] == 0
    assert run(capsys, settings, "allow", "calculator", "current_time")[0] == 0
    assert run(capsys, settings, "web", "on")[0] == 0

    assert load_grants(settings.data_dir) == Grants(
        enabled=True, tools=frozenset({"calculator", "current_time"}), web=True
    )
    _code, out, _err = run(capsys, settings, "status")
    assert "agent: ON" in out and "calculator       on " in out and "search_notes     off" in out


def test_a_task_runs_end_to_end_and_is_logged(
    capsys: pytest.CaptureFixture[str], settings: Settings, model: ScriptedModel
) -> None:
    save_grants(settings.data_dir, Grants(enabled=True, tools=frozenset({"calculator"})))
    model.turns = [asks("calculator", expression="2**10"), says("It is 1024.")]

    code, out, err = run(capsys, settings, "run", "what", "is", "2", "to", "the", "10th?")

    assert code == 0 and "It is 1024." in out and "Calculate 2**10: done" in out
    assert "thinking ..." in err and "running: Calculate 2**10" in err
    _code, log, _err = run(capsys, settings, "log")
    assert "calculator" in log and "allow/policy" in log and "run_finished" in log


def test_a_task_cannot_run_while_the_agent_is_off(
    capsys: pytest.CaptureFixture[str], settings: Settings, model: ScriptedModel
) -> None:
    code, _out, err = run(capsys, settings, "run", "hello")

    assert code == 1 and "the agent is off" in err and model.chats == []


def test_nothing_runs_unless_the_owner_switched_that_tool_on(
    capsys: pytest.CaptureFixture[str], settings: Settings, model: ScriptedModel
) -> None:
    save_grants(settings.data_dir, Grants(enabled=True))  # the agent is on, no tool is
    model.turns = [says("I have no tools, so I cannot calculate that.")]

    code, out, _err = run(capsys, settings, "run", "what is 2+2?")

    assert code == 0 and "no tools" in out
    assert model.chats[0][2] == []  # the model was not even offered the calculator


def test_the_owner_answers_a_question_at_the_keyboard(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    model: ScriptedModel,
    typed: list[str],
) -> None:
    from app.ai.llm.base import LLMUnavailableError

    save_grants(settings.data_dir, Grants(enabled=True))
    model.turns = [LLMUnavailableError("Ollama is not running"), says("Back again.")]
    typed.append("r")

    code, out, _err = run(capsys, settings, "run", "hello")

    assert code == 0 and "Something went wrong" in out and "Ollama is not running" in out
    assert "Back again." in out


def test_a_closed_keyboard_means_stop(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    model: ScriptedModel,
    typed: list[str],
) -> None:
    from app.ai.llm.base import LLMUnavailableError

    save_grants(settings.data_dir, Grants(enabled=True))
    model.turns = [LLMUnavailableError("down")]  # nothing typed: input() raises EOFError

    code, out, _err = run(capsys, settings, "run", "hello")

    assert code == 1 and "no answer: stopping" in out


def test_the_log_can_be_exported_and_erased(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    model: ScriptedModel,
    tmp_path: Path,
) -> None:
    save_grants(settings.data_dir, Grants(enabled=True))
    model.turns = [says("Nothing to do.")]
    run(capsys, settings, "run", "idle")
    target = tmp_path / "log.json"

    code, out, _err = run(capsys, settings, "log", "--export", str(target))
    assert code == 0 and "wrote 2 entries" in out
    assert [r["kind"] for r in json.loads(target.read_text(encoding="utf-8"))] == [
        "run_started",
        "run_finished",
    ]

    code, out, _err = run(capsys, settings, "log", "--erase", "--yes")
    assert code == 0 and "erased 2 entries." in out
    assert "has not done anything yet" in run(capsys, settings, "log")[1]


def test_bad_arguments_are_one_line_errors(
    capsys: pytest.CaptureFixture[str], settings: Settings
) -> None:
    code, _out, err = run(capsys, settings, "allow", "delete_all")
    assert code == 1 and "no such tool: delete_all" in err

    code, _out, err = run(capsys, settings, "web", "maybe")
    assert code == 1 and "web on" in err

    with pytest.raises(SystemExit) as stopped:
        main(["agent", "fly"], settings=settings)
    assert stopped.value.code == 2


# --- searching the notes through the agent --------------------------------------------------


@pytest.fixture
def library(
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    models_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Settings:
    embedder = HashingEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    folder = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")
    monkeypatch.setattr(cli, "load_embedder", lambda *_args: embedder)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "oats.md").write_text(OATS)
    result = make_settings(allowed_folders=[notes], retrieval_top_k=3, answer_min_score=0.3)
    assert main(["ingest"], settings=result) == 0
    return result


def test_the_agent_can_search_the_library_when_the_owner_switched_that_on(
    capsys: pytest.CaptureFixture[str], library: Settings, model: ScriptedModel
) -> None:
    save_grants(library.data_dir, Grants(enabled=True, tools=frozenset({"search_notes"})))
    model.turns = [
        asks("search_notes", query="how long to simmer oats in milk"),
        says("Five minutes."),
    ]

    code, out, _err = run(capsys, library, "run", "how long do I simmer oats?")

    assert code == 0 and "Five minutes." in out
    observation = model.chats[1][1][-1].content
    assert "Simmer the oats in milk for five minutes." in observation
    assert "oats.md" in observation and "reference data only" in observation


def test_a_task_that_never_searches_never_loads_the_embedding_model(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    model: ScriptedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_load(*_args: object) -> None:
        raise AssertionError("the embedding model was loaded")

    monkeypatch.setattr(cli, "load_embedder", must_not_load)
    save_grants(settings.data_dir, Grants(enabled=True, tools=frozenset({"calculator"})))
    model.turns = [asks("calculator", expression="1+1"), says("Two.")]

    assert run(capsys, settings, "run", "add one and one")[0] == 0


def test_a_missing_embedding_model_is_a_question_to_the_owner_not_a_crash(
    capsys: pytest.CaptureFixture[str],
    settings: Settings,
    model: ScriptedModel,
    typed: list[str],
) -> None:
    save_grants(settings.data_dir, Grants(enabled=True, tools=frozenset({"search_notes"})))
    model.turns = [asks("search_notes", query="oats")]
    typed.append("s")

    code, out, _err = run(capsys, settings, "run", "find oats")

    assert code == 1 and "search_notes failed" in out and "download-model" in out
