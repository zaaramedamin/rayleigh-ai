"""`python -m app agent`: the owner's cards and commands, without the real command line."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.agent import console
from app.agent.audit import recent_events
from app.agent.console import (
    ConsoleError,
    Environment,
    LazySearch,
    TerminalApprover,
    agent_command,
)
from app.agent.loop import ALLOW, CONTINUE, DENY, RETRY, SKIP, STOP, Question
from app.agent.permissions import Grants, load_grants, save_grants
from app.agent.tools import Tool, ToolError, ToolRegistry
from app.ai.llm.base import LLMUnavailableError
from tests.fakes import ScriptedModel, asks, says

# --- the cards ------------------------------------------------------------------------------


def approver(*typed: str | type[BaseException]) -> tuple[TerminalApprover, list[str], list[str]]:
    answers = list(typed)
    shown: list[str] = []
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        answer = answers.pop(0)
        if isinstance(answer, type):
            raise answer()
        return answer

    return TerminalApprover(read, shown.append), shown, prompts


APPROVE = Question(
    "approve",
    "Allow open_thing?",
    "Open C:/notes/plan.txt",
    (ALLOW, DENY, STOP),
    tool="open_thing",
    arguments={"path": "C:/notes/plan.txt"},
    effect="Open C:/notes/plan.txt",
    reason="This opens something on your computer, so I need your approval each time.",
    read_sources=("search_notes", "echo"),
)
PROBLEM = Question(
    "problem", "Something went wrong", "shaky failed: the disk did not answer", (RETRY, SKIP, STOP)
)


def test_the_approval_card_shows_exactly_what_will_happen_and_why_it_asks() -> None:
    terminal, shown, _ = approver("s")

    terminal.ask(APPROVE)

    card = "\n".join(shown)
    assert "The agent wants your approval" in card
    assert "Open C:/notes/plan.txt" in card
    assert "Exactly: path='C:/notes/plan.txt'" in card
    assert "Why you are asked: This opens something on your computer" in card
    assert "The model has read the results of: search_notes, echo" in card
    assert "[a] allow once   [d] do not allow   [s] stop the task" in card


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("a", ALLOW),
        ("A", ALLOW),
        ("  allow  ", ALLOW),
        ("y", ALLOW),
        ("yes", ALLOW),
        ("d", DENY),
        ("n", DENY),
        ("no", DENY),
        ("deny", DENY),
        ("s", STOP),
        ("STOP", STOP),
    ],
)
def test_one_key_or_word_answers_the_card(typed: str, expected: str) -> None:
    terminal, _, _ = approver(typed)

    assert terminal.ask(APPROVE) == expected


def test_a_problem_card_offers_its_own_choices_and_nothing_else() -> None:
    terminal, shown, _ = approver("r")

    assert terminal.ask(PROBLEM) == RETRY
    assert "Something went wrong" in "\n".join(shown)
    assert "[r] try again   [k] skip it   [s] stop the task" in "\n".join(shown)
    assert "Exactly:" not in "\n".join(shown) and "Why you are asked" not in "\n".join(shown)


def test_a_choice_the_card_does_not_offer_is_not_accepted() -> None:
    # "a" and "yes" mean allow, which a problem card does not offer.
    terminal, shown, prompts = approver("a", "yes", "k")

    assert terminal.ask(PROBLEM) == SKIP
    assert len(prompts) == 3 and shown.count("Please press one of the keys shown above.") == 2


def test_three_wrong_answers_in_a_row_mean_stop() -> None:
    terminal, shown, _ = approver("?", "", "maybe")

    assert terminal.ask(APPROVE) == STOP
    assert shown[-1] == "(no valid answer: stopping)"


@pytest.mark.parametrize("closed", [EOFError, KeyboardInterrupt])
def test_a_closed_terminal_or_ctrl_c_means_stop(closed: type[BaseException]) -> None:
    terminal, shown, _ = approver(closed)

    assert terminal.ask(APPROVE) == STOP
    assert shown[-1] == "(no answer: stopping)"


def test_continue_is_available_on_the_limit_card() -> None:
    terminal, shown, _ = approver("c")
    limit = Question(
        "problem", "Something went wrong", "The task used its 8 steps.", (CONTINUE, STOP)
    )

    assert terminal.ask(limit) == CONTINUE
    assert "[c] let it go on   [s] stop the task" in "\n".join(shown)


# --- the commands ---------------------------------------------------------------------------


class Screen:
    def __init__(self, *typed: str) -> None:
        self.lines: list[str] = []
        self.typed = list(typed)
        self.progress: list[str] = []

    def read(self, _prompt: str) -> str:
        return self.typed.pop(0)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def make_env(
    session: Session,
    data_dir: Path,
    screen: Screen,
    model: ScriptedModel | Callable[[], Any] | None = None,
) -> Environment:
    def make_llm() -> Any:
        if callable(model):
            return model()
        return model or ScriptedModel()

    return Environment(
        data_dir=data_dir,
        session=session,
        make_llm=make_llm,
        search=None,
        min_score=0.3,
        emit=screen.lines.append,
        progress=screen.progress.append,
        input_fn=screen.read,
    )


def command(env: Environment, action: str, *words: str, **options: Any) -> int:
    return agent_command(env, action, list(words), **options)


def test_the_status_of_a_fresh_install_says_everything_is_off(
    session: Session, tmp_path: Path
) -> None:
    screen = Screen()

    assert command(make_env(session, tmp_path, screen), "status") == 0

    assert screen.lines[0] == "agent: OFF" and screen.lines[1] == "web access: off"
    assert "calculator       off  only reads" in screen.text
    assert "current_time     off  only reads" in screen.text
    assert "search_notes     off  only reads" in screen.text
    assert "does nothing until you run `python -m app agent enable`" in screen.text


def test_switching_the_agent_on_and_off_is_saved_and_explained(
    session: Session, tmp_path: Path
) -> None:
    screen = Screen()
    env = make_env(session, tmp_path, screen)

    command(env, "enable")
    assert load_grants(tmp_path).enabled and "asks you every time" in screen.text
    command(env, "disable")
    assert not load_grants(tmp_path).enabled and "will not run anything" in screen.text


def test_switching_the_agent_on_does_not_switch_any_tool_on(
    session: Session, tmp_path: Path
) -> None:
    command(make_env(session, tmp_path, Screen()), "enable")

    assert load_grants(tmp_path) == Grants(enabled=True)


def test_tools_are_switched_on_and_off_one_by_one(session: Session, tmp_path: Path) -> None:
    screen = Screen()
    env = make_env(session, tmp_path, screen)

    command(env, "allow", "calculator", "current_time")
    assert load_grants(tmp_path).tools == {"calculator", "current_time"}
    assert "calculator: on (only reads)" in screen.text
    assert "The agent itself is still off" in screen.text  # it was never enabled

    command(env, "revoke", "calculator")
    assert load_grants(tmp_path).tools == {"current_time"}
    assert "calculator: off" in screen.text


def test_a_tool_that_does_not_exist_or_no_tool_at_all_is_an_error_naming_the_real_ones(
    session: Session, tmp_path: Path
) -> None:
    env = make_env(session, tmp_path, Screen())

    with pytest.raises(ConsoleError, match="no such tool: delete_all. The tools are: calculator"):
        command(env, "allow", "delete_all")
    with pytest.raises(ConsoleError, match="name at least one tool"):
        command(env, "allow")
    assert load_grants(tmp_path) == Grants()


def test_a_destructive_tool_cannot_be_switched_on(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wipe = Tool("wipe", "d", "destructive", {}, lambda _a: "")
    monkeypatch.setattr(console, "default_registry", lambda **_k: ToolRegistry([wipe]))

    with pytest.raises(ConsoleError, match="destructive actions are never allowed"):
        command(make_env(session, tmp_path, Screen()), "allow", "wipe")
    assert load_grants(tmp_path).tools == frozenset()


def test_a_tool_that_reaches_the_web_says_it_needs_the_web_switch(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = Tool("fetch_web_page", "d", "external_read", {}, lambda _a: "")
    monkeypatch.setattr(console, "default_registry", lambda **_k: ToolRegistry([fetch]))
    screen = Screen()

    command(make_env(session, tmp_path, screen), "allow", "fetch_web_page")

    assert "it also needs `python -m app agent web on`" in screen.text


def test_the_web_has_its_own_switch_and_it_changes_nothing_else(
    session: Session, tmp_path: Path
) -> None:
    save_grants(tmp_path, Grants(enabled=True, tools=frozenset({"calculator"}), web=False))
    screen = Screen()
    env = make_env(session, tmp_path, screen)

    command(env, "web", "on")
    assert load_grants(tmp_path) == Grants(enabled=True, tools=frozenset({"calculator"}), web=True)
    assert "still asks you every time" in screen.text
    command(env, "web", "off")
    assert not load_grants(tmp_path).web
    with pytest.raises(ConsoleError, match="web on"):
        command(env, "web", "maybe")


def test_an_unknown_action_is_an_error(session: Session, tmp_path: Path) -> None:
    with pytest.raises(ConsoleError, match="unknown action 'fly'"):
        command(make_env(session, tmp_path, Screen()), "fly")


# --- running a task ---------------------------------------------------------------------------


def switched_on(tmp_path: Path, *tools: str) -> None:
    save_grants(tmp_path, Grants(enabled=True, tools=frozenset(tools)))


def test_a_task_is_run_and_the_answer_and_the_steps_are_shown(
    session: Session, tmp_path: Path
) -> None:
    switched_on(tmp_path, "calculator")
    model = ScriptedModel(asks("calculator", expression="12*12"), says("It is 144."))
    screen = Screen()

    code = command(make_env(session, tmp_path, screen, model), "run", "what", "is", "12*12?")

    assert code == 0
    assert "It is 144." in screen.lines
    assert "What it did:" in screen.lines and "  Calculate 12*12: done" in screen.lines
    assert screen.lines[-1].startswith("(run ") and "agent log --run" in screen.lines[-1]
    assert screen.progress[:2] == ["thinking ...", "running: Calculate 12*12"]
    assert [m.content for m in model.chats[0][1]] == ["what is 12*12?"]


def test_the_owner_answers_the_agents_questions_on_the_keyboard(
    session: Session, tmp_path: Path
) -> None:
    switched_on(tmp_path, "calculator")
    # Allow level tools never ask, so make the model fail once and answer the question.
    model = ScriptedModel(LLMUnavailableError("Ollama is not running"), says("Back."))
    screen = Screen("r")

    code = command(make_env(session, tmp_path, screen, model), "run", "hello")

    assert code == 0 and "Something went wrong" in screen.text
    assert "The local model failed: Ollama is not running" in screen.text


def test_a_task_the_owner_stopped_exits_with_an_error_code(
    session: Session, tmp_path: Path
) -> None:
    switched_on(tmp_path)
    model = ScriptedModel(LLMUnavailableError("down"))

    code = command(make_env(session, tmp_path, Screen("s"), model), "run", "hello")

    assert code == 1


def test_a_task_cannot_start_while_the_agent_is_off_and_the_model_is_not_called(
    session: Session, tmp_path: Path
) -> None:
    model = ScriptedModel()

    with pytest.raises(ConsoleError, match="the agent is off"):
        command(make_env(session, tmp_path, Screen(), model), "run", "hello")
    assert model.chats == []


def test_a_task_needs_words_and_not_too_many(session: Session, tmp_path: Path) -> None:
    switched_on(tmp_path)
    env = make_env(session, tmp_path, Screen())

    with pytest.raises(ConsoleError, match="say what the agent should do"):
        command(env, "run")
    with pytest.raises(ConsoleError, match="too long"):
        command(env, "run", "x" * 2001)


def test_a_model_that_cannot_be_created_is_an_error_not_a_crash(
    session: Session, tmp_path: Path
) -> None:
    switched_on(tmp_path)

    def broken() -> Any:
        raise LLMUnavailableError("Ollama is not running")

    with pytest.raises(ConsoleError, match="Ollama is not running"):
        command(make_env(session, tmp_path, Screen(), broken), "run", "hello")


# --- the log ----------------------------------------------------------------------------------


def test_the_log_shows_what_the_agent_did_in_order(session: Session, tmp_path: Path) -> None:
    switched_on(tmp_path, "calculator")
    model = ScriptedModel(asks("calculator", expression="2+2"), says("Four."))
    run_screen = Screen()
    command(make_env(session, tmp_path, run_screen, model), "run", "add")
    screen = Screen()

    command(make_env(session, tmp_path, screen), "log")

    kinds = [line.split()[4] for line in screen.lines]
    assert kinds == ["run_started", "request", "decision", "result", "run_finished"]
    assert any("calculator" in line and "allow/policy" in line for line in screen.lines)
    assert '{"expression": "2+2"}' in screen.text


def test_one_run_can_be_picked_and_an_empty_log_says_so(session: Session, tmp_path: Path) -> None:
    empty = Screen()
    command(make_env(session, tmp_path, empty), "log")
    assert empty.lines == ["the agent has not done anything yet."]

    switched_on(tmp_path)
    command(make_env(session, tmp_path, Screen(), ScriptedModel(says("a"))), "run", "first")
    command(make_env(session, tmp_path, Screen(), ScriptedModel(says("b"))), "run", "second")
    first = recent_events(session)[-1].run_id
    one, nothing = Screen(), Screen()

    command(make_env(session, tmp_path, one), "log", run_id=first)
    command(make_env(session, tmp_path, nothing), "log", run_id="no-such-run")

    assert len(one.lines) == 2 and all(first[:8] in line for line in one.lines)
    assert nothing.lines == ["no such run in the log."]


def test_the_log_can_be_exported_as_json(session: Session, tmp_path: Path) -> None:
    switched_on(tmp_path, "calculator")
    command(
        make_env(
            session,
            tmp_path,
            Screen(),
            ScriptedModel(asks("calculator", expression="1+1"), says("2")),
        ),
        "run",
        "add",
    )
    target = tmp_path / "log.json"
    screen = Screen()

    command(make_env(session, tmp_path, screen), "log", export=str(target))

    rows = json.loads(target.read_text(encoding="utf-8"))
    assert [r["kind"] for r in rows] == [
        "run_started",
        "request",
        "decision",
        "result",
        "run_finished",
    ]
    assert rows[1]["tool"] == "calculator" and rows[2]["decided_by"] == "policy"
    assert f"wrote {len(rows)} entries" in screen.text


@pytest.mark.parametrize(
    ("typed", "erased"), [("yes", True), ("YES", True), ("no", False), ("", False)]
)
def test_erasing_the_log_needs_a_typed_yes(
    session: Session, tmp_path: Path, typed: str, erased: bool
) -> None:
    switched_on(tmp_path)
    command(make_env(session, tmp_path, Screen(), ScriptedModel(says("a"))), "run", "x")
    screen = Screen(typed)

    command(make_env(session, tmp_path, screen), "log", erase=True)

    assert (recent_events(session) == []) is erased
    assert ("erased 2 entries." in screen.text) is erased and (
        "nothing erased." in screen.text
    ) is not erased


def test_erasing_with_yes_does_not_ask(session: Session, tmp_path: Path) -> None:
    switched_on(tmp_path)
    command(make_env(session, tmp_path, Screen(), ScriptedModel(says("a"))), "run", "x")

    command(make_env(session, tmp_path, Screen()), "log", erase=True, yes=True)

    assert recent_events(session) == []


# --- the search, loaded only when asked for ---------------------------------------------------


def test_the_embedding_model_is_not_loaded_until_the_agent_searches() -> None:
    loads: list[int] = []
    search = LazySearch(lambda: loads.append(1) or "embedder", lambda e, q, n: [])  # type: ignore[func-returns-value]

    assert loads == []
    search("oats", 3)
    search("rice", 3)
    assert loads == [1]  # once


def test_the_search_passes_the_query_and_count_to_the_retrieval() -> None:
    seen: list[tuple[Any, str, int]] = []
    search = LazySearch(lambda: "the-embedder", lambda e, q, n: seen.append((e, q, n)) or [])  # type: ignore[func-returns-value]

    search("oats", 3)

    assert seen == [("the-embedder", "oats", 3)]


def test_a_search_that_fails_becomes_a_tool_error_with_a_pointer_and_no_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def load() -> Any:
        raise OSError("a program blocked by policy")

    search = LazySearch(load, lambda e, q, n: [])

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(ToolError, match="OSError: a program blocked.*doctor"),
    ):
        search("oats", 3)
    assert "search failed type=OSError" in caplog.text and "blocked by policy" not in caplog.text


def test_a_tool_error_from_the_search_is_passed_on_as_it_is() -> None:
    def load() -> Any:
        raise ToolError("The embedding model is missing: run `python -m app download-model`.")

    search = LazySearch(load, lambda e, q, n: [])

    with pytest.raises(ToolError, match="download-model"):
        search("oats", 3)
