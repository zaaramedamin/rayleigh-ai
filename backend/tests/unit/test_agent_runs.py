"""Running the agent in the background: waiting on questions, answers, stopping, failing safely."""

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.agent import runs as runs_module
from app.agent.audit import run_events
from app.agent.loop import ALLOW, DENY, STOP, Limits
from app.agent.permissions import Grants
from app.agent.runs import RunBusyError, RunManager, RunState, StaleQuestionError
from app.agent.tools import Param, Tool, ToolRegistry
from app.ai.llm.base import ChatMessage, ChatReply, ToolSpec
from app.storage.database import Base, create_db_engine
from tests.fakes import ScriptedModel, asks, says

ON = Grants(enabled=True, tools=frozenset({"echo", "open_thing"}))


@pytest.fixture
def make_session(data_dir: Path) -> Iterator[Callable[[], Session]]:
    engine = create_db_engine(data_dir)
    Base.metadata.create_all(engine)
    yield lambda: Session(engine)
    engine.dispose()


class Things:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.registry = ToolRegistry(
            [
                Tool("echo", "d", "read_local", {"word": Param("string", "w")}, self.echo),
                Tool(
                    "open_thing",
                    "d",
                    "open_local",
                    {"path": Param("string", "p")},
                    self.open_thing,
                    describe=lambda a: f"Open {a['path']}",
                ),
            ]
        )

    def echo(self, args: Mapping[str, Any]) -> str:
        return f"echo: {args['word']}"

    def open_thing(self, args: Mapping[str, Any]) -> str:
        self.opened.append(args["path"])
        return "opened"


class Waiting(ScriptedModel):
    """A model that does not answer until the test lets it."""

    def __init__(self, *turns: ChatReply | Exception) -> None:
        super().__init__(*turns)
        self.release = threading.Event()
        self.asked = threading.Event()

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        tools: Sequence[ToolSpec] = (),
    ) -> ChatReply:
        self.asked.set()
        assert self.release.wait(5), "the test never released the model"
        return super().chat(system, messages, temperature=temperature, tools=tools)


def manager(
    make_session: Callable[[], Session],
    model: Any,
    things: Things | None = None,
    grants: Grants = ON,
    **kwargs: Any,
) -> RunManager:
    things = things or Things()
    return RunManager(
        make_session=make_session,  # type: ignore[arg-type]
        make_llm=lambda: model,
        make_registry=lambda: things.registry,
        load_grants=lambda: grants,
        **kwargs,
    )


def until(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def finished(m: RunManager, run_id: str) -> RunState:
    until(lambda: (m.get(run_id) or RunState("", "")).finished)
    state = m.get(run_id)
    assert state is not None
    return state


def waiting(m: RunManager, run_id: str) -> RunState:
    until(lambda: (m.get(run_id) or RunState("", "")).status == "waiting")
    state = m.get(run_id)
    assert state is not None and state.pending is not None
    return state


# --- a task that needs no questions ------------------------------------------------------------


def test_a_task_runs_in_the_background_and_reports_what_it_did(
    make_session: Callable[[], Session],
) -> None:
    m = manager(make_session, ScriptedModel(asks("echo", word="hi"), says("Said hi.")))

    started = m.start("  say   hi ")
    done = finished(m, started.run_id)

    assert started.status == "running" and started.task == "say hi"
    assert (done.status, done.answer, done.model_turns) == ("done", "Said hi.", 2)
    assert [(s.tool, s.outcome) for s in done.steps] == [("echo", "done")]
    assert done.progress == [
        "thinking ...",
        "running: echo(word='hi')",
        "done: echo",
        "thinking ...",
    ]
    assert done.finished_at is not None and done.pending is None
    with make_session() as session:
        assert [e.kind for e in run_events(session, started.run_id)][0] == "run_started"


def test_an_empty_task_is_refused_before_anything_starts(
    make_session: Callable[[], Session],
) -> None:
    m = manager(make_session, ScriptedModel())

    with pytest.raises(ValueError, match="say what"):
        m.start("   ")
    assert m.active() is None


def test_an_unknown_run_is_none_and_a_copy_cannot_change_the_run(
    make_session: Callable[[], Session],
) -> None:
    m = manager(make_session, ScriptedModel(says("ok")))
    assert m.get("nope") is None

    run_id = m.start("x").run_id
    finished(m, run_id)
    copy = m.get(run_id)
    assert copy is not None
    copy.progress.append("tampered")
    copy.status = "failed"

    again = m.get(run_id)
    assert again is not None and again.status == "done" and "tampered" not in again.progress


def test_a_switched_off_agent_ends_at_once_as_failed(make_session: Callable[[], Session]) -> None:
    model = ScriptedModel()
    m = manager(make_session, model, grants=Grants())

    done = finished(m, m.start("anything").run_id)

    assert done.status == "failed" and "switched off" in done.answer and model.chats == []


# --- a question the owner answers --------------------------------------------------------------


def test_the_run_waits_on_a_question_and_goes_on_when_the_owner_allows(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(
        make_session, ScriptedModel(asks("open_thing", path="C:/x.txt"), says("Opened.")), things
    )

    run_id = m.start("open it").run_id
    state = waiting(m, run_id)

    assert state.pending is not None and state.pending.question.kind == "approve"
    assert state.pending.question.effect == "Open C:/x.txt"
    assert state.pending.question.options == (ALLOW, DENY, STOP)
    time.sleep(0.1)
    assert things.opened == []  # nothing happens while the owner has not answered

    m.answer(run_id, state.pending.id, ALLOW)
    done = finished(m, run_id)

    assert things.opened == ["C:/x.txt"] and done.status == "done" and done.answer == "Opened."


def test_the_owner_can_say_no(make_session: Callable[[], Session]) -> None:
    things = Things()
    m = manager(
        make_session, ScriptedModel(asks("open_thing", path="C:/x.txt"), says("Not done.")), things
    )
    run_id = m.start("open it").run_id
    state = waiting(m, run_id)

    assert state.pending is not None
    m.answer(run_id, state.pending.id, DENY)
    done = finished(m, run_id)

    assert things.opened == [] and done.status == "done"
    assert [s.outcome for s in done.steps] == ["not allowed by you"]


def test_an_answer_must_be_one_of_the_offered_choices_and_the_run_keeps_waiting(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(make_session, ScriptedModel(asks("open_thing", path="C:/x.txt")), things)
    run_id = m.start("open it").run_id
    state = waiting(m, run_id)
    assert state.pending is not None

    for bad in ("retry", "yes", "ALLOW", "", "allow; also stop"):
        with pytest.raises(ValueError, match="must be one of: allow, deny, stop"):
            m.answer(run_id, state.pending.id, bad)

    still = m.get(run_id)
    assert still is not None and still.status == "waiting" and things.opened == []
    m.answer(run_id, state.pending.id, STOP)
    assert finished(m, run_id).status == "stopped"


def test_an_answer_to_the_wrong_question_or_run_is_refused(
    make_session: Callable[[], Session],
) -> None:
    m = manager(make_session, ScriptedModel(asks("open_thing", path="C:/x.txt"), says("Done.")))
    run_id = m.start("open it").run_id
    state = waiting(m, run_id)
    assert state.pending is not None

    with pytest.raises(StaleQuestionError):
        m.answer(run_id, state.pending.id + 100, ALLOW)
    with pytest.raises(LookupError, match="no such task"):
        m.answer("no-such-run", state.pending.id, ALLOW)

    m.answer(run_id, state.pending.id, DENY)
    finished(m, run_id)
    with pytest.raises(StaleQuestionError):  # the same answer again, after the run moved on
        m.answer(run_id, state.pending.id, DENY)


def test_a_question_that_is_answered_twice_only_counts_once(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(
        make_session, ScriptedModel(asks("open_thing", path="C:/x.txt"), says("ok")), things
    )
    run_id = m.start("open it").run_id
    state = waiting(m, run_id)
    assert state.pending is not None

    m.answer(run_id, state.pending.id, DENY)
    with pytest.raises(StaleQuestionError):
        # Straight away, so it lands before the run has woken up and cleared the question.
        m.answer(run_id, state.pending.id, ALLOW)
    finished(m, run_id)

    assert things.opened == []  # the first answer, no, is the one that counted


def test_each_question_has_its_own_number(make_session: Callable[[], Session]) -> None:
    m = manager(
        make_session,
        ScriptedModel(
            asks("open_thing", path="C:/a.txt"), asks("open_thing", path="C:/b.txt"), says("ok")
        ),
    )
    run_id = m.start("open both").run_id

    first = waiting(m, run_id).pending
    assert first is not None
    m.answer(run_id, first.id, DENY)
    until(lambda: (m.get(run_id).pending or first).id != first.id)  # type: ignore[union-attr]
    second = waiting(m, run_id).pending
    assert second is not None and second.id > first.id
    m.answer(run_id, second.id, DENY)
    finished(m, run_id)


# --- one at a time, stopping, timing out --------------------------------------------------------


def test_only_one_task_runs_at_a_time(make_session: Callable[[], Session]) -> None:
    m = manager(make_session, ScriptedModel(asks("open_thing", path="C:/x.txt"), says("ok")))
    run_id = m.start("first").run_id
    state = waiting(m, run_id)
    assert state.pending is not None

    with pytest.raises(RunBusyError, match="already working"):
        m.start("second")
    assert m.active() is not None and m.active().run_id == run_id  # type: ignore[union-attr]

    m.answer(run_id, state.pending.id, DENY)
    finished(m, run_id)
    assert m.active() is None
    m2 = m.start("third")  # free again
    assert m2.run_id != run_id


def test_stopping_a_task_that_is_waiting_for_the_owner_ends_it_without_acting(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(make_session, ScriptedModel(asks("open_thing", path="C:/x.txt")), things)
    run_id = m.start("open it").run_id
    waiting(m, run_id)

    assert m.stop(run_id) is True
    done = finished(m, run_id)

    assert (
        done.status == "stopped" and things.opened == [] and "You stopped the task." in done.answer
    )


def test_stopping_a_task_that_is_thinking_ends_it_at_the_next_step(
    make_session: Callable[[], Session],
) -> None:
    model = Waiting(asks("echo", word="hi"), says("never reached"))
    m = manager(make_session, model)
    run_id = m.start("x").run_id
    assert model.asked.wait(5)

    assert m.stop(run_id) is True
    model.release.set()
    done = finished(m, run_id)

    assert done.status == "stopped" and "You stopped the task." in done.answer


def test_stopping_a_finished_or_unknown_task_does_nothing(
    make_session: Callable[[], Session],
) -> None:
    m = manager(make_session, ScriptedModel(says("ok")))
    run_id = m.start("x").run_id
    finished(m, run_id)

    assert m.stop(run_id) is False and m.stop("nope") is False


def test_a_question_nobody_answers_ends_the_task_as_stopped(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(
        make_session,
        ScriptedModel(asks("open_thing", path="C:/x.txt")),
        things,
        question_wait=0.2,
    )

    done = finished(m, m.start("open it").run_id)

    assert done.status == "stopped" and things.opened == []


# --- failing safely --------------------------------------------------------------------------


def test_a_run_that_breaks_is_failed_with_a_fixed_sentence_and_no_leak(
    make_session: Callable[[], Session], caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> Any:
        raise RuntimeError("Quillon-Marmalade-4821 in a traceback")

    m = RunManager(
        make_session=make_session,  # type: ignore[arg-type]
        make_llm=broken,
        make_registry=lambda: Things().registry,
        load_grants=lambda: ON,
    )

    with caplog.at_level(logging.DEBUG):
        done = finished(m, m.start("x").run_id)

    assert done.status == "failed" and "Nothing was changed" in done.answer
    assert "Quillon-Marmalade-4821" not in done.answer + caplog.text
    assert "ended with an error type=RuntimeError" in caplog.text
    assert m.active() is None  # a failed run does not block the next one


# --- housekeeping ----------------------------------------------------------------------------


def test_only_the_latest_finished_runs_are_kept(
    make_session: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_module, "KEEP_RUNS", 2)
    m = manager(make_session, ScriptedModel(*[says("ok")] * 5))
    ids = []
    for _ in range(5):
        run_id = m.start("x").run_id
        finished(m, run_id)
        ids.append(run_id)

    kept = [i for i in ids if m.get(i) is not None]
    assert kept == ids[-3:]  # the two kept, and the one just started


def test_the_progress_shown_is_limited(
    make_session: Callable[[], Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_module, "MAX_PROGRESS_LINES", 3)
    m = manager(
        make_session,
        ScriptedModel(*[asks("echo", word="x")] * 4, says("end")),
        limits=Limits(max_steps=20),
    )

    done = finished(m, m.start("x").run_id)

    assert len(done.progress) == 3 and done.progress[-1] == "thinking ..."


# --- a stop is never lost ------------------------------------------------------------------------


def test_a_question_asked_after_a_stop_is_not_shown_and_does_not_wait(
    make_session: Callable[[], Session],
) -> None:
    from app.agent.loop import Question

    model = Waiting(says("never"))
    m = manager(make_session, model)
    run_id = m.start("x").run_id
    assert model.asked.wait(5)
    m.stop(run_id)  # stop arrives while the model is still thinking

    started = time.monotonic()
    answer = m._ask(run_id, Question("approve", "t", "m", (ALLOW, DENY, STOP)))  # noqa: SLF001

    assert answer == STOP and time.monotonic() - started < 1  # not the fifteen minutes
    state = m.get(run_id)
    assert state is not None and state.pending is None and state.status == "running"
    model.release.set()
    assert finished(m, run_id).status == "stopped"


def test_a_stop_is_shown_as_received_while_the_task_winds_down(
    make_session: Callable[[], Session],
) -> None:
    model = Waiting(says("never"))
    m = manager(make_session, model)
    run_id = m.start("x").run_id
    assert model.asked.wait(5)

    m.stop(run_id)

    state = m.get(run_id)
    assert state is not None and state.progress[-1] == "stopping ..."
    model.release.set()
    assert finished(m, run_id).status == "stopped"


def test_an_answer_nobody_picked_up_does_not_leak_into_the_next_task(
    make_session: Callable[[], Session],
) -> None:
    things = Things()
    m = manager(make_session, ScriptedModel(asks("open_thing", path="C:/x.txt")), things)
    run_id = m.start("open it").run_id
    state = waiting(m, run_id)
    assert state.pending is not None
    m.stop(run_id)
    finished(m, run_id)

    assert run_id not in m._answers  # noqa: SLF001


def test_the_facts_about_the_computer_are_given_to_every_task(
    make_session: Callable[[], Session],
) -> None:
    model = ScriptedModel(says("ok"))
    m = manager(
        make_session,
        model,
        make_context=lambda: "About this computer:\n- The user's Downloads folder is C:/D.",
    )

    finished(m, m.start("open my downloads").run_id)

    assert model.chats[0][1][0].content.endswith("The user's request:\nopen my downloads")
    assert "C:/D." in model.chats[0][1][0].content


def test_a_task_is_read_first_when_the_manager_is_set_to_do_so(
    make_session: Callable[[], Session],
) -> None:
    import json

    model = ScriptedModel(says("Voilà."))
    model.readings = [
        json.dumps(
            {"language": "French", "request": "Calcule 7 fois 8.", "clear": True, "question": ""}
        )
    ]
    m = manager(make_session, model, interpret=True)

    done = finished(m, m.start("calcul 7*8 stp").run_id)

    assert "Understood as: Calcule 7 fois 8." in model.chats[0][1][0].content
    assert "Write your reply in French." in model.chats[0][1][0].content
    assert "understood as: Calcule 7 fois 8." in done.progress


def test_an_unclear_task_ends_with_the_question_for_the_owner(
    make_session: Callable[[], Session],
) -> None:
    import json

    model = ScriptedModel()
    model.readings = [
        json.dumps(
            {
                "language": "English",
                "request": "Open it.",
                "clear": False,
                "question": "Which file?",
            }
        )
    ]
    m = manager(make_session, model, interpret=True)

    done = finished(m, m.start("open it").run_id)

    assert (done.status, done.answer) == ("done", "Which file?") and model.chats == []


def test_a_manager_that_was_not_asked_to_read_tasks_does_not(
    make_session: Callable[[], Session],
) -> None:
    model = ScriptedModel(says("ok"))

    m = manager(make_session, model)
    finished(m, m.start("hello").run_id)

    assert model.read == []
