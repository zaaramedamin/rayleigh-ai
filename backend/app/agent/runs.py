"""Running the agent in the background, so a screen can watch it and answer its questions.

One task runs at a time, in its own thread. While it runs, a screen can ask what it is doing, and
when the loop needs the owner (approve this action, something went wrong) the run waits, shows the
question, and goes on only when `answer()` is called with one of the offered choices for that very
question. `answer()` is called by the owner's side of the application (a button behind the access
password), never by anything the model writes. A question nobody answers within a time limit
counts as "stop", and so does the owner pressing stop.

Nothing here keeps what a task or a tool said beyond the run's own short record in memory; the audit
log is the lasting record.
"""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.orm import Session

from app.agent import audit
from app.agent.loop import STOP, Limits, Question, RunResult, StepLog, run_agent
from app.agent.permissions import Grants
from app.agent.tools import ToolRegistry
from app.ai.llm.base import LLMProvider

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "waiting", "done", "stopped", "failed"]
MAX_PROGRESS_LINES = 200
KEEP_RUNS = 20
QUESTION_WAIT_SECONDS = 900.0


class RunBusyError(RuntimeError):
    """Another task is already running."""


class StaleQuestionError(RuntimeError):
    """The question being answered is not the one the run is waiting on."""


@dataclass
class Pending:
    id: int
    question: Question


@dataclass
class RunState:
    run_id: str
    task: str
    status: RunStatus = "running"
    answer: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    progress: list[str] = field(default_factory=list)
    pending: Pending | None = None
    steps: tuple[StepLog, ...] = ()
    model_turns: int = 0

    @property
    def finished(self) -> bool:
        return self.status in ("done", "stopped", "failed")


class RunManager:
    """Starts tasks and keeps the state of the last few."""

    def __init__(
        self,
        *,
        make_session: Callable[[], AbstractContextManager[Session]],
        make_llm: Callable[[], LLMProvider],
        make_registry: Callable[[], ToolRegistry],
        load_grants: Callable[[], Grants],
        limits: Limits | None = None,
        question_wait: float = QUESTION_WAIT_SECONDS,
    ) -> None:
        self._make_session = make_session
        self._make_llm = make_llm
        self._make_registry = make_registry
        self._load_grants = load_grants
        self._limits = limits
        self._question_wait = question_wait
        self._changed = threading.Condition()
        self._runs: dict[str, RunState] = {}
        self._stops: dict[str, threading.Event] = {}
        self._answers: dict[str, str] = {}
        self._next_question = 1

    # --- for the screen ------------------------------------------------------------------------

    def start(self, task: str) -> RunState:
        """Begin a task in the background. Raises ValueError for a bad task, RunBusyError."""
        cleaned = " ".join(task.split())
        if not cleaned:
            raise ValueError("say what the agent should do")
        with self._changed:
            if any(not s.finished for s in self._runs.values()):
                raise RunBusyError(
                    "The agent is already working on a task. Wait for it or stop it."
                )
            state = RunState(run_id=audit.new_run_id(), task=cleaned)
            self._runs[state.run_id] = state
            self._stops[state.run_id] = threading.Event()
            self._forget_old()
        threading.Thread(
            target=self._work, args=(state,), daemon=True, name=f"agent-run-{state.run_id[:8]}"
        ).start()
        return self.get(state.run_id)  # type: ignore[return-value]

    def get(self, run_id: str) -> RunState | None:
        """A copy of the run's state, safe to read while the run goes on."""
        with self._changed:
            state = self._runs.get(run_id)
            if state is None:
                return None
            pending = Pending(state.pending.id, state.pending.question) if state.pending else None
            return RunState(
                run_id=state.run_id,
                task=state.task,
                status=state.status,
                answer=state.answer,
                started_at=state.started_at,
                finished_at=state.finished_at,
                progress=list(state.progress),
                pending=pending,
                steps=state.steps,
                model_turns=state.model_turns,
            )

    def active(self) -> RunState | None:
        with self._changed:
            ids = [i for i, s in self._runs.items() if not s.finished]
        return self.get(ids[0]) if ids else None

    def answer(self, run_id: str, question_id: int, choice: str) -> None:
        """The owner's answer to the question the run is waiting on.

        Raises LookupError for an unknown run, StaleQuestionError when the run is not waiting on
        that question (or it was already answered), ValueError when `choice` is not one of the
        question's options.
        """
        with self._changed:
            state = self._runs.get(run_id)
            if state is None:
                raise LookupError("There is no such task.")
            pending = state.pending
            if pending is None or pending.id != question_id or run_id in self._answers:
                # Also when it was answered a moment ago: the first answer is the one that counts.
                raise StaleQuestionError("That question is no longer waiting for an answer.")
            if choice not in pending.question.options:
                raise ValueError(
                    f"The answer must be one of: {', '.join(pending.question.options)}"
                )
            self._answers[run_id] = choice
            self._changed.notify_all()

    def stop(self, run_id: str) -> bool:
        """Ask a running task to stop. A question it is waiting on is answered with stop."""
        with self._changed:
            state = self._runs.get(run_id)
            event = self._stops.get(run_id)
            if state is None or event is None or state.finished:
                return False
            event.set()
            state.progress.append("stopping ...")
            if state.pending is not None:
                self._answers[run_id] = STOP
            self._changed.notify_all()
            return True

    # --- inside the run's thread ------------------------------------------------------------------

    def _forget_old(self) -> None:
        finished = [i for i, s in self._runs.items() if s.finished]
        for run_id in finished[: max(0, len(finished) - KEEP_RUNS)]:
            self._runs.pop(run_id, None)
            self._stops.pop(run_id, None)

    def _say(self, run_id: str, text: str) -> None:
        with self._changed:
            state = self._runs[run_id]
            state.progress.append(text)
            del state.progress[:-MAX_PROGRESS_LINES]

    def _ask(self, run_id: str, question: Question) -> str:
        """Called by the loop: show the question and wait for the owner. Returns their choice."""
        with self._changed:
            state = self._runs[run_id]
            stopped = self._stops[run_id]
            if stopped.is_set():
                return STOP  # nobody is waiting for the question of a task that was stopped
            question_id = self._next_question
            self._next_question += 1
            state.pending = Pending(question_id, question)
            state.status = "waiting"
            self._changed.notify_all()
            deadline = time.monotonic() + self._question_wait
            while run_id not in self._answers and not stopped.is_set():
                left = deadline - time.monotonic()
                if left <= 0 or not self._changed.wait(timeout=left):
                    break
            choice = self._answers.pop(run_id, STOP)
            state.pending = None
            state.status = "running"
            return choice

    def _work(self, state: RunState) -> None:
        run_id = state.run_id
        manager = self

        class Asker:
            def ask(self, question: Question) -> str:
                return manager._ask(run_id, question)

        result: RunResult | None = None
        try:
            with self._make_session() as session:
                result = run_agent(
                    self._make_llm(),
                    self._make_registry(),
                    self._load_grants(),
                    session,
                    Asker(),
                    state.task,
                    limits=self._limits,
                    run_id=run_id,
                    stop=self._stops[run_id],
                    progress=lambda text: self._say(run_id, text),
                )
        except Exception as exc:  # noqa: BLE001 - nothing may leave a run hanging
            logger.error("an agent run ended with an error type=%s", type(exc).__name__)
        with self._changed:
            self._answers.pop(run_id, None)  # an answer nobody picked up must not outlive its task
            state.finished_at = datetime.now(UTC).isoformat()
            state.pending = None
            if result is None:
                state.status = "failed"
                state.answer = "The task could not be carried out. Nothing was changed."
            else:
                state.status = result.status
                state.answer = result.answer
                state.steps = result.steps
                state.model_turns = result.model_turns
            self._changed.notify_all()
