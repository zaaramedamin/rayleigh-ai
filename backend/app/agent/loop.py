"""The agent loop: think, ask for a tool, check it, run it, look at the result, repeat.

The model only ever *asks*. Everything that matters is decided here, in code the model cannot
influence:

- A request is checked against the tool's declared parameters (tools.py) and then against the
  owner's grants (permissions.py). A tool that is unknown, switched off, or malformed is refused and
  the model is told why.
- Where the policy says "ask", the run stops and asks the owner through an `Approver`: a button in
  the interface, an answer typed in a terminal. The model's words never reach it, so no text the
  model writes can approve anything. An answer that is not one of the offered options means stop.
- Whenever something goes wrong (the model fails, a tool fails or takes too long, the model keeps
  asking for what cannot be done, a limit is reached) the run stops and asks the owner what to do:
  try again, skip it, or stop. It never retries silently and never improvises.
- Every request, decision, approval, result and problem is written to the audit log before anything
  runs. If the log cannot be written the run stops: nothing runs without a record.
- What a tool brings back goes to the model fenced as untrusted data, and the owner's approval card
  says which tools' results the model has read so far.

The loop has hard limits (steps, time, a tool's own time) and a clean stop that says what was done.
"""

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from sqlalchemy.orm import Session

from app.agent import audit
from app.agent.audit import AuditError
from app.agent.permissions import Grants, decide
from app.agent.tools import (
    ArgumentError,
    Tool,
    ToolError,
    ToolRegistry,
    cap_result,
    fence_result,
    validate_arguments,
)
from app.ai.llm.base import ChatMessage, LLMError, LLMProvider, ToolCall, ToolSpec
from app.ai.llm.budget import FRAMING_CHARS, input_chars
from app.knowledge.answering.prompts import AGENT

logger = logging.getLogger(__name__)

MAX_GOAL_CHARS = 2000
MAX_CONTEXT_CHARS = 2000
TEMPERATURE = 0.0

Status = Literal["done", "stopped", "failed"]
QuestionKind = Literal["approve", "problem"]

ALLOW, DENY, STOP = "allow", "deny", "stop"
RETRY, SKIP, CONTINUE = "retry", "skip", "continue"


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8  # model turns before the owner is asked whether to go on
    max_seconds: float = 300.0  # the same, for the whole run
    tool_seconds: float = 60.0  # one tool call
    max_failures_in_a_row: int = 3  # requests refused or invalid, back to back
    max_model_retries: int = 5  # times the owner may tell the run to try the model again
    max_extensions: int = 2  # times the owner may let a stopped run go on


@dataclass(frozen=True)
class Question:
    """What the owner is asked when the run cannot go on without them."""

    kind: QuestionKind
    title: str
    message: str  # plain words: what it wants to do, or what went wrong
    options: tuple[str, ...]  # the only answers that mean anything
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)  # the exact arguments, as checked
    effect: str = ""
    reason: str = ""  # why the owner is being asked
    read_sources: tuple[str, ...] = ()  # tools whose results the model has read in this run


class Approver(Protocol):
    """Whoever answers the owner's questions: the interface, or a terminal. Never the model."""

    def ask(self, question: Question) -> str:
        """Block until the owner answers. Return one of `question.options`."""
        ...


@dataclass(frozen=True)
class StepLog:
    tool: str
    effect: str
    outcome: str  # done, refused, not allowed by you, failed, skipped


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: Status
    answer: str  # what to tell the owner
    model_turns: int
    steps: tuple[StepLog, ...]


class _Finish(Exception):
    """Ends the run now, with this status and message."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _spec_chars(spec: ToolSpec) -> int:
    return len(spec.name) + len(spec.description) + len(json.dumps(spec.parameters))


def first_message(goal: str, context: str = "") -> str:
    """What the model is first told: the facts it may use, then the request, last, so it is what the
    model answers."""
    facts = context.strip()[:MAX_CONTEXT_CHARS]
    if not facts:
        return goal
    return f"{facts}\n\nThe user's request:\n{goal}"


def fit_messages(messages: Sequence[ChatMessage], room: int) -> list[ChatMessage]:
    """The goal and the latest message are never cut; older turns go first, in pairs.

    Raises ValueError when those two alone do not fit.
    """
    kept = list(messages)

    def size(items: Sequence[ChatMessage]) -> int:
        return sum(len(m.content) + FRAMING_CHARS for m in items)

    while len(kept) > 2 and size(kept) > room:
        del kept[1:3]  # the oldest request and its result
    if size(kept) > room:
        raise ValueError("the task and the latest result are too long for the model to read")
    return kept


class _Run:
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        grants: Grants,
        session: Session,
        approver: Approver,
        limits: Limits,
        run_id: str,
        stop: threading.Event | None,
        clock: Callable[[], float],
        progress: Callable[[str], None] | None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.grants = grants
        self.session = session
        self.approver = approver
        self.limits = limits
        self.run_id = run_id
        self.stop = stop
        self.clock = clock
        self.progress = progress
        self.started = clock()
        self.messages: list[ChatMessage] = []
        self.turns = 0
        self.step_cap = limits.max_steps
        self.deadline = limits.max_seconds
        self.failures = 0
        self.model_retries = 0
        self.extensions = 0
        self.read_sources: list[str] = []
        self.steps: list[StepLog] = []
        self.closed = False  # the log has its run_finished entry

    # --- helpers ---------------------------------------------------------------------------

    def log(self, kind: str, **fields: Any) -> None:
        audit.record(self.session, self.run_id, kind, step=self.turns, **fields)

    def ask(self, question: Question) -> str:
        """Ask the owner. Anything but an offered option, or a failing approver, means stop. A task
        that was stopped is never put to the owner: nobody is waiting for its question."""
        if self.stop is not None and self.stop.is_set():
            return STOP
        try:
            answer = self.approver.ask(question)
        except Exception:  # noqa: BLE001 - the approver is outside code; failing means stop
            logger.warning(
                "the agent could not reach the owner with a question kind=%s", question.kind
            )
            return STOP
        return answer if answer in question.options else STOP

    def problem(self, message: str, options: tuple[str, ...]) -> str:
        self.log("problem", detail=audit.summarize_result(message))
        answer = self.ask(Question("problem", "Something went wrong", message, options))
        self.log(
            "approval",
            decision="allow" if answer != STOP else "deny",
            decided_by="owner",
            detail=f"problem: {answer}",
        )
        return answer

    def check_limits(self) -> None:
        if self.stop is not None and self.stop.is_set():
            raise _Finish("stopped", "You stopped the task.")
        over = None
        if self.turns >= self.step_cap:
            over = f"The task has used its {self.step_cap} steps without finishing."
        elif self.clock() - self.started >= self.deadline:
            over = f"The task has been running for {int(self.deadline)} seconds without finishing."
        if over is None:
            return
        if self.extensions >= self.limits.max_extensions:
            raise _Finish("stopped", f"{over} I stopped, as it has already been extended.")
        if self.problem(f"{over} Let it go on?", (CONTINUE, STOP)) != CONTINUE:
            raise _Finish("stopped", over)
        self.extensions += 1
        self.step_cap += self.limits.max_steps
        self.deadline += self.limits.max_seconds

    def too_many_failures(self) -> None:
        if self.failures < self.limits.max_failures_in_a_row:
            return
        message = (
            f"The model has made {self.failures} requests in a row that could not be carried "
            "out. Let it keep trying?"
        )
        if (
            self.extensions >= self.limits.max_extensions
            or self.problem(message, (CONTINUE, STOP)) != CONTINUE
        ):
            raise _Finish(
                "stopped",
                "I stopped because the model kept making requests that could not be carried out.",
            )
        self.extensions += 1
        self.failures = 0

    # --- thinking ----------------------------------------------------------------------------

    def offered(self) -> list[ToolSpec]:
        names = [n for n in self.registry.names() if self.allowed_to_offer(n)]
        return self.registry.specs(names)

    def allowed_to_offer(self, name: str) -> bool:
        tool = self.registry.get(name)
        return tool is not None and decide(tool, self.grants).decision != "deny"

    def say(self, text: str) -> None:
        """Tell whoever is watching what is happening. A watcher that fails changes nothing."""
        if self.progress is None:
            return
        try:
            self.progress(text)
        except Exception as exc:  # noqa: BLE001 - a broken display must not stop the run
            logger.warning("the agent's progress display failed type=%s", type(exc).__name__)

    def think(self) -> Any:
        self.say("thinking ...")
        specs = self.offered()
        limit: int = getattr(self.llm, "input_chars", input_chars())
        room = limit - len(AGENT.system) - sum(_spec_chars(s) for s in specs)
        while True:
            try:
                fitted = fit_messages(self.messages, room)
                return self.llm.chat(AGENT.system, fitted, temperature=TEMPERATURE, tools=specs)
            except ValueError as exc:
                raise _Finish("failed", f"I could not go on: {exc}.") from exc
            except LLMError as exc:
                if self.model_retries >= self.limits.max_model_retries:
                    raise _Finish("failed", f"The local model keeps failing: {exc}") from exc
                answer = self.problem(f"The local model failed: {exc}", (RETRY, STOP))
                if answer != RETRY:
                    raise _Finish(
                        "stopped", f"The local model failed ({exc}) and you chose to stop."
                    ) from exc
                self.model_retries += 1

    # --- acting ------------------------------------------------------------------------------

    def act(self, call: ToolCall) -> str:
        """Carry out one request for a tool. Returns what the model is told about it."""
        name = call.name[:40]
        arguments = call.arguments if isinstance(call.arguments, Mapping) else {}
        self.log("request", tool=name, detail=audit.redact_arguments(arguments))
        tool = self.registry.get(call.name)
        if tool is None:
            self.log(
                "decision", tool=name, decision="deny", decided_by="policy", detail="no such tool"
            )
            self.failures += 1
            self.steps.append(StepLog(name, "A tool that does not exist", "refused"))
            available = ", ".join(self.registry.names()) or "none"
            return (
                f"The application refused the request: there is no tool called {name!r}. "
                f"The tools are: {available}."
            )
        try:
            checked = validate_arguments(tool, arguments)
        except ArgumentError as exc:
            self.log(
                "decision",
                tool=tool.name,
                level=tool.level,
                decision="deny",
                decided_by="policy",
                detail=str(exc),
            )
            self.failures += 1
            self.steps.append(StepLog(tool.name, tool.name, "refused"))
            return f"The application refused the request: {exc}."

        effect = tool.effect(checked)
        while True:  # again only when the owner says "try again"
            verdict = decide(tool, self.grants)
            if verdict.decision == "deny":
                self.log(
                    "decision",
                    tool=tool.name,
                    level=tool.level,
                    decision="deny",
                    decided_by="policy",
                    detail=verdict.reason,
                )
                self.failures += 1
                self.steps.append(StepLog(tool.name, effect, "refused"))
                return f"The application refused the request: {verdict.reason}"
            self.log(
                "decision",
                tool=tool.name,
                level=tool.level,
                decision=verdict.decision,
                decided_by="policy",
                detail=verdict.reason,
            )
            if verdict.decision == "ask":
                question = Question(
                    "approve",
                    f"Allow {tool.name}?",
                    effect,
                    (ALLOW, DENY, STOP),
                    tool=tool.name,
                    arguments=checked,
                    effect=effect,
                    reason=verdict.reason,
                    read_sources=tuple(dict.fromkeys(self.read_sources)),
                )
                answer = self.ask(question)
                self.log(
                    "approval",
                    tool=tool.name,
                    level=tool.level,
                    decision="allow" if answer == ALLOW else "deny",
                    decided_by="owner",
                    detail=answer,
                )
                if answer == STOP:
                    raise _Finish("stopped", "You stopped the task.")
                if answer != ALLOW:
                    self.steps.append(StepLog(tool.name, effect, "not allowed by you"))
                    return (
                        "The owner did not allow this request. Do not ask for it again; "
                        "say what you could not do."
                    )
            self.say(f"running: {effect}")
            try:
                text = self.run_tool(tool, checked)
            except ToolError as exc:
                answer = self.problem(f"{tool.name} failed: {exc}", (RETRY, SKIP, STOP))
                if answer == RETRY:
                    continue
                if answer == SKIP:
                    self.steps.append(StepLog(tool.name, effect, "skipped"))
                    return (
                        f"The tool {tool.name} failed and the owner chose to skip it. "
                        "Do not use it again for this."
                    )
                raise _Finish(
                    "stopped", f"{tool.name} failed ({exc}) and you chose to stop."
                ) from exc
            capped, cut = cap_result(tool, text)
            self.log(
                "result", tool=tool.name, level=tool.level, detail=audit.summarize_result(capped)
            )
            self.steps.append(StepLog(tool.name, effect, "done"))
            self.say(f"done: {tool.name}")
            self.failures = 0
            self.read_sources.append(tool.name)
            note = "\n(The result was cut: it was longer than the tool returns.)" if cut else ""
            return fence_result(tool.name, capped + note, secrets.token_hex(8))

    def run_tool(self, tool: Tool, arguments: Mapping[str, Any]) -> str:
        """Run a tool with its own time limit. Raises ToolError for any failure."""
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["text"] = tool.run(arguments)
            except ToolError as exc:
                box["error"] = exc
            except Exception as exc:  # noqa: BLE001 - a tool that crashes must not end the run
                logger.warning("agent tool crashed tool=%s type=%s", tool.name, type(exc).__name__)
                box["error"] = ToolError(f"{tool.name} stopped unexpectedly.")

        worker = threading.Thread(target=work, daemon=True, name=f"agent-tool-{tool.name}")
        worker.start()
        worker.join(self.limits.tool_seconds)
        if worker.is_alive():
            raise ToolError(
                f"{tool.name} did not finish within {int(self.limits.tool_seconds)} seconds."
            )
        if "error" in box:
            raise box["error"]
        return str(box["text"])

    # --- the run -----------------------------------------------------------------------------

    def execute(self, goal: str, context: str = "") -> RunResult:
        self.messages.append(ChatMessage("user", first_message(goal, context)))
        try:
            while True:
                self.check_limits()
                self.too_many_failures()
                reply = self.think()
                self.turns += 1
                if self.stop is not None and self.stop.is_set():
                    # Stopped while the model was thinking: what it asked for is not carried out.
                    raise _Finish("stopped", "You stopped the task.")
                if not reply.tool_calls:
                    text = reply.text.strip()
                    if not text:
                        answer = self.problem("The local model returned nothing.", (RETRY, STOP))
                        if answer != RETRY:
                            raise _Finish(
                                "stopped", "The local model returned nothing and you chose to stop."
                            )
                        continue
                    raise _Finish("done", text)
                call = reply.tool_calls[0]
                shown = json.dumps(dict(call.arguments), default=str)[:600]
                if len(reply.tool_calls) > 1:
                    self.log(
                        "problem",
                        detail=f"{len(reply.tool_calls) - 1} extra tool request(s) ignored",
                    )
                self.messages.append(
                    ChatMessage(
                        "assistant",
                        f"I will use the tool {call.name[:40]} with {shown}",
                    )
                )
                self.messages.append(ChatMessage("user", self.act(call)))
        except _Finish as end:
            return self.finish(end.status, end.message)

    def finish(self, status: Status, message: str) -> RunResult:
        if status != "done" and self.steps:
            done = "; ".join(f"{s.effect}: {s.outcome}" for s in self.steps)
            message = f"{message} What happened before that: {done}."
        self.log("run_finished", detail=f"{status}: {audit.summarize_result(message)}")
        self.closed = True
        return RunResult(self.run_id, status, message, self.turns, tuple(self.steps))


def run_agent(
    llm: LLMProvider,
    registry: ToolRegistry,
    grants: Grants,
    session: Session,
    approver: Approver,
    goal: str,
    *,
    limits: Limits | None = None,
    run_id: str | None = None,
    stop: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] | None = None,
    context: str = "",
) -> RunResult:
    """Carry out `goal`. Raises ValueError for an empty or too long goal.

    Never raises for anything that goes wrong during the run: that becomes a question to the owner,
    or a result with status "stopped" or "failed" that says why.
    """
    task = " ".join(goal.split())
    if not task:
        raise ValueError("say what the agent should do")
    if len(task) > MAX_GOAL_CHARS:
        raise ValueError(f"the task is too long ({len(task)} characters, at most {MAX_GOAL_CHARS})")
    run = _Run(
        llm,
        registry,
        grants,
        session,
        approver,
        limits or Limits(),
        run_id or audit.new_run_id(),
        stop,
        clock,
        progress,
    )
    try:
        run.log("run_started", detail=audit.summarize_result(task))
        if not grants.enabled:
            return run.finish(
                "failed", "The agent is switched off. Switch it on in the settings first."
            )
        return run.execute(task, context)
    except AuditError:
        logger.error("the agent stopped because its log could not be written")
        return RunResult(
            run.run_id,
            "failed",
            "I stopped because I could not write the agent's log, and I do not act "
            "without a record.",
            run.turns,
            tuple(run.steps),
        )
    except BaseException as exc:
        # Ctrl+C, or a bug: the task still ends in the log, so no task is left looking unfinished.
        if not run.closed:
            try:
                run.log("run_finished", detail=f"failed: interrupted ({type(exc).__name__})")
            except Exception:  # noqa: BLE001 - the log may be what broke; the original error matters more
                logger.error("an agent task ended without its log entry")
        raise
