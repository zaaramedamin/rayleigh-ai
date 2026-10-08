"""`python -m app agent`: run a task, see and change what the agent may do, read its log.

This is also the owner's side of the conversation in a terminal: the questions the agent loop asks
(approve this action, something went wrong) appear as cards and are answered with one key. Nothing
here lets the model answer for the owner: an answer that is not one of the offered options, an
empty answer, a closed terminal, all mean stop.

The module does not import the command line (app.cli) so that it can be tested alone; everything it
needs from outside (the model, the search, the screen) is passed in.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.agent import audit
from app.agent.audit import AuditError
from app.agent.builtin import Searcher
from app.agent.loop import (
    ALLOW,
    CONTINUE,
    DENY,
    RETRY,
    SKIP,
    STOP,
    Limits,
    Question,
    run_agent,
)
from app.agent.permissions import Grants, load_grants, save_grants
from app.agent.registry import default_registry
from app.agent.tools import ToolError, ToolRegistry
from app.ai.llm.base import LLMError, LLMProvider
from app.knowledge.retrieval.service import RetrievedChunk

logger = logging.getLogger(__name__)

ACTIONS = ("run", "status", "enable", "disable", "allow", "revoke", "web", "log")
MAX_BAD_ANSWERS = 3
LOG_LIST_LIMIT = 40

_LEVEL_WORDS = {
    "read_local": "only reads",
    "open_local": "opens things, asks every time",
    "external_read": "reads the web, asks every time",
    "write_local": "changes things, asks every time",
    "destructive": "never allowed",
}
# What the owner presses for each answer the loop can ask for.
_KEYS = {
    ALLOW: ("a", "allow once"),
    DENY: ("d", "do not allow"),
    STOP: ("s", "stop the task"),
    RETRY: ("r", "try again"),
    SKIP: ("k", "skip it"),
    CONTINUE: ("c", "let it go on"),
}
_SYNONYMS = {"y": ALLOW, "yes": ALLOW, "n": DENY, "no": DENY}


class ConsoleError(Exception):
    """Something the owner should be told in one sentence, with exit code 1."""


# --- the owner's side of the conversation -------------------------------------------------------


class TerminalApprover:
    """Shows the agent's questions as cards and reads the owner's answer from the keyboard."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        emit: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._emit = emit

    def _card(self, question: Question) -> None:
        emit = self._emit
        emit("")
        emit("=" * 64)
        emit(
            "The agent wants your approval"
            if question.kind == "approve"
            else "Something went wrong"
        )
        emit(f"  {question.message}")
        if question.kind == "approve":
            if question.arguments:
                emit("  Exactly: " + ", ".join(f"{k}={v!r}" for k, v in question.arguments.items()))
            emit(f"  Why you are asked: {question.reason}")
            if question.read_sources:
                emit(
                    "  The model has read the results of: "
                    + ", ".join(question.read_sources)
                    + ". Text it read could have influenced this request."
                )
        emit("  " + "   ".join(f"[{_KEYS[o][0]}] {_KEYS[o][1]}" for o in question.options))

    def ask(self, question: Question) -> str:
        self._card(question)
        for _ in range(MAX_BAD_ANSWERS):
            try:
                typed = self._input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._emit("(no answer: stopping)")
                return STOP
            for option in question.options:
                if typed in (_KEYS[option][0], option) or _SYNONYMS.get(typed) == option:
                    return option
            self._emit("Please press one of the keys shown above.")
        self._emit("(no valid answer: stopping)")
        return STOP


# --- the search, loaded only when the agent asks for it ----------------------------------------


class LazySearch:
    """The library's search for the agent. The embedding model is loaded the first time a search is
    asked for, so a task that never searches never needs it. `retrieve_notes` opens whatever it
    needs (a database session, the search index) and closes it again: a tool runs in its own
    thread, so it must not share a session with the loop."""

    def __init__(
        self,
        load_embedder: Callable[[], Any],
        retrieve_notes: Callable[[Any, str, int], list[RetrievedChunk]],
    ) -> None:
        self._load = load_embedder
        self._retrieve = retrieve_notes
        self._embedder: Any = None

    def __call__(self, query: str, count: int) -> list[RetrievedChunk]:
        try:
            if self._embedder is None:
                self._embedder = self._load()
            return self._retrieve(self._embedder, query, count)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure of the search is a question to the owner
            logger.warning("the agent's search failed type=%s", type(exc).__name__)
            raise ToolError(
                f"The search could not run ({type(exc).__name__}: {exc}). "
                "`python -m app doctor` says what is wrong."
            ) from exc


# --- the commands --------------------------------------------------------------------------------


@dataclass
class Environment:
    data_dir: Path
    session: Session
    make_llm: Callable[[], LLMProvider]
    search: Searcher | None
    min_score: float
    emit: Callable[[str], None] = print
    progress: Callable[[str], None] | None = None
    input_fn: Callable[[str], str] = input


def _registry(env: Environment) -> ToolRegistry:
    # Listing and switching tools must show every tool, even when there is no search to run.
    return default_registry(
        search=env.search or (lambda _query, _count: []), min_score=env.min_score
    )


def _status(env: Environment) -> int:
    grants = load_grants(env.data_dir)
    registry = _registry(env)
    env.emit(f"agent: {'ON' if grants.enabled else 'OFF'}")
    env.emit(f"web access: {'on' if grants.web else 'off'}")
    env.emit("tools (switch on with `python -m app agent allow <tool>`):")
    for name in registry.names():
        tool = registry.get(name)
        assert tool is not None
        state = "on " if name in grants.tools else "off"
        env.emit(f"  {name:<16} {state}  {_LEVEL_WORDS[tool.level]}")
    if not grants.enabled:
        env.emit("The agent does nothing until you run `python -m app agent enable`.")
    return 0


def _switch(env: Environment, enabled: bool) -> int:
    grants = load_grants(env.data_dir)
    save_grants(env.data_dir, Grants(enabled, grants.tools, grants.web))
    if enabled:
        env.emit("The agent is on. Tools stay off until you switch each one on, and anything that")
        env.emit("opens a file, changes something or reaches the internet asks you every time.")
    else:
        env.emit("The agent is off. It will not run anything.")
    return 0


def _tools_named(env: Environment, names: list[str]) -> list[str]:
    registry = _registry(env)
    if not names:
        raise ConsoleError("name at least one tool: " + ", ".join(registry.names()))
    unknown = [n for n in names if registry.get(n) is None]
    if unknown:
        raise ConsoleError(
            f"no such tool: {', '.join(unknown)}. The tools are: {', '.join(registry.names())}"
        )
    return names


def _allow(env: Environment, names: list[str], on: bool) -> int:
    chosen = _tools_named(env, names)
    registry = _registry(env)
    grants = load_grants(env.data_dir)
    for name in chosen:
        tool = registry.get(name)
        assert tool is not None
        if on and tool.level == "destructive":
            raise ConsoleError(
                f"{name} cannot be switched on: destructive actions are never allowed."
            )
    tools = grants.tools | set(chosen) if on else grants.tools - set(chosen)
    save_grants(env.data_dir, Grants(grants.enabled, frozenset(tools), grants.web))
    for name in chosen:
        tool = registry.get(name)
        assert tool is not None
        env.emit(f"{name}: {'on' if on else 'off'} ({_LEVEL_WORDS[tool.level]})")
        if on and tool.level == "external_read" and not grants.web:
            env.emit("  it also needs `python -m app agent web on`.")
    if on and not grants.enabled:
        env.emit("The agent itself is still off: `python -m app agent enable`.")
    return 0


def _web(env: Environment, words: list[str]) -> int:
    if words not in (["on"], ["off"]):
        raise ConsoleError("say `python -m app agent web on` or `python -m app agent web off`")
    grants = load_grants(env.data_dir)
    save_grants(env.data_dir, Grants(grants.enabled, grants.tools, words == ["on"]))
    env.emit(f"web access: {words[0]}")
    if words == ["on"]:
        env.emit("Reading a page still asks you every time, and shows the exact address first.")
    return 0


def _log(
    env: Environment, run_id: str | None, limit: int, export: str | None, erase: bool, yes: bool
) -> int:
    if erase:
        if (
            not yes
            and env.input_fn("Erase the agent's whole log? Type yes to confirm: ").strip().lower()
            != "yes"
        ):
            env.emit("nothing erased.")
            return 0
        env.emit(f"erased {audit.erase_log(env.session)} entries.")
        return 0
    if export:
        rows = [_row(e) for e in audit.all_events(env.session)]
        Path(export).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        env.emit(f"wrote {len(rows)} entries to {export}")
        return 0
    events = (
        audit.run_events(env.session, run_id)
        if run_id
        else list(reversed(audit.recent_events(env.session, limit)))
    )
    if not events:
        env.emit(
            "the agent has not done anything yet." if run_id is None else "no such run in the log."
        )
        return 0
    for entry in events:
        when = entry.created_at.astimezone().strftime("%d %b %H:%M:%S")
        who = f"{entry.decision or ''}{'/' + entry.decided_by if entry.decided_by else ''}"
        head = f"{when}  {entry.run_id[:8]}  {entry.kind:<12} {entry.tool or '':<14} {who:<12}"
        env.emit(f"{head} {entry.detail or ''}")
    return 0


def _row(entry: Any) -> dict[str, Any]:
    return {
        "time": entry.created_at.isoformat(),
        "run": entry.run_id,
        "step": entry.step,
        "kind": entry.kind,
        "tool": entry.tool,
        "level": entry.level,
        "decision": entry.decision,
        "decided_by": entry.decided_by,
        "detail": entry.detail,
    }


def _run(env: Environment, words: list[str]) -> int:
    task = " ".join(words)
    if not task.strip():
        raise ConsoleError("say what the agent should do: python -m app agent run <the task>")
    grants = load_grants(env.data_dir)
    if not grants.enabled:
        raise ConsoleError("the agent is off. Switch it on with `python -m app agent enable`.")
    try:
        llm = env.make_llm()
        result = run_agent(
            llm,
            _registry(env),
            grants,
            env.session,
            TerminalApprover(env.input_fn, env.emit),
            task,
            limits=Limits(),
            progress=env.progress,
        )
    except (ValueError, LLMError, AuditError) as exc:
        raise ConsoleError(str(exc)) from exc
    env.emit("")
    env.emit(result.answer)
    if result.steps:
        env.emit("")
        env.emit("What it did:")
        for step in result.steps:
            env.emit(f"  {step.effect}: {step.outcome}")
    env.emit(
        f"(run {result.run_id[:8]}; every step: `python -m app agent log --run {result.run_id}`)"
    )
    return 0 if result.status == "done" else 1


def agent_command(
    env: Environment,
    action: str,
    words: list[str],
    *,
    run_id: str | None = None,
    limit: int = LOG_LIST_LIMIT,
    export: str | None = None,
    erase: bool = False,
    yes: bool = False,
) -> int:
    """Carry out `python -m app agent <action> <words>`. Raises ConsoleError with one sentence."""
    if action == "run":
        return _run(env, words)
    if action == "status":
        return _status(env)
    if action in ("enable", "disable"):
        return _switch(env, action == "enable")
    if action in ("allow", "revoke"):
        return _allow(env, words, action == "allow")
    if action == "web":
        return _web(env, words)
    if action == "log":
        return _log(env, run_id, limit, export, erase, yes)
    raise ConsoleError(f"unknown action {action!r}; use one of: {', '.join(ACTIONS)}")
