"""The agent: what it may do, starting a task, watching it, answering its questions, its log.

Everything here sits behind the access password. Nothing the model writes can reach these routes: a
question is answered only by a call to /answer, which is what a button in the interface does.
"""

import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent import audit
from app.agent.audit import AuditError
from app.agent.console import LEVEL_WORDS, LazySearch
from app.agent.loop import MAX_GOAL_CHARS, Question
from app.agent.permissions import Grants, load_grants, save_grants
from app.agent.registry import default_registry
from app.agent.runs import (
    RunBusyError,
    RunManager,
    RunState,
    StaleQuestionError,
)
from app.api.deps import SessionDep, SettingsDep, _engine, locked_vector_store
from app.core.config import Settings
from app.knowledge.components import create_llm, load_embedder
from app.knowledge.retrieval.service import RetrievedChunk, retrieve

router = APIRouter(prefix="/agent", tags=["agent"])

MAX_LOG_ENTRIES = 200

# --- the manager of runs: one per data folder -----------------------------------------------------

_managers: dict[Path, RunManager] = {}
_managers_lock = threading.Lock()


def build_manager(settings: Settings) -> RunManager:
    """The run manager wired to this installation's model, library and grants."""

    def retrieve_notes(embedder: Any, query: str, count: int) -> list[RetrievedChunk]:
        # A tool runs in its own thread, so the search opens its own session.
        with (
            Session(_engine(settings.data_dir)) as own,
            locked_vector_store(settings, embedder) as store,
        ):
            return retrieve(own, embedder, store, query, top_k=count, mode=settings.search_mode)

    search = LazySearch(
        lambda: load_embedder(settings.embedding_model, settings.models_dir), retrieve_notes
    )
    return RunManager(
        make_session=lambda: Session(_engine(settings.data_dir)),
        make_llm=lambda: create_llm(settings),
        make_registry=lambda: default_registry(search=search, min_score=settings.answer_min_score),
        load_grants=lambda: load_grants(settings.data_dir),
    )


def get_manager(settings: SettingsDep) -> RunManager:
    with _managers_lock:
        manager = _managers.get(settings.data_dir)
        if manager is None:
            manager = _managers[settings.data_dir] = build_manager(settings)
        return manager


ManagerDep = Annotated[RunManager, Depends(get_manager)]


# --- models -------------------------------------------------------------------------------


class AgentToolOut(BaseModel):
    name: str
    description: str
    level: Literal["read_local", "open_local", "external_read", "write_local", "destructive"]
    what_it_does: str = Field(description='In plain words: "only reads", "opens things, asks".')
    enabled: bool


class QuestionOut(BaseModel):
    id: int = Field(description="Send it back with the answer, so a stale question is refused.")
    kind: Literal["approve", "problem"]
    title: str
    message: str
    options: list[str] = Field(description="The only answers that mean anything.")
    tool: str | None
    arguments: dict[str, Any] = Field(description="The exact arguments, as checked.")
    effect: str
    reason: str
    read_sources: list[str] = Field(description="Tools whose results the model has read so far.")


class StepOut(BaseModel):
    tool: str
    effect: str
    outcome: str


class RunOut(BaseModel):
    run_id: str
    task: str
    status: Literal["running", "waiting", "done", "stopped", "failed"]
    answer: str
    started_at: str
    finished_at: str | None
    progress: list[str]
    question: QuestionOut | None
    steps: list[StepOut]
    model_turns: int


class AgentSettingsOut(BaseModel):
    enabled: bool
    web: bool
    tools: list[AgentToolOut]
    active_run: RunOut | None = Field(description="The task that is running now, if any.")


class AgentSettingsIn(BaseModel):
    enabled: bool | None = None
    web: bool | None = None
    tools: list[str] | None = Field(default=None, description="The tools to have switched on.")


class StartRun(BaseModel):
    task: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)


class AnswerIn(BaseModel):
    question_id: int
    choice: str = Field(max_length=40)


class EventOut(BaseModel):
    id: int
    time: str
    run_id: str
    step: int
    kind: str
    tool: str | None
    level: str | None
    decision: str | None
    decided_by: str | None
    detail: str | None


class EventList(BaseModel):
    events: list[EventOut]


class ErasedLog(BaseModel):
    erased: int


# --- helpers ------------------------------------------------------------------------------


def _question(question: Question, question_id: int) -> QuestionOut:
    return QuestionOut(
        id=question_id,
        kind=question.kind,
        title=question.title,
        message=question.message,
        options=list(question.options),
        tool=question.tool,
        arguments=dict(question.arguments),
        effect=question.effect,
        reason=question.reason,
        read_sources=list(question.read_sources),
    )


def _run(state: RunState) -> RunOut:
    return RunOut(
        run_id=state.run_id,
        task=state.task,
        status=state.status,
        answer=state.answer,
        started_at=state.started_at,
        finished_at=state.finished_at,
        progress=state.progress,
        question=_question(state.pending.question, state.pending.id) if state.pending else None,
        steps=[StepOut(tool=s.tool, effect=s.effect, outcome=s.outcome) for s in state.steps],
        model_turns=state.model_turns,
    )


def _settings_out(settings: Settings, manager: RunManager) -> AgentSettingsOut:
    grants = load_grants(settings.data_dir)
    registry = default_registry(search=lambda _query, _count: [], min_score=0.0)
    tools = []
    for name in registry.names():
        tool = registry.get(name)
        assert tool is not None
        tools.append(
            AgentToolOut(
                name=name,
                description=tool.description,
                level=tool.level,
                what_it_does=LEVEL_WORDS[tool.level],
                enabled=name in grants.tools,
            )
        )
    active = manager.active()
    return AgentSettingsOut(
        enabled=grants.enabled,
        web=grants.web,
        tools=tools,
        active_run=_run(active) if active else None,
    )


# --- routes -------------------------------------------------------------------------------


@router.get("", response_model=AgentSettingsOut)
def agent_settings(settings: SettingsDep, manager: ManagerDep) -> AgentSettingsOut:
    return _settings_out(settings, manager)


@router.put("", response_model=AgentSettingsOut)
def change_agent_settings(
    body: AgentSettingsIn, settings: SettingsDep, manager: ManagerDep
) -> AgentSettingsOut:
    """Switch the agent, the web or tools on or off. Only the fields sent are changed."""
    registry = default_registry(search=lambda _query, _count: [], min_score=0.0)
    grants = load_grants(settings.data_dir)
    tools = grants.tools
    if body.tools is not None:
        unknown = sorted(set(body.tools) - set(registry.names()))
        if unknown:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"No such tool: {', '.join(unknown)}."
            )
        destructive = [n for n in body.tools if (t := registry.get(n)) and t.level == "destructive"]
        if destructive:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{', '.join(destructive)} cannot be switched on: "
                "destructive actions are never allowed.",
            )
        tools = frozenset(body.tools)
    save_grants(
        settings.data_dir,
        Grants(
            enabled=grants.enabled if body.enabled is None else body.enabled,
            tools=tools,
            web=grants.web if body.web is None else body.web,
        ),
    )
    return _settings_out(settings, manager)


@router.post("/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def start_run(body: StartRun, settings: SettingsDep, manager: ManagerDep) -> RunOut:
    """Start a task in the background. Watch it with GET /runs/{id}."""
    if not load_grants(settings.data_dir).enabled:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The agent is switched off. Switch it on in the settings first.",
        )
    try:
        return _run(manager.start(body.task))
    except RunBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, manager: ManagerDep) -> RunOut:
    state = manager.get(run_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "There is no such task.")
    return _run(state)


@router.post("/runs/{run_id}/answer", status_code=status.HTTP_204_NO_CONTENT)
def answer_question(run_id: str, body: AnswerIn, manager: ManagerDep) -> Response:
    """The owner's answer to the question the task is waiting on."""
    try:
        manager.answer(run_id, body.question_id, body.choice)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except StaleQuestionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/runs/{run_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
def stop_run(run_id: str, manager: ManagerDep) -> Response:
    if manager.get(run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "There is no such task.")
    if not manager.stop(run_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "That task has already finished.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/log", response_model=EventList)
def agent_log(
    session: SessionDep,
    run_id: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LOG_ENTRIES)] = 50,
) -> EventList:
    """The audit log: one task in the order it happened, or the latest entries newest first."""
    entries = (
        audit.run_events(session, run_id)[:limit] if run_id else audit.recent_events(session, limit)
    )
    return EventList(
        events=[
            EventOut(
                id=e.id,
                time=e.created_at.isoformat(),
                run_id=e.run_id,
                step=e.step,
                kind=e.kind,
                tool=e.tool,
                level=e.level,
                decision=e.decision,
                decided_by=e.decided_by,
                detail=e.detail,
            )
            for e in entries
        ]
    )


@router.delete("/log", response_model=ErasedLog)
def erase_agent_log(session: SessionDep) -> ErasedLog:
    """Erase the whole log. The owner's choice; an entry can never be edited."""
    try:
        return ErasedLog(erased=audit.erase_log(session))
    except AuditError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
