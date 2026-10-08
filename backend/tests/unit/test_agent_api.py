"""The /agent API: switches, starting a task, watching it, answering its questions, the log."""

import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agent import audit
from app.agent.permissions import Grants, load_grants, save_grants
from app.agent.runs import RunManager
from app.agent.tools import Param, Tool, ToolRegistry
from app.api.access import require_access
from app.api.deps import get_session
from app.api.v1 import agent as agent_module
from app.api.v1.agent import get_manager
from app.core.config import Settings, get_settings
from app.main import app
from tests.fakes import ScriptedModel, asks, says

URL = "/api/v1/agent"


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


class Harness:
    def __init__(self, data_dir: Path, session: Session, settings: Settings) -> None:
        self.data_dir = data_dir
        self.session = session
        self.settings = settings
        self.model = ScriptedModel()
        self.things = Things()
        engine = session.get_bind()
        self.manager = RunManager(
            make_session=lambda: Session(engine),  # type: ignore[arg-type]
            make_llm=lambda: self.model,
            make_registry=lambda: self.things.registry,
            load_grants=lambda: load_grants(data_dir),
        )

    def switch_on(self, *tools: str) -> None:
        save_grants(self.data_dir, Grants(enabled=True, tools=frozenset(tools)))


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def harness(
    session: Session, data_dir: Path, make_settings: Callable[..., Settings]
) -> Iterator[Harness]:
    settings = make_settings()
    h = Harness(data_dir, session, settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_manager] = lambda: h.manager
    yield h


def until(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def run_state(client: TestClient, run_id: str) -> dict[str, Any]:
    reply = client.get(f"{URL}/runs/{run_id}")
    assert reply.status_code == 200
    return reply.json()  # type: ignore[no-any-return]


def wait_for(client: TestClient, run_id: str, *statuses: str) -> dict[str, Any]:
    until(lambda: run_state(client, run_id)["status"] in statuses)
    return run_state(client, run_id)


# --- the switches ----------------------------------------------------------------------------


def test_a_fresh_install_has_everything_off_and_lists_every_tool(
    client: TestClient, harness: Harness
) -> None:
    body = client.get(URL).json()

    assert body["enabled"] is False and body["web"] is False and body["active_run"] is None
    tools = {t["name"]: t for t in body["tools"]}
    assert list(tools) == [
        "calculator",
        "current_time",
        "search_notes",
        "open_path",
        "open_app",
        "fetch_web_page",
    ]
    assert not any(t["enabled"] for t in tools.values())
    assert (
        tools["calculator"]["level"] == "read_local"
        and tools["calculator"]["what_it_does"] == "only reads"
    )
    assert tools["open_path"]["what_it_does"] == "opens things, asks every time"
    assert (
        tools["fetch_web_page"]["level"] == "external_read"
        and tools["fetch_web_page"]["description"]
    )


def test_the_switches_are_changed_one_field_at_a_time_and_saved(
    client: TestClient, harness: Harness
) -> None:
    reply = client.put(URL, json={"enabled": True})
    assert reply.status_code == 200 and reply.json()["enabled"] is True
    assert load_grants(harness.data_dir) == Grants(enabled=True)

    client.put(URL, json={"tools": ["calculator", "open_path"]})
    client.put(URL, json={"web": True})
    assert load_grants(harness.data_dir) == Grants(
        enabled=True, tools=frozenset({"calculator", "open_path"}), web=True
    )
    body = client.put(URL, json={"tools": ["calculator"]}).json()
    assert [t["name"] for t in body["tools"] if t["enabled"]] == ["calculator"]
    assert body["enabled"] is True and body["web"] is True  # untouched


def test_an_empty_change_changes_nothing(client: TestClient, harness: Harness) -> None:
    harness.switch_on("calculator")

    client.put(URL, json={})

    assert load_grants(harness.data_dir) == Grants(enabled=True, tools=frozenset({"calculator"}))


def test_a_tool_that_does_not_exist_cannot_be_switched_on(
    client: TestClient, harness: Harness
) -> None:
    reply = client.put(URL, json={"tools": ["calculator", "delete_everything"]})

    assert reply.status_code == 422 and "No such tool: delete_everything" in reply.json()["detail"]
    assert load_grants(harness.data_dir) == Grants()


def test_a_destructive_tool_cannot_be_switched_on(
    client: TestClient, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    wipe = Tool("wipe", "d", "destructive", {}, lambda _a: "")
    monkeypatch.setattr(agent_module, "default_registry", lambda **_k: ToolRegistry([wipe]))

    reply = client.put(URL, json={"tools": ["wipe"]})

    assert reply.status_code == 422 and "never allowed" in reply.json()["detail"]
    assert load_grants(harness.data_dir).tools == frozenset()


def test_wrong_kinds_of_value_are_refused(client: TestClient, harness: Harness) -> None:
    assert client.put(URL, json={"enabled": "yes please"}).status_code == 422
    assert client.put(URL, json={"tools": "calculator"}).status_code == 422
    assert load_grants(harness.data_dir) == Grants()


# --- running a task ----------------------------------------------------------------------------


def test_a_task_cannot_start_while_the_agent_is_off(client: TestClient, harness: Harness) -> None:
    reply = client.post(f"{URL}/runs", json={"task": "hello"})

    assert reply.status_code == 409 and "switched off" in reply.json()["detail"]
    assert harness.model.chats == []


@pytest.mark.parametrize("task", ["", "x" * 2001])
def test_an_empty_or_too_long_task_is_refused(
    client: TestClient, harness: Harness, task: str
) -> None:
    harness.switch_on()

    assert client.post(f"{URL}/runs", json={"task": task}).status_code == 422
    assert client.post(f"{URL}/runs", json={}).status_code == 422


def test_a_blank_task_is_refused_with_a_sentence(client: TestClient, harness: Harness) -> None:
    harness.switch_on()

    reply = client.post(f"{URL}/runs", json={"task": "   "})

    assert reply.status_code == 422 and "say what the agent should do" in reply.json()["detail"]


def test_a_task_is_started_and_watched_until_it_is_done(
    client: TestClient, harness: Harness
) -> None:
    harness.switch_on("echo")
    harness.model.turns = [asks("echo", word="hi"), says("It said hi.")]

    started = client.post(f"{URL}/runs", json={"task": "say hi"})
    assert started.status_code == 202
    first = started.json()
    assert first["status"] in ("running", "done") and first["task"] == "say hi"
    done = wait_for(client, first["run_id"], "done")

    assert done["answer"] == "It said hi." and done["question"] is None
    assert done["steps"] == [{"tool": "echo", "effect": "echo(word='hi')", "outcome": "done"}]
    assert done["progress"][:2] == ["thinking ...", "running: echo(word='hi')"]
    assert done["finished_at"] is not None and done["model_turns"] == 2


def test_an_unknown_task_is_a_404(client: TestClient, harness: Harness) -> None:
    assert client.get(f"{URL}/runs/nope").status_code == 404
    assert client.post(f"{URL}/runs/nope/stop").status_code == 404
    reply = client.post(f"{URL}/runs/nope/answer", json={"question_id": 1, "choice": "allow"})
    assert reply.status_code == 404


def test_the_task_that_is_running_is_shown_with_the_settings(
    client: TestClient, harness: Harness
) -> None:
    harness.switch_on("open_thing")
    harness.model.turns = [asks("open_thing", path="C:/x.txt"), says("ok")]
    run_id = client.post(f"{URL}/runs", json={"task": "open it"}).json()["run_id"]
    wait_for(client, run_id, "waiting")

    active = client.get(URL).json()["active_run"]

    assert active is not None and active["run_id"] == run_id and active["status"] == "waiting"
    client.post(f"{URL}/runs/{run_id}/stop")
    wait_for(client, run_id, "stopped")
    assert client.get(URL).json()["active_run"] is None


# --- the owner answers ---------------------------------------------------------------------------


def start_waiting(client: TestClient, harness: Harness) -> tuple[str, dict[str, Any]]:
    harness.switch_on("open_thing")
    harness.model.turns = [asks("open_thing", path="C:/notes/plan.txt"), says("Opened it.")]
    run_id = client.post(f"{URL}/runs", json={"task": "open the plan"}).json()["run_id"]
    return run_id, wait_for(client, run_id, "waiting")


def test_a_task_that_needs_approval_shows_the_exact_question_and_waits(
    client: TestClient, harness: Harness
) -> None:
    run_id, waiting = start_waiting(client, harness)

    question = waiting["question"]
    assert question["kind"] == "approve" and question["tool"] == "open_thing"
    assert question["effect"] == "Open C:/notes/plan.txt"
    assert question["arguments"] == {"path": "C:/notes/plan.txt"}
    assert question["options"] == ["allow", "deny", "stop"]
    assert "each time" in question["reason"] and question["read_sources"] == []
    time.sleep(0.1)
    assert harness.things.opened == []


def test_allowing_runs_the_action_and_finishes_the_task(
    client: TestClient, harness: Harness
) -> None:
    run_id, waiting = start_waiting(client, harness)

    reply = client.post(
        f"{URL}/runs/{run_id}/answer",
        json={"question_id": waiting["question"]["id"], "choice": "allow"},
    )

    assert reply.status_code == 204
    done = wait_for(client, run_id, "done")
    assert harness.things.opened == ["C:/notes/plan.txt"] and done["answer"] == "Opened it."


def test_denying_does_not_run_it(client: TestClient, harness: Harness) -> None:
    run_id, waiting = start_waiting(client, harness)

    client.post(
        f"{URL}/runs/{run_id}/answer",
        json={"question_id": waiting["question"]["id"], "choice": "deny"},
    )
    done = wait_for(client, run_id, "done")

    assert harness.things.opened == [] and done["steps"][0]["outcome"] == "not allowed by you"


def test_an_answer_that_is_not_an_option_or_is_for_another_question_is_refused(
    client: TestClient, harness: Harness
) -> None:
    run_id, waiting = start_waiting(client, harness)
    question_id = waiting["question"]["id"]

    for choice in ("yes", "retry", "ALLOW", ""):
        bad = client.post(
            f"{URL}/runs/{run_id}/answer", json={"question_id": question_id, "choice": choice}
        )
        assert bad.status_code == 422
    stale = client.post(
        f"{URL}/runs/{run_id}/answer", json={"question_id": question_id + 7, "choice": "allow"}
    )
    assert stale.status_code == 409
    assert (
        client.post(f"{URL}/runs/{run_id}/answer", json={"question_id": question_id}).status_code
        == 422
    )
    assert run_state(client, run_id)["status"] == "waiting" and harness.things.opened == []


def test_a_second_task_cannot_start_while_one_is_waiting(
    client: TestClient, harness: Harness
) -> None:
    run_id, _waiting = start_waiting(client, harness)

    reply = client.post(f"{URL}/runs", json={"task": "another"})

    assert reply.status_code == 409 and "already working" in reply.json()["detail"]


def test_stopping_ends_the_task_without_acting(client: TestClient, harness: Harness) -> None:
    run_id, _waiting = start_waiting(client, harness)

    assert client.post(f"{URL}/runs/{run_id}/stop").status_code == 204
    done = wait_for(client, run_id, "stopped")

    assert harness.things.opened == [] and "You stopped the task." in done["answer"]
    assert client.post(f"{URL}/runs/{run_id}/stop").status_code == 409  # already finished


# --- the log -----------------------------------------------------------------------------------


def run_something(client: TestClient, harness: Harness) -> str:
    harness.switch_on("echo")
    harness.model.turns = [asks("echo", word="hi"), says("Done.")]
    run_id = client.post(f"{URL}/runs", json={"task": "say hi"}).json()["run_id"]
    wait_for(client, run_id, "done")
    return run_id


def test_the_log_shows_a_task_in_the_order_it_happened(
    client: TestClient, harness: Harness
) -> None:
    run_id = run_something(client, harness)

    events = client.get(f"{URL}/log", params={"run_id": run_id}).json()["events"]

    assert [e["kind"] for e in events] == [
        "run_started",
        "request",
        "decision",
        "result",
        "run_finished",
    ]
    assert events[1]["tool"] == "echo" and events[1]["detail"] == '{"word": "hi"}'
    assert (events[2]["decision"], events[2]["decided_by"], events[2]["level"]) == (
        "allow",
        "policy",
        "read_local",
    )
    assert all(e["run_id"] == run_id and e["time"] for e in events)


def test_the_log_without_a_task_is_the_latest_first_and_limited(
    client: TestClient, harness: Harness
) -> None:
    run_something(client, harness)

    events = client.get(f"{URL}/log", params={"limit": 2}).json()["events"]

    assert [e["kind"] for e in events] == ["run_finished", "result"]


@pytest.mark.parametrize("limit", [0, 201, -1])
def test_a_limit_out_of_range_is_refused(client: TestClient, harness: Harness, limit: int) -> None:
    assert client.get(f"{URL}/log", params={"limit": limit}).status_code == 422


def test_the_log_can_be_erased_but_not_edited(client: TestClient, harness: Harness) -> None:
    run_something(client, harness)

    assert client.delete(f"{URL}/log").json() == {"erased": 5}
    assert client.get(f"{URL}/log").json() == {"events": []}
    assert client.delete(f"{URL}/log").json() == {"erased": 0}
    for method in ("put", "patch", "post"):
        assert getattr(client, method)(f"{URL}/log", json={}).status_code in (404, 405)


# --- guarding it ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "route", "body"),
    [
        ("get", "", None),
        ("put", "", {"enabled": True}),
        ("post", "/runs", {"task": "x"}),
        ("get", "/runs/abc", None),
        ("post", "/runs/abc/answer", {"question_id": 1, "choice": "allow"}),
        ("post", "/runs/abc/stop", None),
        ("get", "/log", None),
        ("delete", "/log", None),
    ],
)
def test_every_agent_route_needs_the_access_password(
    client: TestClient, harness: Harness, method: str, route: str, body: dict[str, Any] | None
) -> None:
    app.dependency_overrides.pop(require_access, None)  # the real check, as in production

    reply = getattr(client, method)(f"{URL}{route}", **({"json": body} if body is not None else {}))

    assert reply.status_code == 401
    assert load_grants(harness.data_dir) == Grants()


def test_the_routes_are_in_the_api_description(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    for path in (
        "/api/v1/agent",
        "/api/v1/agent/runs",
        "/api/v1/agent/runs/{run_id}",
        "/api/v1/agent/runs/{run_id}/answer",
        "/api/v1/agent/runs/{run_id}/stop",
        "/api/v1/agent/log",
    ):
        assert path in paths


# --- one manager per data folder -------------------------------------------------------------


def test_the_manager_is_made_once_for_each_data_folder(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_module, "_managers", {})
    made: list[Path] = []
    monkeypatch.setattr(
        agent_module, "build_manager", lambda s: made.append(s.data_dir) or object()
    )

    first = get_manager(make_settings(data_dir=tmp_path / "a"))
    again = get_manager(make_settings(data_dir=tmp_path / "a"))
    other = get_manager(make_settings(data_dir=tmp_path / "b"))

    assert first is again and first is not other
    assert made == [tmp_path / "a", tmp_path / "b"]


def test_the_real_manager_offers_the_real_tools_and_the_search_only_loads_when_used(
    make_settings: Callable[..., Settings], data_dir: Path
) -> None:
    manager = agent_module.build_manager(make_settings())

    registry = manager._make_registry()  # noqa: SLF001 - checking what the wiring builds

    assert registry.names() == [
        "calculator",
        "current_time",
        "search_notes",
        "open_path",
        "open_app",
        "fetch_web_page",
    ]
    assert manager._load_grants() == Grants()  # noqa: SLF001


def test_starting_the_agent_closes_the_tasks_a_crash_left_unfinished(
    make_settings: Callable[..., Settings], session: Session
) -> None:
    from datetime import UTC, datetime, timedelta

    from app.storage.models import AgentEvent

    session.add(
        AgentEvent(
            run_id="left-over",
            kind="run_started",
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
    )
    session.commit()

    agent_module.build_manager(make_settings())

    session.expire_all()
    assert [e.kind for e in audit.run_events(session, "left-over")] == [
        "run_started",
        "run_finished",
    ]


def test_a_log_that_cannot_be_tidied_does_not_stop_the_agent_from_starting(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_session: Session) -> int:
        raise RuntimeError("the log is locked")

    monkeypatch.setattr(agent_module.audit, "close_unfinished_runs", broken)

    assert isinstance(agent_module.build_manager(make_settings()), RunManager)


def test_the_real_manager_tells_the_agent_where_the_notes_folders_are(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    notes = tmp_path / "my-notes"
    notes.mkdir()
    manager = agent_module.build_manager(make_settings(allowed_folders=[notes]))

    facts = manager._make_context()  # noqa: SLF001 - checking what the wiring builds

    assert f"- The user's notes are in: {notes}." in facts
    assert "never invent a path" in facts and "home folder is" in facts
