from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.assistant.identity import DEFAULT_ROLE
from app.assistant.memory import MAX_MEMORIES, MAX_MEMORY_CHARS, add_memory
from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def client(session: Session, make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- identity ---------------------------------------------------------------------------------


def test_the_identity_starts_with_the_defaults(client: TestClient) -> None:
    body = client.get("/api/v1/assistant").json()

    assert body == {
        "name": "Reyleight",
        "address": "sir",
        "role": DEFAULT_ROLE,
        "use_profile": True,
        "use_memory": True,
        "default_role": DEFAULT_ROLE,
    }


def test_the_identity_can_be_changed_and_is_kept(client: TestClient) -> None:
    wanted = {
        "name": "Jarvis",
        "address": "boss",
        "role": "Keep my lab running.",
        "use_profile": False,
        "use_memory": True,
    }

    saved = client.put("/api/v1/assistant", json=wanted).json()

    assert {key: saved[key] for key in wanted} == wanted
    assert client.get("/api/v1/assistant").json()["name"] == "Jarvis"


def test_an_empty_name_and_role_go_back_to_the_defaults(client: TestClient) -> None:
    saved = client.put("/api/v1/assistant", json={"name": " ", "address": "", "role": ""}).json()

    assert (saved["name"], saved["address"], saved["role"]) == ("Reyleight", "", DEFAULT_ROLE)


@pytest.mark.parametrize(
    "body",
    [{"name": "x" * 41}, {"address": "x" * 41}, {"role": "x" * 2001}, {"use_memory": "perhaps"}],
)
def test_oversized_or_malformed_identities_are_rejected(
    client: TestClient, body: dict[str, object]
) -> None:
    assert client.put("/api/v1/assistant", json=body).status_code == 422


# --- memories ---------------------------------------------------------------------------------


def test_memories_can_be_added_listed_and_deleted(client: TestClient) -> None:
    created = client.post("/api/v1/assistant/memories", json={"text": "I live in Lyon"})
    assert created.status_code == 201
    memory = created.json()
    assert (memory["text"], memory["origin"]) == ("I live in Lyon", "owner")

    listed = client.get("/api/v1/assistant/memories").json()
    assert [m["text"] for m in listed["memories"]] == ["I live in Lyon"]
    assert listed["limit"] == MAX_MEMORIES

    assert client.delete(f"/api/v1/assistant/memories/{memory['id']}").status_code == 204
    assert client.delete(f"/api/v1/assistant/memories/{memory['id']}").status_code == 404
    assert client.get("/api/v1/assistant/memories").json()["memories"] == []


def test_what_the_assistant_saved_is_marked_as_such(client: TestClient, session: Session) -> None:
    add_memory(session, "Prefers short answers", "assistant")

    memories = client.get("/api/v1/assistant/memories").json()["memories"]

    assert [m["origin"] for m in memories] == ["assistant"]


def test_everything_can_be_forgotten_at_once(client: TestClient) -> None:
    for text in ("one", "two", "three"):
        client.post("/api/v1/assistant/memories", json={"text": text})

    assert client.delete("/api/v1/assistant/memories").json() == {"deleted": 3}
    assert client.get("/api/v1/assistant/memories").json()["memories"] == []


@pytest.mark.parametrize("text", ["", "x" * (MAX_MEMORY_CHARS + 1)])
def test_empty_and_oversized_memories_are_rejected(client: TestClient, text: str) -> None:
    assert client.post("/api/v1/assistant/memories", json={"text": text}).status_code == 422


def test_a_blank_memory_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/assistant/memories", json={"text": "   "})

    assert response.status_code == 422
    assert "nothing to remember" in response.json()["detail"]


def test_a_full_memory_says_so(client: TestClient, session: Session) -> None:
    for number in range(MAX_MEMORIES):
        add_memory(session, f"fact {number}", "owner")

    response = client.post("/api/v1/assistant/memories", json={"text": "one more"})

    assert response.status_code == 409
    assert "full" in response.json()["detail"]


# --- access -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/assistant"),
        ("PUT", "/api/v1/assistant"),
        ("GET", "/api/v1/assistant/memories"),
        ("POST", "/api/v1/assistant/memories"),
        ("DELETE", "/api/v1/assistant/memories"),
        ("DELETE", "/api/v1/assistant/memories/1"),
    ],
)
def test_the_assistant_routes_need_the_access_password(
    client: TestClient, make_settings: Callable[..., Settings], method: str, path: str
) -> None:
    from app.api.access import require_access

    app.dependency_overrides.pop(require_access, None)  # the real check, as in production
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=True)

    assert client.request(method, path, json={}).status_code == 401
