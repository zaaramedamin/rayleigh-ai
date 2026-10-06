"""The conversations API."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.assistant import conversations as service
from app.core.config import Settings, get_settings
from app.main import app
from app.storage.models import Conversation

URL = "/api/v1/conversations"
QUESTION = {"role": "user", "mode": "notes", "content": "How do I cook oats?"}
ANSWER = {
    "role": "assistant",
    "mode": "notes",
    "content": "Simmer them in milk [1].",
    "payload": {
        "grounded": True,
        "reason": "answered",
        "sources": [{"document_id": 4, "citation_id": "4:0", "text": "Simmer in milk."}],
    },
}


@pytest.fixture
def client(session: Session, make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings()
    yield TestClient(app)
    app.dependency_overrides.clear()


def start(client: TestClient, title: str = "") -> dict:
    return client.post(URL, json={"title": title}).json()


def test_an_empty_store_lists_nothing_and_says_what_the_limits_are(client: TestClient) -> None:
    body = client.get(URL).json()

    assert body == {
        "conversations": [],
        "total": 0,
        "limit": service.MAX_CONVERSATIONS,
        "retention_days": 0,
    }


def test_a_conversation_is_started_filled_listed_and_read_back(client: TestClient) -> None:
    created = client.post(URL, json={})
    assert created.status_code == 201
    conversation = created.json()
    assert (conversation["title"], conversation["message_count"]) == ("", 0)

    added = client.post(
        f"{URL}/{conversation['id']}/messages", json={"messages": [QUESTION, ANSWER]}
    )
    assert added.status_code == 201
    assert (added.json()["title"], added.json()["message_count"]) == ("How do I cook oats?", 2)

    listed = client.get(URL).json()
    assert listed["total"] == 1 and listed["conversations"][0]["message_count"] == 2

    detail = client.get(f"{URL}/{conversation['id']}").json()
    assert detail["message_limit"] == service.MAX_MESSAGES
    assert [(m["position"], m["role"], m["mode"]) for m in detail["messages"]] == [
        (0, "user", "notes"),
        (1, "assistant", "notes"),
    ]
    assert detail["messages"][0]["payload"] is None
    assert detail["messages"][1]["payload"] == ANSWER["payload"]  # exactly as saved


def test_a_conversation_can_be_renamed(client: TestClient) -> None:
    conversation = start(client, "old")

    renamed = client.put(f"{URL}/{conversation['id']}", json={"title": "  Oats  "})

    assert renamed.status_code == 200 and renamed.json()["title"] == "Oats"
    assert client.get(f"{URL}/{conversation['id']}").json()["title"] == "Oats"


@pytest.mark.parametrize("title", ["", "   ", "\x00\x01"])
def test_renaming_to_nothing_is_refused(client: TestClient, title: str) -> None:
    conversation = start(client, "keep me")

    response = client.put(f"{URL}/{conversation['id']}", json={"title": title})

    assert response.status_code == 422
    assert client.get(f"{URL}/{conversation['id']}").json()["title"] == "keep me"


def test_deleting_removes_the_conversation_and_its_messages(client: TestClient) -> None:
    conversation = start(client)
    client.post(f"{URL}/{conversation['id']}/messages", json={"messages": [QUESTION, ANSWER]})

    assert client.delete(f"{URL}/{conversation['id']}").status_code == 204

    assert client.get(f"{URL}/{conversation['id']}").status_code == 404
    assert client.get(URL).json()["total"] == 0
    assert client.delete(f"{URL}/{conversation['id']}").status_code == 404


def test_every_conversation_can_be_deleted_at_once(client: TestClient) -> None:
    for _ in range(3):
        client.post(f"{URL}/{start(client)['id']}/messages", json={"messages": [QUESTION]})

    assert client.delete(URL).json() == {"deleted": 3}

    assert client.get(URL).json()["total"] == 0


@pytest.mark.parametrize("call", ["get", "put", "delete"])
def test_a_conversation_that_does_not_exist_is_a_404(client: TestClient, call: str) -> None:
    kwargs = {"json": {"title": "x"}} if call == "put" else {}

    assert getattr(client, call)(f"{URL}/999", **kwargs).status_code == 404
    assert client.post(f"{URL}/999/messages", json={"messages": [QUESTION]}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": [{**QUESTION, "role": "system"}]},
        {"messages": [{**QUESTION, "mode": "secret"}]},
        {"messages": [{**QUESTION, "content": ""}]},
        {"messages": [{**QUESTION, "content": "x" * (service.MAX_CONTENT_CHARS + 1)}]},
        {"messages": [QUESTION] * 11},
        {"messages": [{**QUESTION, "payload": "not an object"}]},
    ],
)
def test_invalid_messages_are_refused_and_nothing_is_saved(client: TestClient, body: dict) -> None:
    conversation = start(client)

    response = client.post(f"{URL}/{conversation['id']}/messages", json=body)

    assert response.status_code == 422
    assert client.get(f"{URL}/{conversation['id']}").json()["messages"] == []


def test_a_message_with_blank_content_is_refused_by_the_service(client: TestClient) -> None:
    conversation = start(client)

    response = client.post(
        f"{URL}/{conversation['id']}/messages", json={"messages": [{**QUESTION, "content": "   "}]}
    )

    assert response.status_code == 422 and "empty" in response.json()["detail"]


def test_a_full_conversation_and_a_full_store_are_conflicts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "MAX_MESSAGES", 2)
    monkeypatch.setattr(service, "MAX_CONVERSATIONS", 1)
    conversation = start(client)
    client.post(f"{URL}/{conversation['id']}/messages", json={"messages": [QUESTION, ANSWER]})

    full = client.post(f"{URL}/{conversation['id']}/messages", json={"messages": [QUESTION]})
    another = client.post(URL, json={})

    assert full.status_code == 409 and "start a new one" in full.json()["detail"]
    assert another.status_code == 409 and "delete some" in another.json()["detail"]


def test_listing_is_paged(client: TestClient) -> None:
    for number in range(3):
        start(client, f"c{number}")

    page = client.get(URL, params={"limit": 2, "offset": 1}).json()

    assert page["total"] == 3 and [c["title"] for c in page["conversations"]] == ["c1", "c0"]
    assert client.get(URL, params={"limit": 0}).status_code == 422
    assert client.get(URL, params={"offset": -1}).status_code == 422


def test_listing_removes_conversations_past_the_retention_period(
    session: Session, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings(conversation_retention_days=30)
    try:
        client = TestClient(app)
        old, recent = start(client, "old"), start(client, "recent")
        row = session.get(Conversation, old["id"])
        assert row is not None
        row.updated_at = datetime.now(UTC) - timedelta(days=31)
        session.commit()

        body = client.get(URL).json()

        assert [c["title"] for c in body["conversations"]] == ["recent"]
        assert body["retention_days"] == 30 and recent["id"] != old["id"]
    finally:
        app.dependency_overrides.clear()


def test_the_conversations_need_the_access_password_like_everything_else(
    session: Session, make_settings: Callable[..., Settings]
) -> None:
    from app.api.access import require_access

    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=True)
    app.dependency_overrides.pop(require_access, None)  # the real check, as in production
    try:
        client = TestClient(app)

        assert client.get(URL).status_code == 401
        assert client.delete(URL).status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_conversations_are_listed_in_the_api_docs(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {f"{URL}", f"{URL}/{{conversation_id}}", f"{URL}/{{conversation_id}}/messages"} <= set(
        paths
    )
