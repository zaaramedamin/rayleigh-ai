"""The feedback API."""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.access import require_access
from app.api.deps import get_session
from app.assistant import feedback as service
from app.core.config import Settings, get_settings
from app.main import app

URL = "/api/v1/feedback"
MARK = {
    "kind": "wrong_source",
    "mode": "notes",
    "question": "How long do oats simmer?",
    "answer": "Rice needs eighteen minutes [1].",
    "details": {"sources": [{"document_id": 4, "source": "rice.txt"}], "reason": "answered"},
}


@pytest.fixture
def client(session: Session, make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings()
    yield TestClient(app)
    app.dependency_overrides.clear()


def add(client: TestClient, **overrides: object) -> dict:
    response = client.post(URL, json={**MARK, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def test_an_empty_list_shows_zero_counts_and_the_limit(client: TestClient) -> None:
    assert client.get(URL).json() == {
        "marks": [],
        "total": 0,
        "counts": {"helpful": 0, "not_helpful": 0, "wrong_source": 0, "missing_info": 0},
        "limit": service.MAX_FEEDBACK,
    }


def test_a_mark_is_added_listed_and_read_back_exactly(client: TestClient) -> None:
    created = add(client, note="it used the rice note")

    assert created["kind"] == "wrong_source" and created["mode"] == "notes"
    assert created["note"] == "it used the rice note"
    assert created["details"] == MARK["details"]  # exactly as it was sent
    listed = client.get(URL).json()
    assert listed["total"] == 1 and listed["counts"]["wrong_source"] == 1
    assert listed["marks"][0]["id"] == created["id"]
    assert listed["marks"][0]["question"] == "How long do oats simmer?"


def test_a_mark_without_a_note_or_details_is_fine(client: TestClient) -> None:
    created = add(client, kind="helpful", note=None, details=None)

    assert (created["note"], created["details"]) == (None, None)


def test_the_list_can_be_filtered_by_kind_and_paged(client: TestClient) -> None:
    add(client, kind="helpful", question="one")
    second = add(client, kind="not_helpful", question="two")
    third = add(client, kind="wrong_source", question="three")

    failures = client.get(URL, params=[("kinds", "not_helpful"), ("kinds", "wrong_source")]).json()
    paged = client.get(URL, params={"limit": 1, "offset": 1}).json()

    assert [m["id"] for m in failures["marks"]] == [third["id"], second["id"]]
    assert failures["total"] == 2  # matching marks, while the counts cover everything kept
    assert sum(failures["counts"].values()) == 3
    assert paged["total"] == 3 and [m["id"] for m in paged["marks"]] == [second["id"]]


@pytest.mark.parametrize(
    "params", [{"kinds": "great"}, {"limit": 0}, {"limit": 501}, {"offset": -1}]
)
def test_a_bad_list_request_is_refused(client: TestClient, params: dict) -> None:
    assert client.get(URL, params=params).status_code == 422


def test_a_mark_can_be_changed(client: TestClient) -> None:
    created = add(client, kind="not_helpful")

    changed = client.put(
        f"{URL}/{created['id']}", json={"kind": "missing_info", "note": "add the hotel"}
    )

    assert changed.status_code == 200
    assert (changed.json()["kind"], changed.json()["note"]) == ("missing_info", "add the hotel")
    assert changed.json()["updated_at"] >= created["updated_at"]
    assert client.get(URL).json()["counts"]["missing_info"] == 1


def test_changing_to_an_unknown_kind_is_refused_and_changes_nothing(client: TestClient) -> None:
    created = add(client, kind="helpful")

    assert client.put(f"{URL}/{created['id']}", json={"kind": "great"}).status_code == 422
    assert client.get(URL).json()["marks"][0]["kind"] == "helpful"


def test_a_mark_is_deleted_for_good_or_all_at_once(client: TestClient) -> None:
    keep, drop = add(client), add(client)

    assert client.delete(f"{URL}/{drop['id']}").status_code == 204
    assert client.delete(f"{URL}/{drop['id']}").status_code == 404
    assert [m["id"] for m in client.get(URL).json()["marks"]] == [keep["id"]]
    add(client)
    assert client.delete(URL).json() == {"deleted": 2}
    assert client.get(URL).json()["total"] == 0


def test_a_mark_that_does_not_exist_is_a_404(client: TestClient) -> None:
    assert client.put(f"{URL}/999", json={"kind": "helpful"}).status_code == 404
    assert client.delete(f"{URL}/999").status_code == 404


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "great"},
        {"mode": "secret"},
        {"question": ""},
        {"answer": ""},
        {"question": "q" * (service.MAX_QUESTION_CHARS + 1)},
        {"answer": "a" * (service.MAX_ANSWER_CHARS + 1)},
        {"note": "n" * (service.MAX_NOTE_CHARS + 1)},
        {"details": "not an object"},
        {"details": ["a", "list"]},
    ],
)
def test_an_invalid_mark_is_refused_and_nothing_is_saved(
    client: TestClient, overrides: dict[str, object]
) -> None:
    assert client.post(URL, json={**MARK, **overrides}).status_code == 422
    assert client.get(URL).json()["total"] == 0


def test_a_mark_with_only_blank_text_is_refused_by_the_service(client: TestClient) -> None:
    response = client.post(URL, json={**MARK, "question": "   "})

    assert response.status_code == 422 and "needs the question" in response.json()["detail"]


def test_a_full_store_is_a_conflict(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "MAX_FEEDBACK", 1)
    add(client)

    response = client.post(URL, json=MARK)

    assert response.status_code == 409 and "delete some" in response.json()["detail"]


def test_marks_need_the_access_password_like_everything_else(
    session: Session, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=True)
    app.dependency_overrides.pop(require_access, None)  # the real check, as in production
    try:
        client = TestClient(app)

        assert client.get(URL).status_code == 401
        assert client.post(URL, json=MARK).status_code == 401
        assert client.delete(URL).status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_the_feedback_routes_are_listed_in_the_api_docs(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {URL, f"{URL}/{{feedback_id}}"} <= set(paths)
