import json
import time
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.access import password as passwords
from app.access.sessions import LoginThrottle, SessionStore, sessions, throttle
from app.api.access import require_access
from app.core.config import Settings, get_settings
from app.main import app

PASSWORD = "correct horse battery"


@pytest.fixture
def client(make_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    """The real sign-in check (the rest of the suite runs with it switched off)."""
    app.dependency_overrides.pop(require_access, None)
    app.dependency_overrides[get_settings] = lambda: make_settings()
    sessions.clear()
    throttle.reset()
    yield TestClient(app)
    app.dependency_overrides.clear()
    sessions.clear()
    throttle.reset()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _setup(client: TestClient) -> str:
    response = client.post("/api/v1/auth/setup", json={"password": PASSWORD})
    assert response.status_code == 201
    return str(response.json()["token"])


# --- password file ----------------------------------------------------------------------------


def test_the_password_is_never_stored_in_the_clear(data_dir) -> None:  # type: ignore[no-untyped-def]
    passwords.set_password(data_dir, PASSWORD)

    stored = (data_dir / passwords.ACCESS_FILENAME).read_text(encoding="utf-8")

    assert PASSWORD not in stored
    assert set(json.loads(stored)) == {"version", "n", "r", "p", "salt", "hash"}


def test_only_the_right_password_verifies(data_dir) -> None:  # type: ignore[no-untyped-def]
    passwords.set_password(data_dir, PASSWORD)

    assert passwords.verify_password(data_dir, PASSWORD)
    assert not passwords.verify_password(data_dir, PASSWORD + " ")
    assert not passwords.verify_password(data_dir, "")


def test_no_password_set_means_nothing_verifies(data_dir) -> None:  # type: ignore[no-untyped-def]
    assert not passwords.is_configured(data_dir)
    assert not passwords.verify_password(data_dir, PASSWORD)


def test_a_damaged_password_file_never_lets_anyone_in(data_dir) -> None:  # type: ignore[no-untyped-def]
    data_dir.mkdir(parents=True)
    (data_dir / passwords.ACCESS_FILENAME).write_text("not json")

    assert not passwords.verify_password(data_dir, PASSWORD)


def test_short_passwords_are_refused(data_dir) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(passwords.PasswordError):
        passwords.set_password(data_dir, "short")


def test_equivalent_unicode_forms_are_the_same_password(data_dir) -> None:  # type: ignore[no-untyped-def]
    passwords.set_password(data_dir, "café-password")

    assert passwords.verify_password(data_dir, "café-password")


# --- sessions and the brake -------------------------------------------------------------------


def test_a_session_token_is_valid_until_revoked() -> None:
    store = SessionStore()
    token = store.create()

    assert store.validate(token)
    store.revoke(token)
    assert not store.validate(token)
    assert not store.validate("made-up")


def test_an_idle_session_expires() -> None:
    store = SessionStore(idle_seconds=0.05)
    token = store.create()

    time.sleep(0.12)

    assert not store.validate(token)


def test_repeated_wrong_passwords_slow_down_further_attempts() -> None:
    brake = LoginThrottle()
    for _ in range(4):
        brake.record_failure()
    assert brake.seconds_to_wait() == 0

    brake.record_failure()
    assert brake.seconds_to_wait() > 0

    brake.record_success()
    assert brake.seconds_to_wait() == 0


# --- the API ----------------------------------------------------------------------------------


def test_the_status_says_a_password_must_be_chosen_first(client: TestClient) -> None:
    body = client.get("/api/v1/auth/status").json()

    assert body == {"required": True, "configured": False, "authenticated": False}


def test_protected_routes_refuse_requests_without_a_token(client: TestClient) -> None:
    for path in ("/api/v1/ingestion/file-types", "/api/v1/library/documents", "/api/v1/profile"):
        response = client.get(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"

    assert client.get("/api/v1/health").status_code == 200


def test_a_made_up_token_is_refused(client: TestClient) -> None:
    response = client.get("/api/v1/ingestion/file-types", headers=_auth("made-up"))

    assert response.status_code == 401


def test_choosing_a_password_signs_you_in(client: TestClient) -> None:
    token = _setup(client)

    assert client.get("/api/v1/ingestion/file-types", headers=_auth(token)).status_code == 200
    status = client.get("/api/v1/auth/status", headers=_auth(token)).json()
    assert status == {"required": True, "configured": True, "authenticated": True}


def test_the_password_can_only_be_chosen_once(client: TestClient) -> None:
    _setup(client)

    again = client.post("/api/v1/auth/setup", json={"password": "another password"})

    assert again.status_code == 409


def test_a_weak_password_is_refused_at_setup(client: TestClient) -> None:
    response = client.post("/api/v1/auth/setup", json={"password": "short"})

    assert response.status_code == 422
    assert "at least" in response.json()["detail"]


def test_login_needs_the_right_password(client: TestClient) -> None:
    _setup(client)
    sessions.clear()

    wrong = client.post("/api/v1/auth/login", json={"password": "not the password"})
    right = client.post("/api/v1/auth/login", json={"password": PASSWORD})

    assert wrong.status_code == 401
    assert right.status_code == 200
    token = right.json()["token"]
    assert client.get("/api/v1/ingestion/file-types", headers=_auth(token)).status_code == 200


def test_logging_out_ends_the_session(client: TestClient) -> None:
    token = _setup(client)

    assert client.post("/api/v1/auth/logout", headers=_auth(token)).status_code == 204

    assert client.get("/api/v1/ingestion/file-types", headers=_auth(token)).status_code == 401


def test_too_many_wrong_passwords_are_slowed_down(client: TestClient) -> None:
    _setup(client)

    codes = [
        client.post("/api/v1/auth/login", json={"password": f"wrong password {i}"}).status_code
        for i in range(6)
    ]
    blocked = client.post("/api/v1/auth/login", json={"password": PASSWORD})

    assert codes[:4] == [401, 401, 401, 401]
    assert 429 in codes
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_changing_the_password_signs_out_other_sessions(client: TestClient) -> None:
    first = _setup(client)
    second = client.post("/api/v1/auth/login", json={"password": PASSWORD}).json()["token"]

    changed = client.post(
        "/api/v1/auth/change-password",
        headers=_auth(first),
        json={"current_password": PASSWORD, "new_password": "a brand new password"},
    )

    assert changed.status_code == 200
    fresh = changed.json()["token"]
    assert client.get("/api/v1/ingestion/file-types", headers=_auth(fresh)).status_code == 200
    assert client.get("/api/v1/ingestion/file-types", headers=_auth(second)).status_code == 401
    assert client.post("/api/v1/auth/login", json={"password": PASSWORD}).status_code == 401
    assert (
        client.post("/api/v1/auth/login", json={"password": "a brand new password"}).status_code
        == 200
    )


def test_changing_the_password_needs_the_current_one(client: TestClient) -> None:
    token = _setup(client)

    response = client.post(
        "/api/v1/auth/change-password",
        headers=_auth(token),
        json={"current_password": "not it at all", "new_password": "a brand new password"},
    )

    assert response.status_code == 401


def test_the_check_can_be_switched_off_for_development(
    client: TestClient, make_settings: Callable[..., Settings]
) -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings(access_required=False)

    assert client.get("/api/v1/ingestion/file-types").status_code == 200
    assert client.get("/api/v1/auth/status").json()["authenticated"] is True
