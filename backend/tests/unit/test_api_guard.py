"""Slice 5.2: a web page cannot drive the local API."""

import asyncio
import json
from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.guard import LocalOnlyGuard, RequestRate, host_name
from app.core.config import Settings, get_settings
from app.main import app, create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def guarded(**options: object) -> TestClient:
    """A tiny app behind the guard, so each rule can be tried on its own."""
    inner = FastAPI()

    @inner.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @inner.get("/thing")
    def thing() -> dict[str, str]:
        return {"ok": "yes"}

    @inner.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    inner.add_middleware(LocalOnlyGuard, **options)  # type: ignore[arg-type]
    return TestClient(inner, base_url="http://127.0.0.1:8000")


# --- the Host header (DNS rebinding) --------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.1:8000", "localhost", "LOCALHOST:5173", "[::1]:8000", "[::1]"]
)
def test_this_computers_own_names_are_accepted(host: str) -> None:
    assert guarded().get("/thing", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8000",
        "127.0.0.1.evil.example",
        "localhost.evil.com",
        "10.0.0.5",
        "",
    ],
)
def test_any_other_host_is_refused(host: str) -> None:
    response = guarded().get("/thing", headers={"Host": host})

    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]


def test_a_name_in_the_settings_is_accepted_too() -> None:
    client = guarded(extra_hosts=["reyleight.lan"])

    assert client.get("/thing", headers={"Host": "reyleight.lan:8000"}).status_code == 200
    assert client.get("/thing", headers={"Host": "other.lan"}).status_code == 400


def test_the_health_check_is_also_protected_from_a_wrong_host() -> None:
    assert guarded().get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 400


def test_host_names_are_read_without_port_scheme_or_brackets() -> None:
    assert host_name("Example.COM:8080") == "example.com"
    assert host_name("http://127.0.0.1:5173") == "127.0.0.1"
    assert host_name("[::1]:8000") == "::1"
    assert host_name("::1") == "::1"
    assert host_name("localhost") == "localhost"


# --- the Origin header (other websites) -----------------------------------------------------------


def test_a_request_from_another_website_is_refused() -> None:
    response = guarded().get("/thing", headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert "other websites" in response.json()["detail"]


def test_a_cross_site_post_is_refused_before_it_reaches_the_app() -> None:
    reached: list[int] = []
    inner = FastAPI()

    @inner.post("/act")
    def act() -> dict[str, str]:
        reached.append(1)
        return {"done": "yes"}

    inner.add_middleware(LocalOnlyGuard)
    client = TestClient(inner, base_url="http://127.0.0.1")

    response = client.post("/act", headers={"Origin": "https://evil.example"}, json={})

    assert response.status_code == 403 and reached == []


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:5173", "http://localhost:5173", "http://[::1]:5173"]
)
def test_this_computers_own_pages_may_call_it(origin: str) -> None:
    assert guarded().get("/thing", headers={"Origin": origin}).status_code == 200


def test_a_listed_origin_is_accepted_and_a_lookalike_is_not() -> None:
    client = guarded(cors_origins=["https://app.reyleight.test"])

    assert client.get("/thing", headers={"Origin": "https://app.reyleight.test"}).status_code == 200
    assert (
        client.get("/thing", headers={"Origin": "https://app.reyleight.test.evil.com"}).status_code
        == 403
    )


def test_a_request_with_no_origin_is_fine() -> None:
    assert guarded().get("/thing").status_code == 200  # the command line, a script, a same-site GET


# --- size and rate --------------------------------------------------------------------------------


def test_a_declared_body_that_is_too_big_is_refused_at_once() -> None:
    response = guarded(max_body_bytes=100).post("/echo", content=b"x" * 101)

    assert response.status_code == 413 and "too large" in response.json()["detail"]


def test_a_body_under_the_limit_goes_through() -> None:
    response = guarded(max_body_bytes=100).post("/echo", content=b"x" * 100)

    assert response.status_code == 200 and response.json() == {"bytes": 100}


def test_a_body_that_lies_about_its_size_is_cut_off() -> None:
    """The header says 10 bytes; the stream keeps going. The guard counts what really arrives."""
    inner = FastAPI()

    @inner.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    wrapped = LocalOnlyGuard(inner, max_body_bytes=100)
    sent: list[dict[str, object]] = []
    chunks = [b"x" * 60, b"x" * 60, b"x" * 60]

    async def receive() -> dict[str, object]:
        body = chunks.pop(0) if chunks else b""
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/echo",
        "raw_path": b"/echo",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1"), (b"content-length", b"10")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(wrapped(scope, receive, send))

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"].startswith("That request is too large")  # type: ignore[arg-type]


def test_too_many_requests_are_slowed_down_with_a_wait_time() -> None:
    client = guarded(requests_per_minute=3)

    codes = [client.get("/thing").status_code for _ in range(5)]
    blocked = client.get("/thing")

    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert 1 <= int(blocked.headers["retry-after"]) <= 61


def test_the_health_check_does_not_count_against_the_limit() -> None:
    client = guarded(requests_per_minute=2)

    codes = [client.get("/api/v1/health").status_code for _ in range(10)]

    assert set(codes) == {200}


def test_the_rate_window_slides() -> None:
    now = [0.0]
    rate = RequestRate(2, clock=lambda: now[0])

    assert rate.allow() == (True, 0) and rate.allow() == (True, 0)
    refused, wait = rate.allow()
    assert refused is False and 1 <= wait <= 61
    now[0] = 61.0
    assert rate.allow() == (True, 0)


# --- the real application -------------------------------------------------------------------------


def test_the_application_refuses_a_forged_host(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Host": "attacker.example"})

    assert response.status_code == 400


def test_the_application_refuses_another_websites_request_even_with_a_valid_login(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers={"Origin": "https://attacker.example", "Authorization": "Bearer anything"},
        json={"message": "hello"},
    )

    assert response.status_code == 403


def test_the_application_answers_its_own_pages_without_cross_site_headers(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5173"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers  # no cross-site access by default


def test_a_preflight_from_another_site_gets_no_permission(client: TestClient) -> None:
    response = client.options(
        "/api/v1/chat",
        headers={"Origin": "https://attacker.example", "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 403


def test_a_configured_origin_gets_cors_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("CORS_ORIGINS", "http://127.0.0.1:5173, https://app.reyleight.test")
    try:
        configured = create_app()
        client = TestClient(configured)

        allowed = client.get("/api/v1/health", headers={"Origin": "https://app.reyleight.test"})
        other = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    finally:
        monkeypatch.delenv("CORS_ORIGINS")
        get_settings.cache_clear()

    assert allowed.headers["access-control-allow-origin"] == "https://app.reyleight.test"
    assert "access-control-allow-credentials" not in allowed.headers
    assert other.status_code == 403


def test_an_unexpected_error_is_a_plain_message_without_a_trace(
    make_settings: Callable[..., Settings], caplog: pytest.LogCaptureFixture
) -> None:
    broken = create_app()

    @broken.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret note text: the garage code is 4821")

    client = TestClient(broken, raise_server_exceptions=False)

    with caplog.at_level("DEBUG"):
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Something went wrong on this computer. See the server log."
    }
    assert "4821" not in response.text and "Traceback" not in response.text
    assert "4821" not in caplog.text  # only the kind of error is logged, never what it said


def test_the_settings_for_names_and_origins_are_read_from_a_comma_list() -> None:
    settings = Settings(_env_file=None, allowed_hosts="a.lan, b.lan", cors_origins="http://x:1")  # type: ignore[arg-type]

    assert settings.allowed_hosts == ["a.lan", "b.lan"]
    assert settings.cors_origins == ["http://x:1"]
