"""Keep the local API from being driven by a web page you happen to visit.

The API listens on this computer only, but a browser on this computer can be told by any website to
send it requests, so three more rules apply to every request:

- **The Host header must name this computer** (`127.0.0.1`, `localhost`, `::1`, or a name listed in
  ALLOWED_HOSTS). That is what stops DNS rebinding, where a hostile site's name is made to point at
  127.0.0.1 so the browser treats the local API as that site's own.
- **An Origin header, when the browser sends one, must be this computer too** (or listed in
  CORS_ORIGINS). A page on another site always sends its own Origin on a cross-site request, so it
  is refused before anything else happens.
- **Requests are limited in size and in number**, so a runaway page or script cannot flood the
  server or make it read an endless body.

Every refusal is a short JSON message, never a stack trace. Nothing in a request is logged.
"""

import json
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any
from urllib.parse import urlsplit

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def host_name(value: str) -> str:
    """The name in a Host header or an Origin, without scheme, port or brackets."""
    text = value.strip().lower()
    if "://" in text:
        text = urlsplit(text).netloc
    if text.startswith("["):  # [::1]:8000
        return text[1 : text.find("]")] if "]" in text else text
    return text.rsplit(":", 1)[0] if text.count(":") == 1 else text


class RequestRate:
    """At most `limit` requests in any 60 seconds, counted over everything this server receives."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._clock = clock
        self._seen: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, int]:
        """(allowed, seconds until the next request would be allowed)."""
        now = self._clock()
        with self._lock:
            while self._seen and now - self._seen[0] >= 60:
                self._seen.popleft()
            if len(self._seen) >= self._limit:
                return False, max(1, int(60 - (now - self._seen[0])) + 1)
            self._seen.append(now)
            return True, 0


class LocalOnlyGuard:
    """An ASGI middleware that applies the rules above before the application sees a request."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        extra_hosts: Iterable[str] = (),
        cors_origins: Iterable[str] = (),
        max_body_bytes: int = 8 * 1024 * 1024,
        requests_per_minute: int = 1200,
        exempt_paths: Iterable[str] = ("/api/v1/health",),
    ) -> None:
        self._app = app
        self._hosts = LOOPBACK_NAMES | {host_name(h) for h in extra_hosts}
        self._origins = {origin.strip().lower().rstrip("/") for origin in cors_origins}
        self._max_body = max_body_bytes
        self._rate = RequestRate(requests_per_minute)
        self._exempt = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}

        if host_name(headers.get("host", "")) not in self._hosts:
            await _refuse(
                send, 400, "This address is not allowed. Use http://127.0.0.1 or localhost."
            )
            return
        origin = headers.get("origin")
        if origin is not None and not self._origin_allowed(origin):
            await _refuse(send, 403, "Requests from other websites are not allowed.")
            return
        if scope["path"] not in self._exempt:
            allowed, wait = self._rate.allow()
            if not allowed:
                await _refuse(send, 429, "Too many requests. Slow down.", retry_after=wait)
                return
        declared = headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self._max_body):
            await _refuse(send, 413, "That request is too large.")
            return

        received = 0
        too_large = False

        async def limited_receive() -> MutableMapping[str, Any]:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body:
                    too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        started = False

        async def guarded_send(message: MutableMapping[str, Any]) -> None:
            nonlocal started
            if too_large and not started:
                return  # the body was cut short; the refusal below replaces any answer
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self._app(scope, limited_receive, guarded_send)
        if too_large and not started:
            await _refuse(send, 413, "That request is too large.")

    def _origin_allowed(self, origin: str) -> bool:
        if origin.strip().lower().rstrip("/") in self._origins:
            return True
        return host_name(origin) in self._hosts


async def _refuse(send: Send, status: int, detail: str, *, retry_after: int | None = None) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if retry_after is not None:
        headers.append((b"retry-after", str(retry_after).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
