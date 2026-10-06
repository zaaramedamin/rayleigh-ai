"""Sign-in sessions and the brake on repeated wrong passwords. Both live in memory only."""

import hashlib
import secrets
import threading
import time

SESSION_IDLE_SECONDS = 8 * 60 * 60
MAX_FREE_FAILURES = 4
MAX_DELAY_SECONDS = 60


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SessionStore:
    """Random bearer tokens. Only their hashes are kept, and a restart signs everyone out."""

    def __init__(self, idle_seconds: float = SESSION_IDLE_SECONDS) -> None:
        self._idle = idle_seconds
        self._expires: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge()
            self._expires[_digest(token)] = time.monotonic() + self._idle
        return token

    def validate(self, token: str) -> bool:
        key = _digest(token)
        now = time.monotonic()
        with self._lock:
            expiry = self._expires.get(key)
            if expiry is None or expiry < now:
                self._expires.pop(key, None)
                return False
            self._expires[key] = now + self._idle  # activity keeps the session open
            return True

    def revoke(self, token: str) -> None:
        with self._lock:
            self._expires.pop(_digest(token), None)

    def clear(self) -> None:
        with self._lock:
            self._expires.clear()

    def _purge(self) -> None:
        now = time.monotonic()
        for key in [k for k, expiry in self._expires.items() if expiry < now]:
            del self._expires[key]


class LoginThrottle:
    """After a few wrong passwords, each further attempt must wait longer (up to a minute)."""

    def __init__(self) -> None:
        self._failures = 0
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    def seconds_to_wait(self) -> int:
        with self._lock:
            return max(0, int(self._blocked_until - time.monotonic() + 0.999))

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            extra = self._failures - MAX_FREE_FAILURES
            if extra > 0:
                delay = min(MAX_DELAY_SECONDS, 2**extra)
                self._blocked_until = time.monotonic() + delay

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._blocked_until = 0.0

    def reset(self) -> None:
        self.record_success()


sessions = SessionStore()
throttle = LoginThrottle()
