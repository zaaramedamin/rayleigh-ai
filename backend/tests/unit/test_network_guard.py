import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from app.evaluation.network_guard import NetworkBlocked, NetworkGuard, destination_is_local


@pytest.fixture
def local_server() -> Iterator[int]:
    """A tiny TCP server on this machine that accepts one connection."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def accept() -> None:
        try:
            conn, _ = server.accept()
            conn.close()
        except OSError:
            pass

    threading.Thread(target=accept, daemon=True).start()
    yield server.getsockname()[1]
    server.close()


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.5.4.3", "::1", "localhost", "LOCALHOST", "localhost.", "", None]
)
def test_this_machine_is_local(host: object) -> None:
    assert destination_is_local(host)


@pytest.mark.parametrize(
    "host",
    [
        "8.8.8.8",
        "192.168.1.20",
        "10.0.0.1",
        "example.com",
        "localhost.evil.com",
        "0.0.0.0",
        b"x.org",
        5,
    ],
)
def test_everything_else_is_not_local(host: object) -> None:
    assert not destination_is_local(host)


def test_an_outside_connection_is_refused_and_recorded() -> None:
    with NetworkGuard() as guard:
        with pytest.raises(NetworkBlocked, match="203.0.113.1:80"):
            socket.socket().connect(("203.0.113.1", 80))
        with pytest.raises(NetworkBlocked):
            socket.socket().connect_ex(("198.51.100.7", 443))

    assert guard.blocked == ["203.0.113.1:80", "198.51.100.7:443"]


def test_looking_up_an_outside_name_is_refused() -> None:
    with NetworkGuard() as guard:
        with pytest.raises(NetworkBlocked, match="example.com"):
            socket.getaddrinfo("example.com", 443)
        with pytest.raises(NetworkBlocked):
            socket.gethostbyname("huggingface.co")

    assert [b.split(":")[0] for b in guard.blocked] == ["example.com", "huggingface.co"]


def test_sending_a_datagram_outside_is_refused() -> None:
    with NetworkGuard() as guard:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with pytest.raises(NetworkBlocked):
                udp.sendto(b"hi", ("8.8.8.8", 53))
        finally:
            udp.close()

    assert guard.blocked == ["8.8.8.8:53"]


def test_a_connection_to_this_machine_works_and_is_counted(local_server: int) -> None:
    with NetworkGuard() as guard:
        socket.create_connection(("127.0.0.1", local_server), timeout=2).close()

    assert guard.blocked == []
    assert guard.local == [f"127.0.0.1:{local_server}"]


def test_http_libraries_are_covered_too() -> None:
    with NetworkGuard() as guard:
        with pytest.raises(urllib.error.URLError) as raised:
            urllib.request.urlopen("http://example.com/", timeout=2)  # noqa: S310

    assert isinstance(raised.value.reason, NetworkBlocked)
    assert any(b.startswith("example.com") for b in guard.blocked)


def test_the_original_functions_are_restored_afterwards() -> None:
    before = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)

    with NetworkGuard():
        assert socket.socket.connect is not before[0]
    after = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)

    assert after == before


def test_the_guard_is_removed_even_if_the_code_inside_raises() -> None:
    original = socket.socket.connect

    with pytest.raises(RuntimeError), NetworkGuard():
        raise RuntimeError("boom")

    assert socket.socket.connect is original


def test_guards_can_be_nested() -> None:
    original = socket.socket.connect

    with NetworkGuard() as outer:
        with NetworkGuard() as inner:
            with pytest.raises(NetworkBlocked):
                socket.socket().connect(("203.0.113.9", 80))
        assert inner.blocked == ["203.0.113.9:80"]
        with pytest.raises(NetworkBlocked):
            socket.socket().connect(("203.0.113.10", 80))

    assert outer.blocked == ["203.0.113.10:80"]
    assert socket.socket.connect is original


def test_the_summary_lists_what_was_blocked_and_what_stayed_local(local_server: int) -> None:
    with NetworkGuard() as guard:
        socket.create_connection(("127.0.0.1", local_server), timeout=2).close()
        with pytest.raises(NetworkBlocked):
            socket.socket().connect(("203.0.113.1", 80))

    summary = guard.summary()
    assert "outbound connection attempts blocked: 1 (203.0.113.1:80)" in summary
    assert f"local connections used: 1 (127.0.0.1:{local_server})" in summary


def test_a_clean_run_reports_zero() -> None:
    with NetworkGuard() as guard:
        pass

    assert guard.summary().startswith("outbound connection attempts blocked: 0")
