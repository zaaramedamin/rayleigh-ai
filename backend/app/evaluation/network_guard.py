"""Prove that a piece of code never talks to anything outside this computer.

Inside `with NetworkGuard():` every attempt to open a network connection or resolve a name is
checked. Connections to this machine (loopback, such as the local Ollama server) are allowed and
counted. Anything else raises `NetworkBlocked` and is recorded, so the caller can report exactly
what the code tried to reach.

Limits: this watches Python's `socket` module, which covers urllib, http.client, requests,
httpx, huggingface_hub and everything else written in Python. It cannot see a native library that
opens its own sockets without Python. For that, also run the check with the network switched off
(see docs/offline-check.md).
"""

import ipaddress
import socket
from types import TracebackType
from typing import Any, Self

_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain"})


class NetworkBlocked(ConnectionError):
    """A connection to something outside this computer was attempted and refused."""


def destination_is_local(host: object) -> bool:
    """True for loopback addresses and the name `localhost`; False for anything else."""
    if host is None or host == "":
        return True  # "no host" means the local machine in getaddrinfo()
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host.lower().rstrip(".") in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _describe(address: Any) -> str:
    if isinstance(address, tuple) and address:
        port = f":{address[1]}" if len(address) > 1 else ""
        return f"{address[0]}{port}"
    return str(address)


class NetworkGuard:
    def __init__(self) -> None:
        self.blocked: list[str] = []  # destinations that were refused
        self.local: list[str] = []  # connections to this machine that were allowed
        self._originals: dict[str, Any] = {}

    # --- checks -------------------------------------------------------------------------------

    def _check_address(self, address: Any) -> None:
        if isinstance(address, (str, bytes)):  # a Unix socket path: stays on this machine
            self.local.append("unix socket")
            return
        host = address[0] if isinstance(address, tuple) and address else address
        if destination_is_local(host):
            self.local.append(_describe(address))
            return
        self.blocked.append(_describe(address))
        raise NetworkBlocked(f"network access to {_describe(address)} is blocked (offline mode)")

    def _check_host(self, host: object, port: object) -> None:
        if destination_is_local(host):
            return
        destination = f"{host}:{port}" if port else str(host)
        self.blocked.append(destination)
        raise NetworkBlocked(f"looking up {destination} is blocked (offline mode)")

    # --- install / remove ---------------------------------------------------------------------

    def __enter__(self) -> Self:
        guard = self
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_sendto = socket.socket.sendto
        original_getaddrinfo = socket.getaddrinfo
        original_gethostbyname = socket.gethostbyname
        self._originals = {
            "connect": original_connect,
            "connect_ex": original_connect_ex,
            "sendto": original_sendto,
            "getaddrinfo": original_getaddrinfo,
            "gethostbyname": original_gethostbyname,
        }

        def connect(sock: socket.socket, address: Any) -> Any:
            guard._check_address(address)
            return original_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> Any:
            guard._check_address(address)
            return original_connect_ex(sock, address)

        def sendto(sock: socket.socket, data: Any, *args: Any) -> Any:
            if args:  # sendto(data, address) or sendto(data, flags, address)
                guard._check_address(args[-1])
            return original_sendto(sock, data, *args)

        def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
            guard._check_host(host, port)
            return original_getaddrinfo(host, port, *args, **kwargs)

        def gethostbyname(host: Any) -> Any:
            guard._check_host(host, None)
            return original_gethostbyname(host)

        # setattr: these are deliberate replacements of standard-library functions, whose exact
        # signatures (overloads, buffer types) are not worth reproducing for the type checker.
        setattr(socket.socket, "connect", connect)  # noqa: B010
        setattr(socket.socket, "connect_ex", connect_ex)  # noqa: B010
        setattr(socket.socket, "sendto", sendto)  # noqa: B010
        setattr(socket, "getaddrinfo", getaddrinfo)  # noqa: B010
        setattr(socket, "gethostbyname", gethostbyname)  # noqa: B010
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        setattr(socket.socket, "connect", self._originals["connect"])  # noqa: B010
        setattr(socket.socket, "connect_ex", self._originals["connect_ex"])  # noqa: B010
        setattr(socket.socket, "sendto", self._originals["sendto"])  # noqa: B010
        setattr(socket, "getaddrinfo", self._originals["getaddrinfo"])  # noqa: B010
        setattr(socket, "gethostbyname", self._originals["gethostbyname"])  # noqa: B010

    # --- reporting ----------------------------------------------------------------------------

    def summary(self) -> str:
        blocked = sorted(set(self.blocked))
        local = sorted(set(self.local))
        lines = [
            f"outbound connection attempts blocked: {len(self.blocked)}"
            + (f" ({', '.join(blocked)})" if blocked else ""),
            f"local connections used: {len(self.local)}"
            + (f" ({', '.join(local)})" if local else ""),
        ]
        return "\n".join(lines)
