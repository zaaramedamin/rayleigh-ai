"""Readers used only by the sandbox tests. Each one misbehaves in a different way."""

import os
import socket
import time

from app.knowledge.ingestion.parsers import Extracted, ParseError


def echo(data: bytes) -> Extracted:
    return Extracted(data.decode("utf-8"))


def two_pages(data: bytes) -> Extracted:
    text = data.decode("utf-8")
    middle = text.index("|")
    return Extracted(text.replace("|", "\n"), (0, middle + 1))


def refuses(data: bytes) -> Extracted:
    raise ParseError("encrypted")


def hangs(data: bytes) -> Extracted:
    time.sleep(120)
    return Extracted("never")


def dies(data: bytes) -> Extracted:
    os._exit(7)


def raises(data: bytes) -> Extracted:
    raise RuntimeError("an unexpected bug with a secret inside: " + data.decode("utf-8", "replace"))


def huge(data: bytes) -> Extracted:
    return Extracted("x" * 5_000_000)


def wrong_type(data: bytes) -> str:
    return "not an Extracted"


def bad_pages(data: bytes) -> Extracted:
    return Extracted("short", (0, 999))


def phones_home(data: bytes) -> Extracted:
    connection = socket.socket()
    try:
        connection.connect(("127.0.0.1", 9))
    finally:
        connection.close()
    return Extracted("reached out")
