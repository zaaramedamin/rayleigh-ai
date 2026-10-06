"""The other side of the parser sandbox: reads one file from stdin, writes the result to stdout.

Started by sandbox.py as `python -m app.knowledge.ingestion.sandbox_worker <module:function>`.
Nothing is read from the network or the environment here, and the only thing written to stdout is
one JSON document. Anything unexpected ends the process with a non-zero exit code; the parent
turns that into a plain "crashed" failure and never shows what happened inside.
"""

import importlib
import json
import socket
import sys
from typing import Any

from app.knowledge.ingestion.parsers import Extracted, ParseError

# Parsers read bytes and return text. They have no business touching the network, so a parser
# that tries (a library that "phones home") fails instead of reaching out.
_REFUSED = "network access is not allowed while reading a file"


def _refuse_network(*_args: Any, **_kwargs: Any) -> None:
    raise OSError(_REFUSED)


def _load_reader(spec: str) -> Any:
    module_name, _, function_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), function_name)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or ":" not in argv[1]:
        return 2
    max_output = int(argv[0])
    socket.socket.connect = _refuse_network  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse_network  # type: ignore[method-assign,assignment]
    data = sys.stdin.buffer.read()
    try:
        extracted = _load_reader(argv[1])(data)
        if not isinstance(extracted, Extracted):
            return 3
        body: dict[str, Any] = {
            "text": extracted.text,
            "page_starts": list(extracted.page_starts) if extracted.page_starts else None,
        }
    except ParseError as exc:
        body = {"error": str(exc)}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if len(payload) > max_output:
        payload = json.dumps({"error": "too_large_output"}).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
