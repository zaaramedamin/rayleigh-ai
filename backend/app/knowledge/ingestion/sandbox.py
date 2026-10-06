"""Read a file in a separate process, with a time limit and a cap on what comes back.

A parser for a complicated format (PDF, Office files) is a lot of code reading bytes that someone
else made. A damaged or hostile file can make it hang, crash, or produce gigabytes. Run in its own
process, the worst outcome is a failed file with a reason code:

- ``timeout``          it did not finish in time and was stopped
- ``crashed``          the process died, or answered with something that is not a result
- ``too_large_output`` the text it produced was bigger than the cap
- anything the reader itself reports (``encrypted``, ``no_text``, ...) comes through unchanged

The file goes in on stdin, one JSON document comes back on stdout, and nothing else is trusted.
What went wrong inside is never reported, because it can contain pieces of the file.

Limits: time and output size. A cap on memory would need a Windows job object; it is not built
because the size limit on files already bounds the input, and the time limit stops the rest.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from app.knowledge.ingestion.parsers import Extracted, ParseError
from app.knowledge.ingestion.readers import OPTIONS_ENV

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
WORKER_MODULE = "app.knowledge.ingestion.sandbox_worker"

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 40 * 1024 * 1024

# The only environment variables the worker gets. Windows needs the first two to start Python.
_ENVIRONMENT_KEPT = ("SYSTEMROOT", "SYSTEMDRIVE", "PATH", "TEMP", "TMP")


def _worker_environment(options: Mapping[str, object] | None = None) -> dict[str, str]:
    kept = {name: os.environ[name] for name in _ENVIRONMENT_KEPT if name in os.environ}
    if options:
        kept[OPTIONS_ENV] = json.dumps(dict(options))
    kept["PYTHONIOENCODING"] = "utf-8"
    kept["PYTHONUTF8"] = "1"
    return kept


def _valid_page_starts(value: object, length: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ParseError("crashed")
    starts: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= length:
            raise ParseError("crashed")
        if starts and item < starts[-1]:
            raise ParseError("crashed")
        starts.append(item)
    return tuple(starts)


def run_in_sandbox(
    reader: str,
    data: bytes,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output: int = DEFAULT_MAX_OUTPUT_BYTES,
    options: Mapping[str, object] | None = None,
) -> Extracted:
    """Run `reader` ("module:function") on `data` in its own process.

    `reader` is always a name written in this program, never something read from a file or a
    request. `options` are plain settings for the reader. Raises ParseError with a reason code
    when the file cannot be read.
    """
    command = [sys.executable, "-m", WORKER_MODULE, str(max_output), reader]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window on Windows
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed command, our own module
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=BACKEND_DIR,
            env=_worker_environment(options),
            creationflags=flags,
        )
    except OSError as exc:
        raise ParseError("crashed") from exc
    try:
        # Room for the JSON around the text: escapes can make it somewhat longer than the text.
        output, _ = process.communicate(input=data, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        logger.warning("a file took too long to read and was stopped")
        raise ParseError("timeout") from exc
    except OSError as exc:  # the process died while the file was being handed over
        process.kill()
        process.communicate()
        raise ParseError("crashed") from exc

    if process.returncode != 0:
        raise ParseError("crashed")
    if len(output) > max_output + 1024:
        raise ParseError("too_large_output")
    try:
        body = json.loads(output)
    except ValueError as exc:
        raise ParseError("crashed") from exc
    if not isinstance(body, dict):
        raise ParseError("crashed")
    if "error" in body:
        code = body["error"]
        raise ParseError(code if isinstance(code, str) and code.isidentifier() else "crashed")
    text = body.get("text")
    if not isinstance(text, str):
        raise ParseError("crashed")
    return Extracted(text, _valid_page_starts(body.get("page_starts"), len(text)))
