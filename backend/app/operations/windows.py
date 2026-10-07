"""Read-only questions put to Windows, so `doctor` can explain what it finds.

- Is the drive that holds the library protected by BitLocker?
- Did Windows block files of this program from loading (Smart App Control, code integrity)?

Nothing here changes a setting, and nothing needs administrator rights. Everything answers `None`
(or an empty list) when Windows cannot be asked, so a failure to look never becomes a failure of
the program.
"""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath

_TIMEOUT_SECONDS = 20
# 3033: a file did not meet the signing level (Smart App Control).
# 3077: it broke the enforced code integrity policy.
BLOCK_EVENT_IDS = (3033, 3077)
LOOKBACK_DAYS = 14
_MAX_EVENTS = 2000
_NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
_DRIVE = re.compile(r"^[A-Za-z]:$")


@dataclass(frozen=True)
class BlockedFile:
    """A file Windows would not let this program load."""

    name: str  # shortened to where it sits in its package, e.g. "scipy\\fft\\_pocketfft\\x.pyd"
    count: int
    last_seen: datetime


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a Windows tool without a console window. None if it is missing, slow or fails."""
    if sys.platform != "win32":
        return None
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def bitlocker_protection(path: Path) -> int | None:
    """What Windows reports for BitLocker on the drive holding `path`, or None if it cannot tell.

    1 means protection is on. Any other number is passed on as it is: the meaning of the others
    differs between Windows versions, so callers should not rely on one.
    """
    if sys.platform != "win32":
        return None
    drive = path.resolve().drive
    if not _DRIVE.match(drive):  # a network share has no drive letter
        return None
    script = (
        "(New-Object -ComObject Shell.Application)"
        f".NameSpace('{drive}\\').Self.ExtendedProperty('System.Volume.BitLockerProtection')"
    )
    done = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    if done is None or done.returncode != 0:
        return None
    answer = done.stdout.strip()
    return int(answer) if re.fullmatch(r"\d{1,3}", answer) else None


def _data(event: ET.Element, *names: str) -> str:
    for item in event.iter(f"{_NS}Data"):
        if item.get("Name") in names:
            return (item.text or "").strip()
    return ""


def _is_ours(file: str, process: str) -> bool:
    """Does a block concern this program's Python files, as opposed to some other program's?"""
    lowered = file.lower()
    process_name = PureWindowsPath(process.lower()).name
    return (
        process_name.startswith("python")
        or "\\site-packages\\" in lowered
        or "\\.venv\\" in lowered
        or PureWindowsPath(lowered).name.startswith("alembic")
    )


def _short(file: str) -> str:
    """The file as it is known inside its package: no user name, no drive."""
    marker = "\\site-packages\\"
    index = file.lower().find(marker)
    return file[index + len(marker) :] if index >= 0 else PureWindowsPath(file).name


def parse_block_events(
    xml_text: str, *, now: datetime | None = None, days: int = LOOKBACK_DAYS
) -> list[BlockedFile]:
    """The blocked files of this program in `wevtutil ... /f:xml` output, most recent first."""
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    try:
        events = ET.fromstring(f"<Events>{xml_text}</Events>")
    except ET.ParseError:
        return []
    seen: dict[str, tuple[int, datetime]] = {}
    for event in events.iter(f"{_NS}Event"):
        stamp = event.find(f"{_NS}System/{_NS}TimeCreated")
        try:
            when = datetime.fromisoformat(
                (stamp.get("SystemTime") or "") if stamp is not None else ""
            )
        except ValueError:
            continue
        file = _data(event, "File Name", "FileNameBuffer")
        if when < cutoff or not file:
            continue
        if not _is_ours(file, _data(event, "Process Name", "ProcessNameBuffer")):
            continue
        name = _short(file)
        count, latest = seen.get(name, (0, when))
        seen[name] = (count + 1, max(latest, when))
    blocked = [BlockedFile(name, count, latest) for name, (count, latest) in seen.items()]
    return sorted(blocked, key=lambda b: b.last_seen, reverse=True)


def recent_blocked_files(days: int = LOOKBACK_DAYS) -> list[BlockedFile] | None:
    """Files of this program that Windows blocked in the last `days` days, or None if the block
    log cannot be read."""
    milliseconds = days * 24 * 3600 * 1000
    ids = " or ".join(f"EventID={i}" for i in BLOCK_EVENT_IDS)
    done = _run(
        [
            "wevtutil",
            "qe",
            "Microsoft-Windows-CodeIntegrity/Operational",
            f"/q:*[System[({ids}) and TimeCreated[timediff(@SystemTime) <= {milliseconds}]]]",
            f"/c:{_MAX_EVENTS}",
            "/rd:true",
            "/f:xml",
        ]
    )
    if done is None or done.returncode != 0:
        return None
    return parse_block_events(done.stdout, days=days)


def summarise(blocked: Iterable[BlockedFile], limit: int = 4) -> str:
    """A short list: "scipy\\...\\x.pyd (3 times), ..." with how many more there are."""
    items = list(blocked)
    shown = ", ".join(
        f"{b.name}" + (f" ({b.count} times)" if b.count > 1 else "") for b in items[:limit]
    )
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def scikit_learn_loads() -> bool | None:
    """Can scikit-learn be imported here? Asked in a separate process, so nothing changes in this
    one. None if it could not be asked."""
    no_window = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        done = subprocess.run(
            [sys.executable, "-c", "import sklearn.metrics"],
            capture_output=True,
            timeout=120,
            check=False,
            creationflags=no_window,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode == 0


def embedding_library_loads() -> tuple[bool | None, str]:
    """Can the library that turns text into vectors be loaded here?

    Asked in a separate process, so nothing changes in this one, and by loading it the way the
    application does (including the scikit-learn placeholder). Returns (True, "") when it loads,
    (False, reason) with the last line of the error when it does not, and (None, "") when it could
    not be asked. Without it nothing can be searched or answered from the notes, which is why the
    doctor asks.
    """
    program = (
        "from app.ai.embeddings.compat import import_sentence_transformer; "
        "import_sentence_transformer()"
    )
    package_root = Path(__file__).resolve().parents[2]  # the folder that holds the `app` package
    no_window = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        done = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
            cwd=package_root,
            creationflags=no_window,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""
    if done.returncode == 0:
        return True, ""
    lines = [line.strip() for line in done.stderr.splitlines() if line.strip()]
    return False, (lines[-1] if lines else "it could not be imported")[:300]
