"""What the agent is told about this computer, so it does not have to guess.

A model that is asked to "open my downloads folder" and does not know where that is will make a
path up, and a made-up path is a wrong action waiting for the owner's approval. So every task starts
with a short note of the facts it may use: today's date, whose computer this is, and the real
places the owner means by "my documents", "my downloads" or "my notes". The model is told to use
these exact paths and to ask when it needs one that is not here.

The places come from Windows itself (the "known folders", which follow a Documents folder moved to
OneDrive, or renamed in another language), and from the folders the owner added to the library.
Only folders that exist are listed. The note is built fresh for every task and is not written to
the log.
"""

import ctypes
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

# Windows "known folder" ids (KNOWNFOLDERID).
KNOWN_FOLDERS: dict[str, str] = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "Videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
}

FolderLookup = Callable[[str], Path | None]


def windows_folder(name: str) -> Path | None:
    """Where Windows says the folder `name` is, or None (not Windows, or it does not say)."""
    guid = KNOWN_FOLDERS.get(name)
    windll = getattr(ctypes, "windll", None)
    if guid is None or windll is None or sys.platform != "win32":
        return None

    class GUID(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - the layout ctypes needs
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    wanted = uuid.UUID(guid)
    folder_id = GUID(
        wanted.time_low,
        wanted.time_mid,
        wanted.time_hi_version,
        (ctypes.c_ubyte * 8)(*wanted.bytes[8:]),
    )
    pointer = ctypes.c_wchar_p()
    try:
        result = windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(folder_id), 0, None, ctypes.byref(pointer)
        )
        if result != 0 or not pointer.value:
            return None
        return Path(pointer.value)
    except (OSError, AttributeError):
        return None
    finally:
        windll.ole32.CoTaskMemFree(pointer)


def _folder(name: str, lookup: FolderLookup, home: Path) -> Path | None:
    for candidate in (lookup(name), home / name):
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def describe_computer(
    *,
    notes_folders: Sequence[Path] = (),
    today: date | None = None,
    home: Path | None = None,
    lookup: FolderLookup = windows_folder,
    now: datetime | None = None,
) -> str:
    """The facts the agent may use, one per line, with the rule about not inventing the rest."""
    home = home if home is not None else Path.home()
    when = (now or datetime.now()).date() if today is None else today
    lines = [
        "About this computer (use these exact paths and facts; never invent a path, a name or "
        "an address):",
        f"- Today is {when.strftime('%A')} {when.day} {when.strftime('%B %Y')}.",
        f"- It runs Windows. The user's home folder is {home}.",
    ]
    for name in KNOWN_FOLDERS:
        path = _folder(name, lookup, home)
        if path is not None:
            lines.append(f"- The user's {name} folder is {path}.")
    existing = [folder for folder in notes_folders if folder.is_dir()]
    if existing:
        lines.append(
            "- The user's notes are in: " + "; ".join(str(folder) for folder in existing) + "."
        )
    return "\n".join(lines)
