"""Which folders may be read, and the checks a new folder must pass before it is allowed."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.knowledge.library.state import UpdateState, load_state

# Folders that hold credentials or keys, wherever they are.
_SECRET_FOLDER_NAMES = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".azure", ".docker"})
_POSIX_SYSTEM = ("/etc", "/bin", "/sbin", "/usr", "/var", "/proc", "/sys", "/dev", "/boot", "/lib")
_WINDOWS_SYSTEM_VARIABLES = (
    "WINDIR",
    "SystemRoot",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramData",
)


class FolderError(ValueError):
    """The folder cannot be added. The message is safe to show to the owner."""


@dataclass(frozen=True)
class AllowedFolder:
    path: Path
    origin: Literal["env", "ui"]


def _system_folders() -> list[Path]:
    found = [Path(p) for p in _POSIX_SYSTEM]
    for name in _WINDOWS_SYSTEM_VARIABLES:
        value = os.environ.get(name)
        if value:
            found.append(Path(value))
    return found


def _same_or_inside(path: Path, parent: Path) -> bool:
    try:
        return path == parent or path.is_relative_to(parent)
    except ValueError:
        return False


def allowed_folders(settings: Settings) -> list[AllowedFolder]:
    """Every folder that may be read: those from .env first, then those added in the interface."""
    groups: list[tuple[Literal["env", "ui"], list[Path]]] = [
        ("env", list(settings.allowed_folders)),
        ("ui", [Path(p) for p in load_state(settings.data_dir).folders]),
    ]
    result: list[AllowedFolder] = []
    seen: set[Path] = set()
    for origin, paths in groups:
        for path in paths:
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            result.append(AllowedFolder(path=path, origin=origin))
    return result


def validate_new_folder(raw: str, settings: Settings) -> Path:
    """Return the resolved folder if it is safe to allow, or raise FolderError."""
    text = raw.strip().strip('"')
    if not text:
        raise FolderError("Enter the full path of a folder.")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise FolderError("Use the full path, for example C:\\Users\\you\\Documents\\notes.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FolderError("That folder does not exist.") from exc
    if not resolved.is_dir():
        raise FolderError("That path is a file, not a folder.")
    if resolved == Path(resolved.anchor):
        raise FolderError("A whole drive cannot be added. Choose a specific folder.")
    home = Path.home().resolve()
    if resolved == home:
        raise FolderError("Your whole user folder cannot be added. Choose a folder inside it.")
    if any(_same_or_inside(resolved, system.resolve()) for system in _system_folders()):
        raise FolderError("System folders cannot be added.")
    if _same_or_inside(resolved, home / "AppData") or any(
        part.lower() in _SECRET_FOLDER_NAMES for part in resolved.parts
    ):
        raise FolderError("Folders that hold application data or keys cannot be added.")
    for own in (settings.data_dir, settings.models_dir):
        own = own.resolve()
        if _same_or_inside(resolved, own) or _same_or_inside(own, resolved):
            raise FolderError(
                "That folder contains (or is inside) Reyleight's own data, which is never read."
            )
    if any(resolved == entry.path.resolve() for entry in allowed_folders(settings)):
        raise FolderError("That folder is already on the list.")
    return resolved


def add_folder(settings: Settings, raw: str) -> Path:
    resolved = validate_new_folder(raw, settings)
    with UpdateState(settings.data_dir) as state:
        state.folders.append(str(resolved))
    return resolved


def remove_folder(settings: Settings, raw: str) -> Path:
    """Remove a folder that was added in the interface. Folders from .env cannot be removed here."""
    try:
        target = Path(raw.strip().strip('"')).resolve()
    except OSError as exc:
        raise FolderError("That folder is not on the list.") from exc
    env_paths = {p.resolve() for p in settings.allowed_folders}
    if target in env_paths:
        raise FolderError("That folder is set in the .env file. Remove it there.")
    with UpdateState(settings.data_dir) as state:
        kept = [p for p in state.folders if Path(p).resolve() != target]
        if len(kept) == len(state.folders):
            raise FolderError("That folder is not on the list.")
        state.folders = kept
    return target
