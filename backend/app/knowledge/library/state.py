"""DATA_DIR/library.json: small bookkeeping the database does not hold.

- folders:  folders added through the interface (folders from ALLOWED_FOLDERS in .env are not
            stored here and cannot be removed from the interface);
- sources:  content hash -> the path the file was last read from, so the interface can say
            where a document came from;
- excluded: content hashes of documents the owner removed, so a later scan does not bring them
            straight back.

The file holds paths and hashes, never note text. A missing or damaged file reads as empty.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LIBRARY_FILENAME = "library.json"
_LOCK = threading.RLock()


@dataclass
class LibraryState:
    folders: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)


def _path(data_dir: Path) -> Path:
    return data_dir / LIBRARY_FILENAME


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def load_state(data_dir: Path) -> LibraryState:
    try:
        raw = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LibraryState()
    if not isinstance(raw, dict):
        return LibraryState()
    sources = raw.get("sources")
    return LibraryState(
        folders=_strings(raw.get("folders")),
        sources={
            key: value
            for key, value in (sources.items() if isinstance(sources, dict) else [])
            if isinstance(key, str) and isinstance(value, str)
        },
        excluded=_strings(raw.get("excluded")),
    )


def _save(data_dir: Path, state: LibraryState) -> None:
    target = _path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(
                {"folders": state.folders, "sources": state.sources, "excluded": state.excluded},
                tmp,
                indent=1,
            )
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class UpdateState:
    """Read-modify-write the state under one lock: `with UpdateState(data_dir) as state: ...`."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def __enter__(self) -> LibraryState:
        _LOCK.acquire()
        self._state = load_state(self._data_dir)
        return self._state

    def __exit__(self, exc_type: object, *_rest: object) -> None:
        try:
            if exc_type is None:
                _save(self._data_dir, self._state)
        finally:
            _LOCK.release()
