import hashlib
import mimetypes
import os
import re
import tempfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import Document

FILES_SUBDIR = "files"
MAX_DISPLAY_NAME_LENGTH = 255


class UnsafePathError(ValueError):
    """Raised when a path would resolve outside the allowed root."""


def display_name(filename: str) -> str:
    """Reduce an untrusted filename to a plain display name (no directories, no control chars)."""
    name = re.split(r"[\\/]", filename)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if name in ("", ".", ".."):
        return "unnamed"
    return name[:MAX_DISPLAY_NAME_LENGTH]


def resolve_inside(root: Path, relative: str | Path) -> Path:
    """Resolve `relative` under `root`, refusing anything that escapes it."""
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise UnsafePathError(f"path escapes storage root: {relative!r}")
    return candidate


def _stored_path_for(content_hash: str) -> str:
    return f"{FILES_SUBDIR}/{content_hash[:2]}/{content_hash}"


def _write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_file(session: Session, data_dir: Path, data: bytes, filename: str) -> Document:
    """Store bytes under a path derived only from their SHA-256 hash and record a Document.

    Saving identical content twice returns the existing Document. The filename is kept as
    display metadata and never influences where the file is written.
    """
    content_hash = hashlib.sha256(data).hexdigest()
    stored_path = _stored_path_for(content_hash)

    target = resolve_inside(data_dir, stored_path)
    if not target.exists():
        _write_atomic(target, data)

    existing = session.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing is not None:
        return existing

    name = display_name(filename)
    document = Document(
        original_filename=name,
        content_hash=content_hash,
        stored_path=stored_path,
        size_bytes=len(data),
        media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
    )
    session.add(document)
    session.commit()
    return document


def read_file(data_dir: Path, document: Document) -> bytes:
    """Read a stored file, re-validating the path so a tampered record cannot escape data_dir."""
    return resolve_inside(data_dir, document.stored_path).read_bytes()
