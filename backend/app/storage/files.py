import hashlib
import mimetypes
import os
import re
import tempfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, object_session

from app.security.errors import LibraryLockedError
from app.security.sqlalchemy_types import vault_of_session
from app.security.vault import Vault
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


def write_file_atomic(target: Path, data: bytes) -> None:
    """Write `data` to `target` through a temporary file, so a crash never leaves a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def store_file(
    session: Session, data_dir: Path, data: bytes, filename: str
) -> tuple[Document, bool]:
    """Store bytes under a path derived only from their SHA-256 hash and record a Document.

    Returns (document, created). Saving identical content twice returns the existing Document
    with created=False. The filename is kept as display metadata and never influences where
    the file is written.
    """
    content_hash = hashlib.sha256(data).hexdigest()
    stored_path = _stored_path_for(content_hash)

    target = resolve_inside(data_dir, stored_path)
    if not target.exists():
        vault = vault_of_session(session)
        # In an encrypted library the stored copy is encrypted too, and bound to its file name.
        write_file_atomic(
            target, data if vault is None else vault.encrypt_bytes(data, file_context(content_hash))
        )

    existing = session.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing is not None:
        return existing, False

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
    return document, True


def save_file(session: Session, data_dir: Path, data: bytes, filename: str) -> Document:
    """Like store_file, for callers that don't care whether the document was new."""
    return store_file(session, data_dir, data, filename)[0]


def file_context(content_hash: str) -> bytes:
    """What an encrypted stored file is bound to, so it cannot be swapped for another file."""
    return f"file:{content_hash}".encode("ascii")


def looks_encrypted(raw: bytes, content_hash: str) -> bool:
    """True for an encrypted stored file. A plaintext file that merely starts with the same
    marker bytes is told apart by its hash, which is also its file name."""
    return Vault.is_encrypted_blob(raw) and hashlib.sha256(raw).hexdigest() != content_hash


def read_file(data_dir: Path, document: Document) -> bytes:
    """Read a stored file (decrypting it if the library is encrypted).

    The path is re-validated, so a tampered record cannot escape data_dir.
    """
    raw = resolve_inside(data_dir, document.stored_path).read_bytes()
    if not looks_encrypted(raw, document.content_hash):
        return raw
    session = object_session(document)
    vault = vault_of_session(session) if session is not None else None
    if vault is None:
        raise LibraryLockedError("this stored file is encrypted and no key is available to read it")
    return vault.decrypt_bytes(raw, file_context(document.content_hash))
