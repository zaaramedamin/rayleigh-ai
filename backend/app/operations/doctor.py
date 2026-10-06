"""`python -m app doctor`: check that the whole setup is healthy, and say how to fix what is not.

Read-only by default. Nothing is created, deleted or changed unless `repair=True`, and even then
the only repair is to mark documents for re-indexing, which is always safe. Orphaned files and
anything else that would delete data are reported, never removed.
"""

import hashlib
import shutil
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from app.ai.embeddings.base import is_model_downloaded
from app.ai.llm.base import LLMError, LLMUnavailableError
from app.core.config import Settings
from app.knowledge.components import create_llm, vector_store_path
from app.knowledge.indexing.service import count_stale_vectors, library_counts
from app.security import keystore
from app.security.errors import (
    DecryptionError,
    KeystoreError,
    LibraryLockedError,
    MigrationIncompleteError,
    SecurityError,
    WrongPassphraseError,
)
from app.security.migrate import plaintext_values
from app.security.sqlalchemy_types import vault_of_session
from app.storage.database import DB_FILENAME, create_db_engine
from app.storage.files import (
    FILES_SUBDIR,
    UnsafePathError,
    file_context,
    looks_encrypted,
    resolve_inside,
)
from app.storage.migrations import database_is_up_to_date
from app.storage.models import Chunk, Document
from app.storage.vector_store import VectorStoreError, collection_name, count_local_vectors

Status = Literal["ok", "warn", "fail"]
_MIN_FREE_BYTES = 1024**3  # 1 GB


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    message: str
    fix: str | None = None


def smart_app_control_state() -> str | None:
    """ "on", "off" or "evaluation" on Windows (read-only registry lookup), else None."""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except OSError:
        return None
    return {0: "off", 1: "on", 2: "evaluation"}.get(int(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0


# --- individual checks ------------------------------------------------------------------------


def _check_settings(settings: Settings) -> list[Check]:
    checks = []
    missing = [f for f in settings.allowed_folders if not f.is_dir()]
    if not settings.allowed_folders:
        checks.append(
            Check(
                "allowed folders",
                "warn",
                "ALLOWED_FOLDERS is empty: nothing will be ingested",
                "set it in .env",
            )
        )
    elif missing:
        checks.append(
            Check(
                "allowed folders",
                "warn",
                f"{len(missing)} of {len(settings.allowed_folders)} allowed folder(s) do not exist",
                "check the paths in ALLOWED_FOLDERS in .env",
            )
        )
    else:
        checks.append(
            Check(
                "allowed folders", "ok", f"{len(settings.allowed_folders)} folder(s), all present"
            )
        )
    data = settings.data_dir.resolve()
    inside = [
        f
        for f in settings.allowed_folders
        if f.resolve().is_relative_to(data) or data.is_relative_to(f.resolve())
    ]
    if inside:
        checks.append(
            Check(
                "data folder",
                "fail",
                "an allowed folder overlaps the data folder, so Reyleight would index its "
                "own storage",
                "choose different folders in ALLOWED_FOLDERS or DATA_DIR",
            )
        )
    return checks


def _check_disk(settings: Settings) -> Check:
    probe = settings.data_dir if settings.data_dir.exists() else settings.data_dir.parent
    free = shutil.disk_usage(probe).free
    used = _folder_size(settings.data_dir)
    needed = max(_MIN_FREE_BYTES, 2 * used)
    free_gb = free / 1024**3
    if free < needed:
        return Check(
            "disk space",
            "warn",
            f"{free_gb:.1f} GB free, and the library uses {used / 1024**2:.0f} MB",
            "free some space before ingesting or backing up",
        )
    return Check(
        "disk space", "ok", f"{free_gb:.1f} GB free; the library uses {used / 1024**2:.0f} MB"
    )


def _check_model(settings: Settings) -> Check:
    if is_model_downloaded(settings.models_dir, settings.embedding_model):
        return Check("embedding model", "ok", f"{settings.embedding_model} is downloaded")
    return Check(
        "embedding model",
        "warn",
        f"{settings.embedding_model} is not downloaded",
        "python -m app download-model",
    )


def _check_llm(settings: Settings) -> Check:
    try:
        installed = create_llm(settings, timeout_seconds=3).list_models()
    except LLMUnavailableError:
        return Check(
            "local LLM",
            "warn",
            f"Ollama is not running at {settings.ollama_url}",
            "open the Ollama app or run `ollama serve`",
        )
    except LLMError as exc:
        return Check("local LLM", "warn", f"Ollama gave an unexpected answer: {exc}")
    wanted = settings.llm_model if ":" in settings.llm_model else f"{settings.llm_model}:latest"
    if wanted in installed:
        return Check("local LLM", "ok", f"Ollama is running and {settings.llm_model} is installed")
    return Check(
        "local LLM",
        "warn",
        f"Ollama is running but {settings.llm_model} is not installed",
        f"ollama pull {settings.llm_model}",
    )


def _check_windows_protection() -> Check | None:
    state = smart_app_control_state()
    if state is None:
        return None
    if state == "off":
        return Check("Windows Smart App Control", "ok", "off")
    return Check(
        "Windows Smart App Control",
        "ok",
        f"{state}: it can block newly installed compiled libraries "
        "(Reyleight already works around scikit-learn; see the README troubleshooting)",
    )


def _open_library(settings: Settings, passphrase: str | None) -> tuple[Engine | None, list[Check]]:
    """Open the library (unlocking it if encrypted). Returns the engine, or None with the reason."""
    db_path = settings.data_dir / DB_FILENAME
    if not db_path.is_file():
        return None, [
            Check("database", "fail", f"no database at {db_path}", "python -m app migrate")
        ]
    try:
        engine = create_db_engine(settings.data_dir, passphrase=passphrase)
    except LibraryLockedError:
        return None, [
            Check(
                "encryption",
                "fail",
                "the library is encrypted and locked: Windows could not unlock it for this account",
                "run any `python -m app` command in a terminal and type the recovery passphrase "
                "when asked; it is then remembered for this Windows account",
            )
        ]
    except WrongPassphraseError:
        return None, [Check("encryption", "fail", "that recovery passphrase is not correct")]
    except MigrationIncompleteError:
        return None, [
            Check(
                "encryption",
                "fail",
                "encrypting the library was started but not finished",
                "python -m app encrypt-library",
            )
        ]
    except SecurityError as exc:
        return None, [Check("encryption", "fail", str(exc))]
    if not database_is_up_to_date(engine):
        engine.dispose()
        return None, [
            Check("database", "fail", "the database is out of date", "python -m app migrate")
        ]
    return engine, [Check("database", "ok", "present and at the latest version")]


def _check_encryption(settings: Settings, engine: Engine, *, deep: bool) -> list[Check]:
    try:
        state = keystore.library_state(settings.data_dir)
    except KeystoreError as exc:
        return [Check("encryption", "fail", str(exc), "restore security.json from a backup")]
    if state == "plaintext":
        return [
            Check(
                "encryption",
                "warn",
                "the library is NOT encrypted: notes, headings and file names are readable on disk",
                "python -m app encrypt-library",
            )
        ]
    checks: list[Check] = []
    leftovers = _plaintext_value_count(settings.data_dir / DB_FILENAME)
    if leftovers:
        checks.append(
            Check(
                "encryption",
                "fail",
                f"{leftovers} value(s) in the database are not encrypted",
                "python -m app encrypt-library",
            )
        )
    else:
        windows = keystore.windows_unlock_works(settings.data_dir)
        checks.append(
            Check(
                "encryption",
                "ok" if windows else "warn",
                "encrypted (AES-256-GCM); "
                + (
                    "Windows unlocks it automatically for this account"
                    if windows
                    else "Windows cannot unlock it for this account; the recovery "
                    "passphrase is needed"
                ),
                None if windows else "run any command and type the recovery passphrase once",
            )
        )
    del engine, deep
    return checks


def _plaintext_value_count(db_path: Path) -> int:
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return plaintext_values(connection)
    finally:
        connection.close()


def _check_files(session: Session, settings: Settings, *, quick: bool) -> list[Check]:
    rows = session.execute(select(Document.id, Document.content_hash, Document.stored_path)).all()
    vault = vault_of_session(session)
    missing = corrupt = unsafe = unencrypted = 0
    for _doc_id, content_hash, stored_path in rows:
        try:
            path = resolve_inside(settings.data_dir, stored_path)
        except UnsafePathError:
            unsafe += 1
            continue
        if not path.is_file():
            missing += 1
        elif not quick:
            raw = path.read_bytes()
            if looks_encrypted(raw, content_hash):
                try:
                    plain = vault.decrypt_bytes(raw, file_context(content_hash)) if vault else b""
                except DecryptionError:
                    corrupt += 1
                    continue
                if vault is None or hashlib.sha256(plain).hexdigest() != content_hash:
                    corrupt += 1
            else:
                if vault is not None:
                    unencrypted += 1
                if hashlib.sha256(raw).hexdigest() != content_hash:
                    corrupt += 1

    checks = []
    problems = []
    if missing:
        problems.append(f"{missing} stored file(s) are missing")
    if corrupt:
        problems.append(f"{corrupt} stored file(s) do not match their checksum")
    if unsafe:
        problems.append(f"{unsafe} document(s) point outside the data folder")
    if unencrypted:
        problems.append(f"{unencrypted} stored file(s) are not encrypted in an encrypted library")
    if problems:
        checks.append(
            Check(
                "stored files",
                "fail",
                "; ".join(problems),
                "run `python -m app ingest` to re-create them from your original files "
                "(searching still works without them), or restore a backup",
            )
        )
    else:
        how = "present" if quick else "present and checksums verified"
        checks.append(Check("stored files", "ok", f"{len(rows)} file(s) {how}"))

    known = {content_hash for _, content_hash, _ in rows}
    files_root = settings.data_dir / FILES_SUBDIR
    on_disk = [p for p in files_root.rglob("*") if p.is_file()] if files_root.is_dir() else []
    orphans = [p for p in on_disk if p.name not in known]
    if orphans:
        checks.append(
            Check(
                "orphaned files",
                "warn",
                f"{len(orphans)} stored file(s) belong to no document",
                "harmless; they are never deleted automatically",
            )
        )
    return checks


def _check_chunks_and_index(session: Session, settings: Settings, *, repair: bool) -> list[Check]:
    checks = []
    empty = session.scalar(
        select(func.count())
        .select_from(Document)
        .where(~select(Chunk.id).where(Chunk.document_id == Document.id).exists())
    )
    if empty:
        checks.append(
            Check(
                "chunks",
                "warn",
                f"{empty} document(s) have no chunks",
                "python -m app ingest (or `rechunk`) will create them",
            )
        )
    else:
        checks.append(Check("chunks", "ok", "every document has chunks"))

    model = settings.embedding_model
    counts = library_counts(session, model)
    documents, pending = counts.active, counts.pending
    stale = count_stale_vectors(session)
    expected = (
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.indexed_model == model)
        )
        or 0
    )
    try:
        vectors = count_local_vectors(vector_store_path(settings), collection_name(model))
    except VectorStoreError as exc:
        checks.append(Check("search index", "warn", f"could not be checked: {exc}"))
        return checks

    if vectors != expected:
        message = (
            f"{vectors} vectors for {expected} indexed chunks: the search index and the "
            "database disagree"
        )
        if repair:
            session.execute(update(Document).values(indexed_model=None))
            session.execute(update(Chunk).values(indexed_model=None))
            session.commit()
            checks.append(
                Check(
                    "search index",
                    "warn",
                    message + ". Marked every document for re-indexing.",
                    "python -m app index --rebuild",
                )
            )
        else:
            checks.append(
                Check(
                    "search index",
                    "warn",
                    message,
                    "python -m app index --rebuild (or `doctor --fix`)",
                )
            )
    elif pending:
        checks.append(
            Check(
                "search index",
                "warn",
                f"{pending} of {documents} document(s) are not searchable yet",
                "python -m app index",
            )
        )
    elif stale:
        checks.append(
            Check(
                "search index",
                "warn",
                f"{stale} replaced or missing document(s) still have vectors in the index; "
                "they are never returned",
                "python -m app index",
            )
        )
    else:
        checks.append(Check("search index", "ok", f"{vectors} vectors for {documents} document(s)"))
    return checks


def _check_library(session: Session) -> list[Check]:
    """What the library knows about edited and deleted files."""
    counts = library_counts(session, "")
    if counts.missing:
        return [
            Check(
                "library",
                "warn",
                f"{counts.missing} document(s) were not found in your folders in the last two "
                "updates, so they are no longer searched",
                "python -m app prune --dry-run shows them; they come back if the file returns",
            )
        ]
    versions = (
        f", {counts.superseded} older version(s) kept as history" if counts.superseded else ""
    )
    return [Check("library", "ok", f"{counts.active} current document(s){versions}")]


# --- running everything -----------------------------------------------------------------------


def _guarded(name: str, check: Callable[[], list[Check]]) -> list[Check]:
    """A check that crashes becomes a failed check instead of stopping the doctor."""
    try:
        return check()
    except Exception as exc:  # the doctor must always finish
        return [Check(name, "fail", f"the check itself failed: {type(exc).__name__}")]


def run_doctor(
    settings: Settings,
    *,
    repair: bool = False,
    quick: bool = False,
    passphrase: str | None = None,
) -> list[Check]:
    results: list[Check] = []
    engine, opened = _open_library(settings, passphrase)
    results += opened

    if engine is not None:
        try:
            results += _guarded(
                "encryption", lambda: _check_encryption(settings, engine, deep=not quick)
            )
            with Session(engine) as session:
                results += _guarded(
                    "stored files", lambda: _check_files(session, settings, quick=quick)
                )
                results += _guarded("library", lambda: _check_library(session))
                results += _guarded(
                    "search index",
                    lambda: _check_chunks_and_index(session, settings, repair=repair),
                )
        finally:
            engine.dispose()

    results += _guarded("settings", lambda: _check_settings(settings))
    results += _guarded("disk space", lambda: [_check_disk(settings)])
    results += _guarded("embedding model", lambda: [_check_model(settings)])
    results += _guarded("local LLM", lambda: [_check_llm(settings)])
    protection = _guarded(
        "Windows Smart App Control", lambda: [c for c in [_check_windows_protection()] if c]
    )
    results += protection
    return results


def format_checks(checks: list[Check]) -> str:
    labels = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
    width = max(len(c.name) for c in checks)
    lines = []
    for check in checks:
        lines.append(f" {labels[check.status]}  {check.name:<{width}}  {check.message}")
        if check.fix and check.status != "ok":
            lines.append(f"       {'':<{width}}  fix: {check.fix}")
    counts = {s: sum(c.status == s for c in checks) for s in ("ok", "warn", "fail")}
    lines.append("")
    lines.append(f"{counts['ok']} ok, {counts['warn']} warning(s), {counts['fail']} failure(s)")
    return "\n".join(lines)
