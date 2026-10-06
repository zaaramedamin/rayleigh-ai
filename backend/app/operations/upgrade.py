"""Bring the library's database up to the version this program needs, safely.

New versions of the program sometimes add a table or a column. Until the database has them, every
request fails with "database out of date". `upgrade_database` does what `alembic upgrade head`
does, with two differences: it saves a copy of the database first (a restorable file in the backups
folder), and it refuses to touch a database it cannot understand.

It is used by `python -m app migrate` and, unless told not to, by `python -m app serve`. It runs
Alembic through Python, so it works even when Windows blocks the `alembic.exe` launcher.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import Settings
from app.security import keystore
from app.storage.database import DB_FILENAME, database_url
from app.storage.migrations import BACKEND_DIR, latest_revision, schema_revision_status

logger = logging.getLogger(__name__)


class UpgradeError(RuntimeError):
    """The database could not be upgraded. The message says why and what to do."""


@dataclass(frozen=True)
class UpgradeResult:
    from_revision: str | None  # None: there was no database yet
    to_revision: str
    backup: Path | None  # the copy made before upgrading, if there was something to copy

    @property
    def upgraded(self) -> bool:
        return self.from_revision != self.to_revision


def default_backup_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "Reyleight" / "backups"


def current_revision(db_path: Path) -> str | None:
    """The schema revision stored in the database file, or None if it has none (or no file)."""
    if not db_path.is_file():
        return None
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except sqlite3.OperationalError:
        return None  # a file with no version table: nothing of ours is in it yet
    finally:
        connection.close()
    return str(row[0]) if row else None


def _copy_database(db_path: Path, destination: Path) -> None:
    """A consistent copy made with SQLite's own backup, safe while another program has it open."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def upgrade_database(settings: Settings, *, backup_dir: Path | None = None) -> UpgradeResult:
    """Upgrade the library's database to the latest revision. Does nothing if it already is."""
    head = latest_revision()
    db_path = settings.data_dir / DB_FILENAME
    current = current_revision(db_path)
    if current == head:
        return UpgradeResult(current, head, None)

    if keystore.library_state(settings.data_dir) == "migrating":
        raise UpgradeError(
            "encrypting this library was started but not finished; run "
            "`python -m app encrypt-library` to resume before upgrading it"
        )
    if current is not None and schema_revision_status(current) == "unknown":
        raise UpgradeError(
            f"this database is at revision {current}, which this version of the program does not "
            "know: it was made by a newer version. Use that version, or restore a backup."
        )

    backup: Path | None = None
    if current is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = (backup_dir or default_backup_dir()) / f"before-upgrade-{current}-{stamp}.db"
        try:
            _copy_database(db_path, backup)
        except (sqlite3.Error, OSError) as exc:
            # Nothing has been changed yet, so stopping here is always safe.
            raise UpgradeError(f"a copy of the database could not be saved first: {exc}") from exc

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url(settings.data_dir))
    try:
        command.upgrade(config, "head")
    except Exception as exc:  # Alembic raises many kinds of errors; none should be a traceback
        hint = f" A copy from before is at {backup}." if backup else ""
        raise UpgradeError(f"the upgrade failed ({type(exc).__name__}).{hint}") from exc

    reached = current_revision(db_path)
    if reached != head:
        raise UpgradeError(f"the upgrade stopped at revision {reached}, not {head}")
    logger.info("database upgraded from=%s to=%s", current, head)
    return UpgradeResult(current, head, backup)
