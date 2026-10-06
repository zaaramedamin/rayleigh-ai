from pathlib import Path
from typing import Literal

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import Engine

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _script_directory() -> ScriptDirectory:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return ScriptDirectory.from_config(config)


def latest_revision() -> str:
    """The newest schema revision this program knows."""
    return _script_directory().get_current_head() or ""


def database_is_up_to_date(engine: Engine) -> bool:
    """True if the database has been migrated to the latest Alembic revision."""
    with engine.connect() as connection:
        current = set(MigrationContext.configure(connection).get_current_heads())
    return current == set(_script_directory().get_heads())


def schema_revision_status(revision: str) -> Literal["current", "older", "unknown"]:
    """Compare a database's schema revision with this program's latest one.

    "unknown" means this program has no migration with that id, typically a database made by a
    newer version, which must not be opened with this one.
    """
    script = _script_directory()
    if revision in script.get_heads():
        return "current"
    try:
        known = script.get_revision(revision) is not None
    except CommandError:  # Alembic's error for "Can't locate revision"
        return "unknown"
    return "older" if known else "unknown"
