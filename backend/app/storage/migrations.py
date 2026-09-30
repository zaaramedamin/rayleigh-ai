from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _script_directory() -> ScriptDirectory:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return ScriptDirectory.from_config(config)


def database_is_up_to_date(engine: Engine) -> bool:
    """True if the database has been migrated to the latest Alembic revision."""
    with engine.connect() as connection:
        current = set(MigrationContext.configure(connection).get_current_heads())
    return current == set(_script_directory().get_heads())
