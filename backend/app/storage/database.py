from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase

DB_FILENAME = "reyleight.db"


class Base(DeclarativeBase):
    pass


def database_url(data_dir: Path) -> str:
    return f"sqlite:///{(data_dir / DB_FILENAME).as_posix()}"


def create_db_engine(data_dir: Path) -> Engine:
    """Create a SQLite engine whose database file lives inside data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url(data_dir))

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        # SQLite ignores foreign keys (and ON DELETE CASCADE) unless this is enabled per connection.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine
