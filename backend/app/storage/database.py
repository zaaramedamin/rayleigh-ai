from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase

DB_FILENAME = "reyleight.db"


class Base(DeclarativeBase):
    pass


def database_url(data_dir: Path) -> str:
    return f"sqlite:///{(data_dir / DB_FILENAME).as_posix()}"


def create_db_engine(data_dir: Path) -> Engine:
    """Create a SQLite engine whose database file lives inside data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url(data_dir))
