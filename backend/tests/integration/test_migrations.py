from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _config(db_path: Path) -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return config


def _tables(db_path: Path) -> list[str]:
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        return inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_upgrade_creates_tables_and_downgrade_removes_them(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    config = _config(db_path)

    command.upgrade(config, "head")
    assert {"documents", "chunks"} <= set(_tables(db_path))

    command.downgrade(config, "0001")
    assert "chunks" not in _tables(db_path)
    assert "documents" in _tables(db_path)

    command.downgrade(config, "base")
    assert "documents" not in _tables(db_path)


def test_migration_matches_models(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    config = _config(db_path)

    command.upgrade(config, "head")

    # Raises if the models have drifted from the migrations.
    command.check(config)
