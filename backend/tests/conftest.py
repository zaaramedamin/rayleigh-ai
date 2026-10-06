import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.storage.database import Base, create_db_engine, database_url

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """CLI tests configure logging globally; put pytest's handlers back afterwards."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture(autouse=True)
def _open_access() -> Iterator[None]:
    """API tests call routes directly; the sign-in itself is tested in test_access_api.py."""
    from app.api.access import require_access
    from app.main import app

    app.dependency_overrides[require_access] = lambda: None
    yield
    app.dependency_overrides.pop(require_access, None)


def _make_test_clients_local() -> None:
    """The API only answers to this computer's own address, so the tests' client says it is one.

    Done when this file is imported, because some test modules create their client at import time.
    """
    from starlette.testclient import TestClient

    original = TestClient.__init__

    def local_init(self: TestClient, app: object, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("base_url", "http://127.0.0.1")
        original(self, app, *args, **kwargs)  # type: ignore[arg-type]

    TestClient.__init__ = local_init  # type: ignore[method-assign]


_make_test_clients_local()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def session(data_dir: Path) -> Iterator[Session]:
    engine = create_db_engine(data_dir)
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()


@pytest.fixture
def make_settings(data_dir: Path, models_dir: Path) -> Callable[..., Settings]:
    """Settings isolated from the real .env: temporary data/models folders, empty allow-list."""

    def _make(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "data_dir": data_dir,
            "models_dir": models_dir,
            "allowed_folders": [],
            **overrides,
        }
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def migrated_data_dir(data_dir: Path) -> Path:
    """A data dir whose database has been created by the real Alembic migrations."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url(data_dir))
    command.upgrade(config, "head")
    return data_dir
