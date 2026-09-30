from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.storage.database import Base, create_db_engine


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def session(data_dir: Path) -> Iterator[Session]:
    engine = create_db_engine(data_dir)
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()
