from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.deps import get_embedder
from app.core.config import Settings, get_settings
from app.main import app
from app.security import keystore
from app.security.migrate import encrypt_library
from tests.fakes import HashingEmbedder

PASSPHRASE = "correct horse battery staple"


@pytest.fixture
def client() -> Iterator[TestClient]:
    deps._engine.cache_clear()
    yield TestClient(app)
    app.dependency_overrides.clear()
    deps._engine.cache_clear()


def _locked_library(data_dir: Path) -> None:
    encrypt_library(data_dir, PASSPHRASE, backup_to=None)
    keyfile = keystore.read_keyfile(data_dir)
    assert keyfile is not None
    keystore._write_keyfile(data_dir, keystore._replace(keyfile, wrapped_by_windows=None))


def test_a_locked_library_gives_a_clear_503_not_a_crash(
    client: TestClient, migrated_data_dir: Path, make_settings: Callable[..., Settings]
) -> None:
    _locked_library(migrated_data_dir)
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()

    response = client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "encrypted and locked" in detail
    assert "recovery passphrase" in detail


def test_an_unfinished_encryption_is_reported_by_the_api(
    client: TestClient, migrated_data_dir: Path, make_settings: Callable[..., Settings]
) -> None:
    keystore.create_keys(migrated_data_dir, PASSPHRASE, state="migrating")
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()

    response = client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 503
    assert "encrypt-library" in response.json()["detail"]


def test_an_unlocked_encrypted_library_serves_requests_normally(
    client: TestClient, migrated_data_dir: Path, make_settings: Callable[..., Settings]
) -> None:
    encrypt_library(migrated_data_dir, PASSPHRASE, backup_to=None)
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()
    # the vector store is opened per request under data_dir/qdrant; an empty index is fine here

    response = client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json() == {"results": []}
