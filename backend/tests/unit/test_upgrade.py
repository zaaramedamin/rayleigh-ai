"""Slice 5.1: `migrate`, `serve`, and commands that wait for a busy search index."""

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from app import cli
from app.cli import main
from app.core.config import Settings
from app.knowledge.components import open_vector_store, vector_store_path
from app.operations import upgrade as upgrade_module
from app.operations.upgrade import (
    UpgradeError,
    current_revision,
    default_backup_dir,
    upgrade_database,
)
from app.security import keystore
from app.storage.database import DB_FILENAME
from app.storage.migrations import BACKEND_DIR, latest_revision
from app.storage.vector_store import QdrantVectorStore, VectorStoreBusyError, collection_name
from tests.fakes import HashingEmbedder

MakeSettings = Callable[..., Settings]


def old_library(data_dir: Path, revision: str = "0003") -> Path:
    """A database made by an older version of the program, holding one document."""
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / DB_FILENAME
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    command.upgrade(config, revision)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO documents (original_filename, content_hash, stored_path, size_bytes, "
        "media_type, created_at) VALUES ('keep.md', 'hash1', 'files/ha/hash1', 1, "
        "'text/markdown', '2026-10-05 00:00:00')"
    )
    connection.commit()
    connection.close()
    return db_path


def names_in(db_path: Path) -> list[str]:
    connection = sqlite3.connect(db_path)
    try:
        return [row[0] for row in connection.execute("SELECT original_filename FROM documents")]
    finally:
        connection.close()


# --- upgrading ------------------------------------------------------------------------------------


def test_a_missing_database_is_created_without_any_backup(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    result = upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    assert (result.from_revision, result.to_revision) == (None, latest_revision())
    assert result.upgraded and result.backup is None
    assert current_revision(data_dir / DB_FILENAME) == latest_revision()
    assert not (tmp_path / "backups").exists()


def test_an_older_database_is_copied_first_and_then_upgraded(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    db_path = old_library(data_dir)

    result = upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    assert (result.from_revision, result.to_revision) == ("0003", latest_revision())
    assert current_revision(db_path) == latest_revision()
    assert names_in(db_path) == ["keep.md"]  # nothing was lost
    assert result.backup is not None and result.backup.parent == tmp_path / "backups"
    assert result.backup.name.startswith("before-upgrade-0003-")
    assert current_revision(result.backup) == "0003"  # the copy is the old, restorable version
    assert names_in(result.backup) == ["keep.md"]


def test_a_database_that_is_up_to_date_is_left_alone(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    upgrade_database(make_settings(), backup_dir=tmp_path / "backups")
    before = (data_dir / DB_FILENAME).read_bytes()

    again = upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    assert not again.upgraded and again.backup is None
    assert (data_dir / DB_FILENAME).read_bytes() == before
    assert not (tmp_path / "backups").exists()


def test_a_database_from_a_newer_version_is_refused_untouched(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    db_path = old_library(data_dir, revision=latest_revision())
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()
    before = db_path.read_bytes()

    with pytest.raises(UpgradeError, match="newer version"):
        upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    assert db_path.read_bytes() == before
    assert not (tmp_path / "backups").exists()


def test_an_unfinished_encryption_must_be_resumed_first(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    old_library(data_dir)
    keystore.create_keys(data_dir, "correct horse battery staple", state="migrating")

    with pytest.raises(UpgradeError, match="encrypt-library"):
        upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    assert current_revision(data_dir / DB_FILENAME) == "0003"


def test_if_the_copy_cannot_be_made_nothing_is_changed(
    make_settings: MakeSettings, data_dir: Path, tmp_path: Path
) -> None:
    db_path = old_library(data_dir)
    not_a_folder = tmp_path / "file-in-the-way"
    not_a_folder.write_text("x")

    with pytest.raises(UpgradeError, match="copy of the database could not be saved"):
        upgrade_database(make_settings(), backup_dir=not_a_folder / "backups")

    assert current_revision(db_path) == "0003"


def test_a_failed_upgrade_says_where_the_copy_is(
    make_settings: MakeSettings,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_library(data_dir)

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("something inside Alembic with a private path C:/secret")

    monkeypatch.setattr(upgrade_module.command, "upgrade", broken)

    with pytest.raises(UpgradeError) as raised:
        upgrade_database(make_settings(), backup_dir=tmp_path / "backups")

    message = str(raised.value)
    assert "RuntimeError" in message and "A copy from before is at" in message
    assert "secret" not in message  # what went wrong inside is not repeated
    assert list((tmp_path / "backups").glob("before-upgrade-0003-*.db"))


def test_the_default_backup_folder_is_outside_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/someone/AppData/Local")

    assert default_backup_dir() == Path("C:/Users/someone/AppData/Local/Reyleight/backups")


# --- the commands ---------------------------------------------------------------------------------


def test_migrate_reports_what_it_did(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    old_library(data_dir)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert main(["migrate"], settings=make_settings()) == 0
    first = capsys.readouterr().out
    assert main(["migrate"], settings=make_settings()) == 0
    second = capsys.readouterr().out

    assert f"revision 0003 -> {latest_revision()}" in first
    assert "a copy from before the upgrade is saved at:" in first
    assert "the database is up to date" in second


def test_migrate_explains_a_refusal_instead_of_a_traceback(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, data_dir: Path
) -> None:
    db_path = old_library(data_dir, revision=latest_revision())
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()

    assert main(["migrate"], settings=make_settings()) == 1

    err = capsys.readouterr().err
    assert "error:" in err and "newer version" in err and "Traceback" not in err


class FakeUvicorn:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> FakeUvicorn:
    import uvicorn

    fake = FakeUvicorn()
    monkeypatch.setattr(uvicorn, "run", fake.run)
    return fake


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_serve_listens_on_this_computer_only_by_default(
    host: str,
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    uvicorn_calls: FakeUvicorn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert main(["serve", "--host", host, "--port", "8123"], settings=make_settings()) == 0

    (args, kwargs) = uvicorn_calls.calls[0]
    assert args == ("app.main:app",)
    assert kwargs == {"host": host, "port": 8123, "log_config": None}
    out = capsys.readouterr().out
    assert f"http://{host}:8123/" in out


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "example.com", ""])
def test_serve_refuses_an_address_other_computers_could_reach(
    host: str,
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    uvicorn_calls: FakeUvicorn,
) -> None:
    assert main(["serve", "--host", host], settings=make_settings()) == 1

    err = capsys.readouterr().err
    assert "refusing to listen" in err and "--unsafe-expose-to-network" in err
    assert uvicorn_calls.calls == []


def test_serve_can_be_forced_onto_the_network_with_a_loud_warning(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    uvicorn_calls: FakeUvicorn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    code = main(
        ["serve", "--host", "0.0.0.0", "--unsafe-expose-to-network"], settings=make_settings()
    )

    assert code == 0 and uvicorn_calls.calls
    assert "WARNING: listening on 0.0.0.0" in capsys.readouterr().err


def test_serve_upgrades_the_database_before_it_starts(
    make_settings: MakeSettings,
    data_dir: Path,
    uvicorn_calls: FakeUvicorn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    import uvicorn

    db_path = old_library(data_dir)
    seen: list[str | None] = []
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: seen.append(current_revision(db_path)))

    assert main(["serve"], settings=make_settings()) == 0

    assert seen == [latest_revision()]  # it was already up to date when the server started


def test_serve_can_skip_the_upgrade(
    make_settings: MakeSettings, data_dir: Path, uvicorn_calls: FakeUvicorn
) -> None:
    db_path = old_library(data_dir)

    assert main(["serve", "--no-migrate"], settings=make_settings()) == 0

    assert current_revision(db_path) == "0003" and uvicorn_calls.calls


def test_serve_does_not_start_when_the_upgrade_is_refused(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    data_dir: Path,
    uvicorn_calls: FakeUvicorn,
) -> None:
    db_path = old_library(data_dir, revision=latest_revision())
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()

    assert main(["serve"], settings=make_settings()) == 1

    assert uvicorn_calls.calls == [] and "newer version" in capsys.readouterr().err


# --- waiting for the search index -----------------------------------------------------------------


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


def hold_the_index(
    settings: Settings, embedder: HashingEmbedder, seconds: float
) -> tuple[threading.Thread, threading.Event]:
    """Open the search index in another thread for a while, as the server does during a request."""
    opened = threading.Event()

    def hold() -> None:
        store = QdrantVectorStore.open_local(
            vector_store_path(settings), collection_name(embedder.model_name), embedder.dimension
        )
        opened.set()
        time.sleep(seconds)
        store.close()

    thread = threading.Thread(target=hold)
    thread.start()
    assert opened.wait(10)
    return thread, opened


def test_a_busy_index_is_waited_for_and_then_opened(
    make_settings: MakeSettings, embedder: HashingEmbedder
) -> None:
    settings = make_settings()
    thread, _ = hold_the_index(settings, embedder, 1.0)
    told: list[int] = []

    started = time.monotonic()
    with open_vector_store(
        settings, embedder, wait_seconds=10, on_wait=lambda: told.append(1)
    ) as store:
        waited = time.monotonic() - started
        assert store.count() == 0
    thread.join()

    assert told == [1]  # told once, however long it waited
    assert 0.5 <= waited < 8


def test_a_busy_index_that_stays_busy_gives_up_with_the_usual_error(
    make_settings: MakeSettings, embedder: HashingEmbedder
) -> None:
    settings = make_settings()
    thread, _ = hold_the_index(settings, embedder, 2.0)

    with pytest.raises(VectorStoreBusyError, match="in use by another process"):
        with open_vector_store(settings, embedder, wait_seconds=0.5):
            pass
    thread.join()


def test_without_a_wait_the_busy_error_is_immediate(
    make_settings: MakeSettings, embedder: HashingEmbedder
) -> None:
    settings = make_settings()
    thread, _ = hold_the_index(settings, embedder, 1.0)

    started = time.monotonic()
    with pytest.raises(VectorStoreBusyError):
        with open_vector_store(settings, embedder):
            pass
    elapsed = time.monotonic() - started
    thread.join()

    assert elapsed < 0.5


def test_search_on_the_command_line_waits_for_the_server_instead_of_failing(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    embedder: HashingEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("a plain note about oats")
    folder = Path(make_settings().models_dir) / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")
    monkeypatch.setattr(cli, "load_embedder", lambda *_a: embedder)
    settings = make_settings(allowed_folders=[notes])
    assert main(["ingest"], settings=settings) == 0
    capsys.readouterr()
    thread, _ = hold_the_index(settings, embedder, 1.0)

    code = main(["search", "oats"], settings=settings)
    thread.join()

    captured = capsys.readouterr()
    assert code == 0 and "a.txt" in captured.out
    assert "the search index is in use (by the server?); waiting for it" in captured.err
