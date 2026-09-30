from collections.abc import Callable
from pathlib import Path

import pytest

import app.ai.embeddings.download as download_module
from app.cli import main
from app.core.config import Settings

MakeSettings = Callable[..., Settings]


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("# T\n\nbody\n")
    (folder / "b.txt").write_text("plain note")
    return folder


def test_types_lists_supported_types(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings
) -> None:
    assert main(["types"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "Supported file types" in out
    assert ".md" in out
    assert ".json" in out


def test_ingest_with_empty_allow_list_touches_nothing(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, data_dir: Path
) -> None:
    assert main(["ingest"], settings=make_settings()) == 0

    assert "ALLOWED_FOLDERS is empty" in capsys.readouterr().out
    assert not data_dir.exists()


def test_ingest_refuses_a_database_that_is_not_migrated(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, notes: Path
) -> None:
    assert main(["ingest"], settings=make_settings(allowed_folders=[notes])) == 1

    assert "alembic upgrade head" in capsys.readouterr().err


def test_ingest_end_to_end(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
) -> None:
    assert main(["ingest"], settings=make_settings(allowed_folders=[notes])) == 0

    out = capsys.readouterr().out
    assert "added:                     2" in out
    assert "chunks created:            2" in out


def test_rechunk_command(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["rechunk"], settings=settings) == 0

    assert "rechunked all stored documents: 2 chunks" in capsys.readouterr().out


def test_download_model_does_nothing_when_already_present(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, models_dir: Path
) -> None:
    target = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    target.mkdir(parents=True)
    (target / "modules.json").write_text("[]")

    assert main(["download-model"], settings=make_settings()) == 0

    assert "already downloaded" in capsys.readouterr().out


def test_download_failure_is_reported_not_raised(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> Path:
        raise OSError("network unreachable")

    monkeypatch.setattr(download_module, "download_model", fail)

    assert main(["download-model"], settings=make_settings()) == 1

    assert "download failed: OSError: network unreachable" in capsys.readouterr().err


def test_unknown_command_exits_with_usage_error(make_settings: MakeSettings) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"], settings=make_settings())

    assert exc.value.code == 2
