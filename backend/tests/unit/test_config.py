from pathlib import Path

import pytest

from app.core.config import REPO_ROOT, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("APP_ENV", "LOG_LEVEL", "DATA_DIR", "MODELS_DIR", "ALLOWED_FOLDERS"):
        monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.allowed_folders == []
    assert settings.data_dir == (REPO_ROOT / "data").resolve()
    assert settings.models_dir == (REPO_ROOT / "models").resolve()


def test_allowed_folders_parsed_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_FOLDERS", "C:/notes, D:/docs ,")

    settings = Settings(_env_file=None)

    assert settings.allowed_folders == [Path("C:/notes"), Path("D:/docs")]


def test_empty_allowed_folders_means_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_FOLDERS", "")

    assert Settings(_env_file=None).allowed_folders == []
