from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import REPO_ROOT, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "APP_ENV",
        "LOG_LEVEL",
        "DATA_DIR",
        "MODELS_DIR",
        "ALLOWED_FOLDERS",
        "MAX_FILE_SIZE_MB",
        "CHUNK_SIZE_CHARS",
        "CHUNK_OVERLAP_CHARS",
        "EMBEDDING_MODEL",
        "RETRIEVAL_TOP_K",
        "OLLAMA_URL",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_THINK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.allowed_folders == []
    assert settings.max_file_size_mb == 5
    assert settings.chunk_size_chars == 1000
    assert settings.chunk_overlap_chars == 150
    assert settings.data_dir == (REPO_ROOT / "data").resolve()
    assert settings.models_dir == (REPO_ROOT / "models").resolve()


def test_allowed_folders_parsed_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_FOLDERS", "C:/notes, D:/docs ,")

    settings = Settings(_env_file=None)

    assert settings.allowed_folders == [Path("C:/notes"), Path("D:/docs")]


def test_empty_allowed_folders_means_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_FOLDERS", "")

    assert Settings(_env_file=None).allowed_folders == []


def test_max_file_size_mb_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "12")

    assert Settings(_env_file=None).max_file_size_mb == 12


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_max_file_size_mb_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunk_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE_CHARS", "500")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "50")

    settings = Settings(_env_file=None)

    assert (settings.chunk_size_chars, settings.chunk_overlap_chars) == (500, 50)


@pytest.mark.parametrize(
    ("size", "overlap"), [("500", "500"), ("500", "600"), ("50", "10"), ("500", "-1")]
)
def test_invalid_chunk_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, size: str, overlap: str
) -> None:
    monkeypatch.setenv("CHUNK_SIZE_CHARS", size)
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", overlap)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_default_embedding_model() -> None:
    assert Settings(_env_file=None).embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize(
    "name", ["", "all-MiniLM-L6-v2", "a/b/c", "../evil", "a/..", "a b/c", r"a/b\..\c", "/abs"]
)
def test_invalid_embedding_model_names_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", name)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_retrieval_top_k_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).retrieval_top_k == 5

    monkeypatch.setenv("RETRIEVAL_TOP_K", "12")
    assert Settings(_env_file=None).retrieval_top_k == 12


@pytest.mark.parametrize("value", ["0", "51", "many"])
def test_invalid_retrieval_top_k_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_url == "http://127.0.0.1:11434"
    assert settings.llm_model == "qwen3.5:4b"
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_think is False


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_ollama_url_on_this_machine_is_accepted(
    monkeypatch: pytest.MonkeyPatch, url: str, expected: str
) -> None:
    monkeypatch.setenv("OLLAMA_URL", url)

    assert Settings(_env_file=None).ollama_url == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "https://api.openai.com/v1",
        "http://192.168.1.20:11434",
        "http://user:pw@localhost:11434",
        "http://localhost:99999",
        "localhost:11434",
        "",
    ],
)
def test_remote_or_malformed_ollama_urls_are_rejected(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("OLLAMA_URL", url)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("name", ["", "has space", "../evil", "model:", ":tag", "a:b:c"])
def test_invalid_llm_model_names_are_rejected(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("LLM_MODEL", name)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", ["4", "901", "soon"])
def test_invalid_llm_timeouts_are_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
