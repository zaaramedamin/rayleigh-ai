from pathlib import Path

import pytest

from app.ai.embeddings.base import (
    EmbeddingRuntimeError,
    ModelNotAvailableError,
    is_model_downloaded,
    model_dir_for,
)
from app.ai.embeddings.sentence_transformer import SentenceTransformerProvider
from app.storage.files import UnsafePathError

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_model_folder_is_one_level_under_models_dir(tmp_path: Path) -> None:
    path = model_dir_for(tmp_path, MODEL)

    assert path == (tmp_path / "sentence-transformers__all-MiniLM-L6-v2").resolve()
    assert path.parent == tmp_path.resolve()


def test_model_folder_cannot_escape_models_dir(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        model_dir_for(tmp_path, "..")


def test_model_is_downloaded_only_when_marker_file_exists(tmp_path: Path) -> None:
    folder = model_dir_for(tmp_path, MODEL)
    assert not is_model_downloaded(tmp_path, MODEL)

    folder.mkdir(parents=True)
    assert not is_model_downloaded(tmp_path, MODEL)

    (folder / "modules.json").write_text("[]")
    assert is_model_downloaded(tmp_path, MODEL)


def test_provider_refuses_to_start_without_a_downloaded_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="download-model"):
        SentenceTransformerProvider(MODEL, tmp_path)


def _fake_downloaded_model(models_dir: Path) -> None:
    folder = model_dir_for(models_dir, MODEL)
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")


def test_a_library_that_cannot_load_is_reported_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_downloaded_model(tmp_path)
    blocked = ImportError("DLL load failed while importing _base: An Application Control policy")
    # Make `from sentence_transformers import ...` fail the way a blocked library file does.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise blocked
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(EmbeddingRuntimeError, match="Smart App Control"):
        SentenceTransformerProvider(MODEL, tmp_path)
