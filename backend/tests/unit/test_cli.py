from collections.abc import Callable
from pathlib import Path

import pytest

import app.ai.embeddings.download as download_module
from app import cli
from app.ai.llm.base import LLMModelNotFoundError, LLMUnavailableError
from app.cli import main
from app.core.config import Settings
from app.storage.vector_store import QdrantVectorStore, collection_name
from tests.fakes import FakeLLM, HashingEmbedder

MakeSettings = Callable[..., Settings]


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """No CLI test may talk to a real Ollama server."""
    llm = FakeLLM(model_name="qwen3.5:4b", installed=["qwen3.5:4b"])
    monkeypatch.setattr(cli, "create_llm", lambda *_args, **_kwargs: llm)
    return llm


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> HashingEmbedder:
    """Pretend the default model is downloaded, and embed with a fast deterministic fake."""
    embedder = HashingEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    folder = models_dir / "sentence-transformers__all-MiniLM-L6-v2"
    folder.mkdir(parents=True)
    (folder / "modules.json").write_text("[]")
    monkeypatch.setattr(cli, "load_embedder", lambda *_args: embedder)
    return embedder


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


def test_ingest_without_a_model_explains_the_next_step(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
) -> None:
    main(["ingest"], settings=make_settings(allowed_folders=[notes]))

    assert "2 document(s) not searchable yet" in capsys.readouterr().out


def test_ingest_indexes_new_documents_when_the_model_is_available(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    assert main(["ingest"], settings=make_settings(allowed_folders=[notes])) == 0

    assert "2 document(s) indexed, 2 chunks embedded" in capsys.readouterr().out


def test_index_command_only_embeds_what_is_pending_unless_rebuilding(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["index"], settings=settings) == 0
    assert "indexed 0 document(s)" in capsys.readouterr().out

    assert main(["index", "--rebuild"], settings=settings) == 0
    assert "indexed 2 document(s), 2 chunks embedded" in capsys.readouterr().out


def test_status_reports_documents_chunks_and_vectors(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["status"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "downloaded:    yes" in out
    assert "documents:       2" in out
    assert "chunks:          2" in out
    assert "searchable:      2 of 2 documents" in out
    assert "vectors:         2" in out


def test_index_without_a_model_is_an_error(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["index"], settings=make_settings()) == 1

    assert "download-model" in capsys.readouterr().err


def test_index_reports_a_busy_vector_store(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_model: HashingEmbedder,
) -> None:
    collection = collection_name(fake_model.model_name)
    with QdrantVectorStore.open_local(
        migrated_data_dir / "qdrant", collection, fake_model.dimension
    ):
        assert main(["index"], settings=make_settings()) == 1

    assert "in use by another process" in capsys.readouterr().err


def test_search_prints_ranked_results_with_sources(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["search", "plain", "note"], settings=settings) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. score ")
    assert "b.txt" in lines[0]
    assert "lines 1-1" in lines[0]
    assert lines[1].strip() == "plain note"


def test_search_filters_by_file_type(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    main(["search", "note", "--type", "md"], settings=settings)

    out = capsys.readouterr().out
    assert "a.md" in out
    assert "b.txt" not in out


def test_search_on_an_empty_index_says_so(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_model: HashingEmbedder,
) -> None:
    assert main(["search", "anything"], settings=make_settings()) == 0

    assert "no matching notes found" in capsys.readouterr().out


def test_search_rejects_an_invalid_top_k(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_model: HashingEmbedder,
) -> None:
    assert main(["search", "oats", "--top-k", "0"], settings=make_settings()) == 1

    assert "top_k must be between" in capsys.readouterr().err


def test_check_llm_prints_the_reply(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = "ready"

    assert main(["check-llm"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "reply: ready" in out
    assert "took " in out


def test_check_llm_reports_a_stopped_server(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, fake_llm: FakeLLM
) -> None:
    fake_llm.error = LLMUnavailableError("Ollama is not reachable; run `ollama serve`")

    assert main(["check-llm"], settings=make_settings()) == 1

    assert "ollama serve" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("installed", "error", "expected"),
    [
        (["qwen3.5:4b"], None, "running, model installed"),
        (["other:1b"], None, "running, but the model is not installed"),
        ([], LLMUnavailableError("down"), "not running at http://127.0.0.1:11434"),
        ([], LLMModelNotFoundError("odd"), "problem: odd"),
    ],
)
def test_status_reports_the_llm_state(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    fake_llm: FakeLLM,
    installed: list[str],
    error: Exception | None,
    expected: str,
) -> None:
    fake_llm.installed = installed
    fake_llm.error = error

    assert main(["status"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "llm model:       qwen3.5:4b" in out
    assert expected in out


def test_invalid_settings_are_reported_in_one_readable_block(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(ollama_url="http://example.com"))

    assert main(["types"]) == 1

    err = capsys.readouterr().err
    assert "invalid settings" in err
    assert "OLLAMA_URL" in err
    assert "Traceback" not in err
