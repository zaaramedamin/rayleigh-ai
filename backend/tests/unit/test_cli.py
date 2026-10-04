import json
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


# --- ask ---------------------------------------------------------------------------------------


def _ingest_notes(make_settings: MakeSettings, notes: Path, **overrides: object) -> Settings:
    settings = make_settings(allowed_folders=[notes], **overrides)
    main(["ingest"], settings=settings)
    return settings


def test_ask_prints_the_answer_and_its_sources(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
) -> None:
    settings = _ingest_notes(make_settings, notes)
    capsys.readouterr()
    fake_llm.reply = "It is a plain note [1]."

    assert main(["ask", "what", "is", "the", "plain", "note"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "It is a plain note [1]." in out
    assert "Sources:" in out
    assert "[1] b.txt, lines 1-1" in out
    assert fake_llm.calls  # the model was asked


def test_ask_refuses_when_no_note_is_relevant_and_never_calls_the_model(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
) -> None:
    settings = _ingest_notes(make_settings, notes)
    capsys.readouterr()

    assert main(["ask", "zebra", "quantum", "giraffe"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "don't have enough information" in out
    assert "Sources:" not in out
    assert fake_llm.calls == []


def test_ask_reports_a_stopped_llm_server(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
) -> None:
    settings = _ingest_notes(make_settings, notes)
    capsys.readouterr()
    fake_llm.error = LLMUnavailableError("Ollama is not reachable; run `ollama serve`")

    assert main(["ask", "plain", "note"], settings=settings) == 1

    assert "ollama serve" in capsys.readouterr().err


def test_ask_without_the_embedding_model_explains_what_to_do(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["ask", "anything"], settings=make_settings()) == 1

    assert "download-model" in capsys.readouterr().err


# --- eval and offline checks ------------------------------------------------------------------


def test_eval_prints_retrieval_and_gate_sections(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, fake_model: HashingEmbedder
) -> None:
    assert main(["eval"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "EVALUATION" in out
    assert "corpus: 12 documents" in out
    assert "RETRIEVAL" in out
    assert "RELEVANCE GATE" in out
    assert "ANSWERS" not in out


def test_eval_with_answers_checks_them_and_writes_json(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.json"

    assert main(["eval", "--answers", "--output", str(output)], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "ANSWERS" in out
    assert "unanswerable" in out
    assert fake_llm.calls  # the model was asked about at least one question
    import json

    assert json.loads(output.read_text(encoding="utf-8"))["corpus"]["documents"] == 12
    assert f"written to {output}" in out


def test_eval_accepts_experiment_settings(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, fake_model: HashingEmbedder
) -> None:
    assert (
        main(
            ["eval", "--chunk-size", "300", "--chunk-overlap", "30", "--top-k", "3"],
            settings=make_settings(),
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "chunk size 300, overlap 30, top_k 3" in out


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--chunk-size", "50"], "at least 100"),
        (["--chunk-size", "200", "--chunk-overlap", "200"], "smaller than the chunk size"),
        (["--top-k", "99"], "between 1 and 50"),
        (["--min-score", "1.5"], "between 0 and 1"),
    ],
)
def test_eval_rejects_bad_experiment_settings(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    flags: list[str],
    message: str,
) -> None:
    assert main(["eval", *flags], settings=make_settings()) == 1

    assert message in capsys.readouterr().err


def test_eval_answers_needs_a_running_llm(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.error = LLMUnavailableError("Ollama is not reachable; run `ollama serve`")

    assert main(["eval", "--answers"], settings=make_settings()) == 1

    assert "ollama serve" in capsys.readouterr().err


def test_eval_without_the_embedding_model_explains_what_to_do(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings
) -> None:
    assert main(["eval"], settings=make_settings()) == 1

    assert "download-model" in capsys.readouterr().err


def test_eval_never_touches_the_users_data_folder(
    make_settings: MakeSettings, fake_model: HashingEmbedder, data_dir: Path
) -> None:
    main(["eval"], settings=make_settings())

    assert not data_dir.exists()


def test_offline_check_passes_when_nothing_leaves_the_machine(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, fake_model: HashingEmbedder
) -> None:
    assert main(["offline-check"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "network guard self-test" in out
    assert "outbound connection attempts blocked: 0" in out
    assert "RESULT: PASS" in out


def test_offline_check_fails_if_the_pipeline_tries_to_reach_outside(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    real_run = cli.run_evaluation

    def leaky(*args: object, **kwargs: object) -> object:
        try:
            socket.create_connection(("203.0.113.5", 443), timeout=1)
        except OSError:
            pass  # a careless library swallowing the failure must still be reported
        return real_run(*args, **kwargs)

    monkeypatch.setattr(cli, "run_evaluation", leaky)

    assert main(["offline-check"], settings=make_settings()) == 1

    out = capsys.readouterr().out
    assert "203.0.113.5:443" in out
    assert "RESULT: FAIL" in out


def test_the_offline_flag_wraps_any_command_and_reports(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings
) -> None:
    assert main(["--offline", "types"], settings=make_settings()) == 0

    assert "outbound connection attempts blocked: 0" in capsys.readouterr().err


def test_the_offline_flag_fails_a_command_that_reaches_outside(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def leaky(*_args: object) -> int:
        try:
            socket.create_connection(("203.0.113.5", 443), timeout=1)
        except OSError:
            pass
        return 0

    monkeypatch.setattr(cli, "_cmd_types", leaky)

    assert main(["--offline", "types"], settings=make_settings()) == 1

    assert "203.0.113.5:443" in capsys.readouterr().err


# --- backup and restore -------------------------------------------------------------------------


def test_backup_and_restore_commands_round_trip(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    tmp_path: Path,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()
    archive = tmp_path / "saved" / "library.zip"

    assert main(["backup", "--to", str(archive)], settings=settings) == 0
    out = capsys.readouterr().out
    assert "backup saved:" in out
    assert "documents:      2" in out
    assert "private notes" in out

    restored = tmp_path / "restored"
    assert main(["restore", str(archive), "--to", str(restored)], settings=settings) == 0
    out = capsys.readouterr().out
    assert "checksums verified" in out
    assert f"DATA_DIR={restored}" in out
    assert (restored / "reyleight.db").is_file()


def test_backup_refuses_to_overwrite_and_reports_it(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "library.zip"
    assert main(["backup", "--to", str(archive)], settings=make_settings()) == 0
    capsys.readouterr()

    assert main(["backup", "--to", str(archive)], settings=make_settings()) == 1

    assert "already exists" in capsys.readouterr().err


def test_backup_without_a_library_explains_itself(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, tmp_path: Path
) -> None:
    assert main(["backup", "--to", str(tmp_path / "x.zip")], settings=make_settings()) == 1

    assert "no library found" in capsys.readouterr().err


def test_restore_into_a_non_empty_folder_is_refused(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "library.zip"
    main(["backup", "--to", str(archive)], settings=make_settings())
    capsys.readouterr()
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file.txt").write_text("x")

    assert main(["restore", str(archive), "--to", str(busy)], settings=make_settings()) == 1

    assert "not empty" in capsys.readouterr().err


def test_restore_requires_a_target(make_settings: MakeSettings, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["restore", str(tmp_path / "a.zip")], settings=make_settings())

    assert exc.value.code == 2


def test_the_default_backup_location_is_outside_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))

    path = cli._default_backup_path()

    assert path.parent == tmp_path / "appdata" / "Reyleight" / "backups"
    assert path.name.startswith("reyleight-backup-") and path.suffix == ".zip"


# --- doctor ---------------------------------------------------------------------------------


def test_doctor_reports_a_healthy_library_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
) -> None:
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["doctor"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "OK    database" in out
    assert "OK    stored files" in out
    assert "0 failure(s)" in out


def test_doctor_exits_one_and_names_the_fix_when_something_is_broken(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings
) -> None:
    assert main(["doctor"], settings=make_settings()) == 1

    out = capsys.readouterr().out
    assert "FAIL  database" in out
    assert "fix: alembic upgrade head" in out


def test_doctor_accepts_fix_and_quick(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["doctor", "--fix", "--quick"], settings=make_settings()) in (0, 1)

    assert "failure(s)" in capsys.readouterr().out


# --- a custom evaluation set --------------------------------------------------------------------


def _write_eval_set(folder: Path) -> Path:
    (folder / "corpus").mkdir(parents=True)
    (folder / "corpus" / "garden.md").write_text("# Garden\n\nTomatoes need watering daily.\n")
    (folder / "corpus" / "bikes.txt").write_text("The red bike has a flat tyre.")
    (folder / "questions.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "tomatoes",
                        "type": "answerable",
                        "question": "How often do tomatoes need watering?",
                        "expected_sources": [{"file": "garden.md"}],
                        "answer_contains": [["daily"]],
                    },
                    {"id": "capital", "type": "unanswerable", "question": "Capital of Peru?"},
                ]
            }
        )
    )
    return folder


def test_eval_can_use_a_custom_set_from_anywhere(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    tmp_path: Path,
) -> None:
    eval_set = _write_eval_set(tmp_path / "my-set")

    assert main(["eval", "--set", str(eval_set)], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "corpus: 2 documents" in out  # the custom corpus, not the built-in 12 notes
    assert "RETRIEVAL  (1 questions" in out


def test_eval_reports_a_broken_custom_set_clearly(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    tmp_path: Path,
) -> None:
    assert main(["eval", "--set", str(tmp_path / "nope")], settings=make_settings()) == 1

    assert "evaluation set problem: corpus folder not found" in capsys.readouterr().err


def test_a_custom_set_never_touches_the_users_library(
    make_settings: MakeSettings,
    fake_model: HashingEmbedder,
    tmp_path: Path,
    data_dir: Path,
) -> None:
    eval_set = _write_eval_set(tmp_path / "my-set")
    before = sorted(p.name for p in eval_set.rglob("*"))

    main(["eval", "--set", str(eval_set)], settings=make_settings())

    assert not data_dir.exists()
    assert sorted(p.name for p in eval_set.rglob("*")) == before  # the set itself is unchanged
