"""The encryption commands, end to end through the command line."""

from collections.abc import Callable
from pathlib import Path

import pytest

from app import cli
from app.cli import main
from app.core.config import Settings
from app.security import keystore
from tests.fakes import FakeLLM, HashingEmbedder

MakeSettings = Callable[..., Settings]
STRONG = "correct horse battery staple"


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """No test here may talk to a real Ollama server."""
    llm = FakeLLM(model_name="qwen3.5:4b", installed=["qwen3.5:4b"])
    monkeypatch.setattr(cli, "create_llm", lambda *_args, **_kwargs: llm)
    return llm


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> HashingEmbedder:
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


def _drop_windows_copy(folder: Path) -> None:
    keyfile = keystore.read_keyfile(folder)
    assert keyfile is not None
    keystore._write_keyfile(folder, keystore._replace(keyfile, wrapped_by_windows=None))


def test_security_reports_an_unencrypted_library(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["security"], settings=make_settings()) == 0

    out = capsys.readouterr().out
    assert "encryption:     off" in out
    assert "encrypt-library" in out


def test_encrypt_library_encrypts_and_every_command_keeps_working(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings(allowed_folders=[notes])
    main(["ingest"], settings=settings)
    capsys.readouterr()

    assert main(["encrypt-library", "--yes"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "The library is encrypted." in out
    assert "safety backup" in out
    assert "cipher /w:" in out
    assert list((tmp_path / "appdata" / "Reyleight" / "backups").glob("*.zip"))
    assert main(["status"], settings=settings) == 0
    assert "documents:       2" in capsys.readouterr().out
    fake_llm.reply = "It is a plain note [1]."
    assert main(["ask", "plain", "note"], settings=settings) == 0
    assert "[1] b.txt" in capsys.readouterr().out
    assert main(["security"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "encryption:     encrypted" in out
    assert "works for this account" in out


def test_ingesting_into_an_encrypted_library_keeps_new_notes_encrypted(
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    notes: Path,
    fake_model: HashingEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings(allowed_folders=[notes])
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)

    main(["ingest"], settings=settings)

    from tests.unit.test_encrypted_library import files_containing

    assert files_containing(migrated_data_dir, [b"plain note", b"a.md", b"b.txt"]) == []


def test_encrypt_library_asks_for_confirmation_and_changes_nothing_if_declined(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", type("Tty", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    assert main(["encrypt-library"], settings=make_settings()) == 0

    assert "Nothing was changed." in capsys.readouterr().out
    assert not (migrated_data_dir / "security.json").exists()


def test_encrypt_library_refuses_to_run_unattended_without_yes(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["encrypt-library"], settings=make_settings()) == 1

    assert "add --yes" in capsys.readouterr().err
    assert not (migrated_data_dir / "security.json").exists()


def test_encrypt_library_rejects_a_weak_passphrase_and_changes_nothing(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, "weak")

    assert main(["encrypt-library", "--yes", "--no-backup"], settings=make_settings()) == 1

    assert "passphrase" in capsys.readouterr().err
    assert not (migrated_data_dir / "security.json").exists()


def test_a_second_encrypt_run_says_it_is_already_done(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    main(["encrypt-library", "--yes", "--no-backup"], settings=make_settings())
    capsys.readouterr()

    assert main(["encrypt-library", "--yes"], settings=make_settings()) == 0

    assert "already encrypted" in capsys.readouterr().out


def test_a_locked_library_asks_for_the_passphrase_and_then_remembers_it(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings()
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    _drop_windows_copy(migrated_data_dir)
    capsys.readouterr()

    # no passphrase available (not a terminal, no variable): a clear error, never a guess
    assert main(["status"], settings=settings) == 1
    assert cli.PASSPHRASE_ENV in capsys.readouterr().err

    monkeypatch.setenv(cli.PASSPHRASE_ENV, STRONG)
    assert main(["status"], settings=settings) == 0
    assert "Unlocked. Windows will now unlock" in capsys.readouterr().err

    monkeypatch.delenv(cli.PASSPHRASE_ENV)
    assert main(["status"], settings=settings) == 0  # remembered for this Windows account


def test_a_wrong_recovery_passphrase_is_reported_not_guessed(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings()
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    _drop_windows_copy(migrated_data_dir)
    capsys.readouterr()
    monkeypatch.setenv(cli.PASSPHRASE_ENV, "not the right passphrase")

    assert main(["status"], settings=settings) == 1

    assert "does not unlock this library" in capsys.readouterr().err


def test_doctor_can_unlock_a_library_with_the_passphrase(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings()
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    _drop_windows_copy(migrated_data_dir)
    capsys.readouterr()

    main(["doctor"], settings=settings)  # nothing available to unlock: a failed check, no crash
    assert "FAIL  encryption" in capsys.readouterr().out

    monkeypatch.setenv(cli.PASSPHRASE_ENV, STRONG)
    main(["doctor"], settings=settings)
    assert "OK    stored files" in capsys.readouterr().out


def test_recovery_check_confirms_a_good_passphrase_and_rejects_a_bad_one(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings()
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    capsys.readouterr()

    monkeypatch.setenv(cli.PASSPHRASE_ENV, STRONG)
    assert main(["recovery-check"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "recovery passphrase: correct" in out
    assert "same key as Windows: yes" in out

    monkeypatch.setenv(cli.PASSPHRASE_ENV, "written down wrongly")
    assert main(["recovery-check"], settings=settings) == 1
    assert "NOT correct" in capsys.readouterr().out


def test_change_passphrase_replaces_the_old_one(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    settings = make_settings()
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    capsys.readouterr()

    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, "an entirely different passphrase")
    assert main(["change-passphrase"], settings=settings) == 0
    assert "old one no longer works" in capsys.readouterr().out

    monkeypatch.setenv(cli.PASSPHRASE_ENV, "an entirely different passphrase")
    assert main(["recovery-check"], settings=settings) == 0
    monkeypatch.setenv(cli.PASSPHRASE_ENV, STRONG)
    assert main(["recovery-check"], settings=settings) == 1


def test_change_passphrase_on_an_unencrypted_library_is_refused(
    capsys: pytest.CaptureFixture[str], make_settings: MakeSettings, migrated_data_dir: Path
) -> None:
    assert main(["change-passphrase"], settings=make_settings()) == 1

    assert "not encrypted" in capsys.readouterr().err


def test_backup_says_whether_it_is_encrypted(
    capsys: pytest.CaptureFixture[str],
    make_settings: MakeSettings,
    migrated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = make_settings()
    main(["backup", "--to", str(tmp_path / "plain.zip")], settings=settings)
    plain_out = capsys.readouterr().out
    assert "encryption:     NO" in plain_out
    assert "encrypt-library" in plain_out

    monkeypatch.setenv(cli.NEW_PASSPHRASE_ENV, STRONG)
    main(["encrypt-library", "--yes", "--no-backup"], settings=settings)
    capsys.readouterr()
    main(["backup", "--to", str(tmp_path / "enc.zip")], settings=settings)
    enc_out = capsys.readouterr().out
    assert "encryption:     yes" in enc_out
    assert "recovery passphrase" in enc_out
