"""The read-only questions put to Windows: BitLocker and the log of blocked files."""

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.operations import windows
from app.operations.windows import (
    BlockedFile,
    bitlocker_protection,
    parse_block_events,
    recent_blocked_files,
    summarise,
)

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
SITE = "\\Device\\HarddiskVolume3\\Users\\someone\\proj\\.venv\\Lib\\site-packages"
PYTHON = "\\Device\\HarddiskVolume3\\Program Files\\Python313\\python.exe"
CHROME = "\\Device\\HarddiskVolume3\\Program Files\\Google\\Chrome\\Application\\chrome.exe"


def event(
    *,
    event_id: int = 3077,
    file: str,
    process: str = PYTHON,
    when: datetime = NOW,
    modern: bool = True,
) -> str:
    """One <Event> as `wevtutil qe ... /f:xml` writes it. 3077 names its fields in words, 3033 in
    buffers."""
    file_field, process_field = (
        ("File Name", "Process Name") if modern else ("FileNameBuffer", "ProcessNameBuffer")
    )
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S.%f") + "0Z"  # Windows writes seven digits
    return (
        "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System>"
        f"<EventID>{event_id}</EventID><TimeCreated SystemTime='{stamp}'/></System><EventData>"
        f"<Data Name='{file_field}'>{file}</Data><Data Name='{process_field}'>{process}</Data>"
        "</EventData></Event>"
    )


def parse(*events: str, days: int = 14) -> list[BlockedFile]:
    return parse_block_events("\n".join(events), now=NOW, days=days)


# --- reading the block log --------------------------------------------------------------------


def test_a_blocked_library_file_is_named_as_it_is_known_inside_its_package() -> None:
    blocked = parse(event(file=f"{SITE}\\scipy\\fft\\_pocketfft\\pyduccfft.pyd"))

    assert [(b.name, b.count) for b in blocked] == [("scipy\\fft\\_pocketfft\\pyduccfft.pyd", 1)]
    assert "someone" not in blocked[0].name  # no user name, no drive


def test_events_that_name_their_fields_in_buffers_are_read_too() -> None:
    blocked = parse(event(event_id=3033, file=f"{SITE}\\sklearn\\x.pyd", modern=False))

    assert [b.name for b in blocked] == ["sklearn\\x.pyd"]


def test_blocks_of_other_programs_are_left_out() -> None:
    blocked = parse(
        event(
            file="\\Device\\HarddiskVolume3\\Program Files\\Google\\Chrome\\a.dll", process=CHROME
        ),
        event(
            file="\\Device\\HarddiskVolume3\\Games\\GTA\\Menyoo.asi",
            process="\\Device\\HarddiskVolume3\\Games\\GTA\\GTA5.exe",
        ),
    )

    assert blocked == []


def test_a_library_file_is_ours_whichever_program_loaded_it() -> None:
    blocked = parse(
        event(
            file=f"{SITE}\\numpy\\core\\x.pyd",
            process="\\Device\\HarddiskVolume3\\Tools\\other.exe",
        )
    )

    assert [b.name for b in blocked] == ["numpy\\core\\x.pyd"]


def test_a_blocked_launcher_in_the_environment_is_ours() -> None:
    launcher = "\\Device\\HarddiskVolume3\\Users\\someone\\proj\\.venv\\Scripts\\alembic.exe"

    blocked = parse(
        event(file=launcher, process="\\Device\\HarddiskVolume3\\Windows\\System32\\cmd.exe")
    )

    assert [b.name for b in blocked] == ["alembic.exe"]


def test_old_events_are_left_out() -> None:
    old = NOW - timedelta(days=15)

    assert parse(event(file=f"{SITE}\\a\\b.pyd", when=old)) == []
    assert [b.name for b in parse(event(file=f"{SITE}\\a\\b.pyd", when=old), days=30)] == [
        "a\\b.pyd"
    ]


def test_the_same_file_is_counted_once_with_its_latest_time_and_the_newest_comes_first() -> None:
    blocked = parse(
        event(file=f"{SITE}\\a\\first.pyd", when=NOW - timedelta(days=3)),
        event(file=f"{SITE}\\b\\second.pyd", when=NOW - timedelta(days=1)),
        event(file=f"{SITE}\\a\\first.pyd", when=NOW - timedelta(hours=2)),
        event(file=f"{SITE}\\a\\first.pyd", when=NOW - timedelta(days=5)),
    )

    assert [(b.name, b.count) for b in blocked] == [("a\\first.pyd", 3), ("b\\second.pyd", 1)]
    assert blocked[0].last_seen == NOW - timedelta(hours=2)


@pytest.mark.parametrize("text", ["", "not xml at all", "<Event>", "<a><b></a>"])
def test_output_that_is_not_events_gives_nothing_instead_of_an_error(text: str) -> None:
    assert parse_block_events(text, now=NOW) == []


def test_an_event_without_a_readable_time_or_file_is_skipped() -> None:
    no_time = event(file=f"{SITE}\\a\\b.pyd").replace("SystemTime='", "SystemTime='nonsense-")
    no_file = event(file="")

    assert parse(no_time, no_file) == []


def test_the_summary_lists_a_few_and_says_how_many_more() -> None:
    blocked = [BlockedFile(f"p{n}.pyd", 1 if n else 3, NOW) for n in range(6)]

    text = summarise(blocked, limit=3)

    assert text == "p0.pyd (3 times), p1.pyd, p2.pyd and 3 more"
    assert summarise(blocked[:2]) == "p0.pyd (3 times), p1.pyd"


# --- asking Windows ---------------------------------------------------------------------------


def answer(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_the_block_log_is_asked_for_the_right_events_and_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[list[str]] = []

    def fake(command: list[str]) -> subprocess.CompletedProcess[str]:
        asked.append(command)
        return answer(event(file=f"{SITE}\\a\\b.pyd", when=datetime.now(UTC)))

    monkeypatch.setattr(windows, "_run", fake)

    blocked = recent_blocked_files(days=7)

    assert blocked is not None and [b.name for b in blocked] == ["a\\b.pyd"]
    command = asked[0]
    assert command[:3] == ["wevtutil", "qe", "Microsoft-Windows-CodeIntegrity/Operational"]
    assert any(
        "EventID=3033 or EventID=3077" in part and str(7 * 24 * 3600 * 1000) in part
        for part in command
    )
    assert "/f:xml" in command  # read-only: only a query


def test_a_block_log_that_cannot_be_read_is_not_an_empty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(windows, "_run", lambda _c: answer(returncode=5))
    assert recent_blocked_files() is None

    monkeypatch.setattr(windows, "_run", lambda _c: None)
    assert recent_blocked_files() is None


@pytest.mark.skipif(sys.platform != "win32", reason="the drive of a path is a Windows idea")
class TestBitLocker:
    def test_the_number_windows_reports_for_the_drive_is_passed_on(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        asked: list[list[str]] = []
        monkeypatch.setattr(windows, "_run", lambda c: (asked.append(c), answer("1\r\n"))[1])

        assert bitlocker_protection(tmp_path) == 1

        script = asked[0][-1]
        drive = tmp_path.resolve().drive
        assert asked[0][0] == "powershell.exe" and "-NoProfile" in asked[0]
        assert f"NameSpace('{drive}\\')" in script and "BitLockerProtection" in script

    @pytest.mark.parametrize(("said", "number"), [("0", 0), ("2\n", 2), (" 3 ", 3)])
    def test_other_numbers_are_passed_on_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, said: str, number: int
    ) -> None:
        monkeypatch.setattr(windows, "_run", lambda _c: answer(said))

        assert bitlocker_protection(tmp_path) == number

    @pytest.mark.parametrize("said", ["", "On", "1; Remove-Item x", "12345"])
    def test_an_answer_that_is_not_a_small_number_is_not_believed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, said: str
    ) -> None:
        monkeypatch.setattr(windows, "_run", lambda _c: answer(said))

        assert bitlocker_protection(tmp_path) is None

    def test_a_failing_or_missing_powershell_means_it_could_not_be_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(windows, "_run", lambda _c: answer("1", returncode=1))
        assert bitlocker_protection(tmp_path) is None

        monkeypatch.setattr(windows, "_run", lambda _c: None)
        assert bitlocker_protection(tmp_path) is None

    def test_a_network_share_has_no_drive_to_ask_about(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            windows, "_run", lambda _c: pytest.fail("PowerShell must not be started")
        )

        assert bitlocker_protection(Path("//server/share/library")) is None


def test_nothing_is_started_where_windows_is_not_there(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(windows.sys, "platform", "linux")
    monkeypatch.setattr(
        windows.subprocess, "run", lambda *_a, **_k: pytest.fail("started a program")
    )

    assert windows._run(["anything"]) is None
    assert bitlocker_protection(Path(".")) is None


def test_a_program_that_is_missing_or_too_slow_gives_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(windows.sys, "platform", "win32")

    def missing(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("wevtutil")

    monkeypatch.setattr(windows.subprocess, "run", missing)
    assert windows._run(["wevtutil"]) is None

    def slow(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired("wevtutil", 20)

    monkeypatch.setattr(windows.subprocess, "run", slow)
    assert windows._run(["wevtutil"]) is None


def test_scikit_learn_is_asked_in_another_process_so_this_one_does_not_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(windows.subprocess, "run", run)

    assert windows.scikit_learn_loads() is False  # a failing import: it cannot be loaded here
    assert seen[0] == [sys.executable, "-c", "import sklearn.metrics"]


def test_scikit_learn_that_loads_is_reported_so_and_a_failure_to_ask_is_not_a_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        windows.subprocess, "run", lambda c, **_k: subprocess.CompletedProcess(c, 0)
    )
    assert windows.scikit_learn_loads() is True

    def slow(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired("python", 120)

    monkeypatch.setattr(windows.subprocess, "run", slow)
    assert windows.scikit_learn_loads() is None


# --- the library that turns text into vectors ------------------------------------------------


def run_returning(stderr: str = "", code: int = 0) -> object:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        run.seen = (command, kwargs)  # type: ignore[attr-defined]
        return subprocess.CompletedProcess(command, code, stdout="", stderr=stderr)

    return run


def test_the_embedding_library_is_loaded_the_way_the_application_loads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_returning()
    monkeypatch.setattr(windows.subprocess, "run", run)

    assert windows.embedding_library_loads() == (True, "")

    command, kwargs = run.seen  # type: ignore[attr-defined]
    assert command[0] == sys.executable and command[1] == "-c"
    assert "import_sentence_transformer()" in command[2]  # with the scikit-learn placeholder
    assert (Path(str(kwargs["cwd"])) / "app" / "__init__.py").is_file()  # where `app` can be found


def test_a_library_that_cannot_be_loaded_gives_the_last_line_of_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "ImportError: DLL load failed while importing _C: An Application Control policy has "
        "blocked this file.\n\n"
    )
    monkeypatch.setattr(windows.subprocess, "run", run_returning(trace, code=1))

    loads, reason = windows.embedding_library_loads()

    assert loads is False
    assert reason == (
        "ImportError: DLL load failed while importing _C: An Application Control policy has "
        "blocked this file."
    )


def test_a_failure_with_no_message_and_a_very_long_one_are_both_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(windows.subprocess, "run", run_returning("", code=1))
    assert windows.embedding_library_loads() == (False, "it could not be imported")

    monkeypatch.setattr(windows.subprocess, "run", run_returning("E" * 900, code=1))
    assert len(windows.embedding_library_loads()[1]) == 300


def test_an_embedding_library_that_could_not_be_asked_about_is_not_called_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("python", 180)

    monkeypatch.setattr(windows.subprocess, "run", slow)

    assert windows.embedding_library_loads() == (None, "")
