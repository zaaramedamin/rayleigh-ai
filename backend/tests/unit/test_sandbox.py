"""The parser sandbox: a bad file or a bad parser becomes a failed file, never a hung program."""

import time

import pytest

from app.knowledge.ingestion.parsers import ParseError
from app.knowledge.ingestion.sandbox import run_in_sandbox

SAMPLES = "tests.sandbox_samples"


def run(function: str, data: bytes = b"hello", **kwargs: float) -> object:
    return run_in_sandbox(f"{SAMPLES}:{function}", data, **kwargs)  # type: ignore[arg-type]


def test_a_reader_in_the_sandbox_returns_its_text() -> None:
    extracted = run("echo", "héllo wörld — ✓".encode())

    assert extracted.text == "héllo wörld — ✓"  # type: ignore[attr-defined]
    assert extracted.page_starts is None  # type: ignore[attr-defined]


def test_page_starts_come_back() -> None:
    extracted = run("two_pages", b"first page|second page")

    assert extracted.text == "first page\nsecond page"  # type: ignore[attr-defined]
    assert extracted.page_starts == (0, 11)  # type: ignore[attr-defined]


def test_a_large_file_goes_through() -> None:
    data = ("line of text\n" * 200_000).encode()  # 2.6 MB

    assert len(run("echo", data).text) == len(data)  # type: ignore[attr-defined]


def test_the_readers_own_reason_comes_through() -> None:
    with pytest.raises(ParseError, match="^encrypted$"):
        run("refuses")


def test_a_reader_that_hangs_is_stopped_at_the_time_limit() -> None:
    started = time.monotonic()

    with pytest.raises(ParseError, match="^timeout$"):
        run("hangs", timeout=2)

    assert time.monotonic() - started < 30


def test_a_reader_that_dies_is_a_failed_file() -> None:
    with pytest.raises(ParseError, match="^crashed$"):
        run("dies")


def test_a_reader_that_breaks_is_a_failed_file_and_does_not_say_what_was_inside() -> None:
    with pytest.raises(ParseError) as raised:
        run("raises", b"my password is hunter2")

    assert str(raised.value) == "crashed"
    assert "hunter2" not in repr(raised.value) and raised.value.__cause__ is None


def test_output_that_is_too_large_is_rejected() -> None:
    with pytest.raises(ParseError, match="^too_large_output$"):
        run("huge", max_output=1_000_000)


def test_the_same_output_is_fine_under_a_bigger_cap() -> None:
    assert len(run("huge", max_output=10_000_000).text) == 5_000_000  # type: ignore[attr-defined]


@pytest.mark.parametrize("function", ["wrong_type", "bad_pages"])
def test_an_answer_that_is_not_a_proper_result_is_a_crash(function: str) -> None:
    with pytest.raises(ParseError, match="^crashed$"):
        run(function)


def test_a_reader_that_does_not_exist_is_a_crash_not_an_exception() -> None:
    with pytest.raises(ParseError, match="^crashed$"):
        run("no_such_reader")


def test_a_reader_cannot_reach_the_network() -> None:
    with pytest.raises(ParseError, match="^crashed$"):
        run("phones_home")


def test_the_worker_does_not_get_this_programs_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.knowledge.ingestion.sandbox import _worker_environment

    monkeypatch.setenv("REYLEIGHT_PASSPHRASE", "do not share")
    monkeypatch.setenv("ACCESS_REQUIRED", "false")

    environment = _worker_environment()

    assert "REYLEIGHT_PASSPHRASE" not in environment
    assert "ACCESS_REQUIRED" not in environment
