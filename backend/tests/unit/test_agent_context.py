"""What the agent is told about this computer."""

import sys
from datetime import date
from pathlib import Path

import pytest

from app.agent.context import KNOWN_FOLDERS, describe_computer, windows_folder

TODAY = date(2026, 10, 8)


def lookup_in(places: dict[str, Path]):  # type: ignore[no-untyped-def]
    return lambda name: places.get(name)


def test_the_note_gives_the_date_the_system_and_the_home_folder(tmp_path: Path) -> None:
    note = describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({}))

    lines = note.splitlines()
    assert lines[0].startswith("About this computer (use these exact paths")
    assert "never invent a path, a name or an address" in lines[0]
    assert lines[1] == "- Today is Thursday 8 October 2026."
    assert lines[2] == f"- It runs Windows. The user's home folder is {tmp_path}."


def test_a_folder_windows_puts_somewhere_else_is_found_where_windows_says(tmp_path: Path) -> None:
    moved = tmp_path / "OneDrive" / "Documents"
    moved.mkdir(parents=True)
    (tmp_path / "Documents").mkdir()  # the plain one also exists; Windows' answer wins

    note = describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({"Documents": moved}))

    assert f"- The user's Documents folder is {moved}." in note
    assert str(tmp_path / "Documents") + "." not in note


def test_a_folder_in_another_language_is_named_by_what_it_is_for(tmp_path: Path) -> None:
    desktop = tmp_path / "OneDrive" / "Bureau"
    desktop.mkdir(parents=True)

    note = describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({"Desktop": desktop}))

    assert f"- The user's Desktop folder is {desktop}." in note


def test_without_an_answer_from_windows_the_usual_place_in_the_home_folder_is_used(
    tmp_path: Path,
) -> None:
    (tmp_path / "Downloads").mkdir()

    note = describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({}))

    assert f"- The user's Downloads folder is {tmp_path / 'Downloads'}." in note


def test_a_folder_that_does_not_exist_is_not_listed(tmp_path: Path) -> None:
    ghost = tmp_path / "gone"

    note = describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({"Music": ghost}))

    assert "Music" not in note and "Pictures" not in note and "Desktop" not in note


def test_the_notes_folders_are_listed_when_they_exist(tmp_path: Path) -> None:
    mine, other = tmp_path / "notes", tmp_path / "work"
    mine.mkdir()
    other.mkdir()

    note = describe_computer(
        today=TODAY,
        home=tmp_path,
        lookup=lookup_in({}),
        notes_folders=[mine, other, tmp_path / "gone"],
    )

    assert note.splitlines()[-1] == f"- The user's notes are in: {mine}; {other}."


def test_no_notes_line_without_notes_folders(tmp_path: Path) -> None:
    assert "notes are in" not in describe_computer(today=TODAY, home=tmp_path, lookup=lookup_in({}))


def test_the_date_follows_the_clock_when_none_is_given(tmp_path: Path) -> None:
    from datetime import datetime

    note = describe_computer(home=tmp_path, lookup=lookup_in({}), now=datetime(2027, 1, 3, 9, 0))

    assert "- Today is Sunday 3 January 2027." in note


def test_every_folder_the_note_can_name_has_a_windows_id() -> None:
    assert set(KNOWN_FOLDERS) == {
        "Desktop",
        "Documents",
        "Downloads",
        "Pictures",
        "Music",
        "Videos",
    }
    assert all(len(guid) == 36 for guid in KNOWN_FOLDERS.values())


@pytest.mark.skipif(sys.platform != "win32", reason="the known folders are a Windows feature")
def test_windows_itself_is_asked_and_its_answer_is_a_real_folder() -> None:
    found = {name: windows_folder(name) for name in KNOWN_FOLDERS}

    assert any(path is not None and path.is_dir() for path in found.values())
    assert windows_folder("Recycle Bin") is None  # a name this module does not know
