"""The tools that open things on the computer. Nothing here opens or starts anything real."""

from pathlib import Path

import pytest

from app.agent.open_tools import (
    APPS,
    BLOCKED_SUFFIXES,
    checked_path,
    open_app_tool,
    open_path_tool,
)
from app.agent.permissions import Grants, decide
from app.agent.tools import ArgumentError, ToolError, validate_arguments

# --- open_path -----------------------------------------------------------------------------


@pytest.fixture
def opened() -> list[Path]:
    return []


def test_a_file_is_opened_with_its_usual_program(tmp_path: Path, opened: list[Path]) -> None:
    note = tmp_path / "plan.txt"
    note.write_text("hello")
    tool = open_path_tool(opened.append)

    result = tool.run(validate_arguments(tool, {"path": str(note)}))

    assert opened == [note.resolve()] and result == f"Opened the file {note.resolve()}."


def test_a_folder_is_opened_in_the_file_explorer(tmp_path: Path, opened: list[Path]) -> None:
    tool = open_path_tool(opened.append)

    result = tool.run({"path": str(tmp_path)})

    assert opened == [tmp_path.resolve()] and result.startswith("Opened the folder ")


@pytest.mark.parametrize(
    "name",
    [
        "setup.exe",
        "run.BAT",
        "x.cmd",
        "s.ps1",
        "m.vbs",
        "a.js",
        "i.msi",
        "l.lnk",
        "w.url",
        "k.reg",
        "p.py",
        "d.dll",
        "i.iso",
        "a.hta",
        "h.chm",
    ],
)
def test_a_program_or_script_is_never_opened(tmp_path: Path, opened: list[Path], name: str) -> None:
    (tmp_path / name).write_text("x")

    with pytest.raises(ToolError, match="programs or scripts"):
        open_path_tool(opened.append).run({"path": str(tmp_path / name)})
    assert opened == []


def test_a_file_with_no_type_is_not_opened(tmp_path: Path, opened: list[Path]) -> None:
    (tmp_path / "mystery").write_text("x")

    with pytest.raises(ToolError, match="no type"):
        open_path_tool(opened.append).run({"path": str(tmp_path / "mystery")})
    assert opened == []


def test_a_path_that_does_not_exist_is_refused(tmp_path: Path, opened: list[Path]) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        open_path_tool(opened.append).run({"path": str(tmp_path / "nothing.txt")})
    assert opened == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\\\\server\\share\\file.txt", "another computer or a device"),
        ("//server/share/file.txt", "another computer or a device"),
        ("\\\\.\\C:\\file.txt", "another computer or a device"),
        ("\\\\?\\C:\\file.txt", "another computer or a device"),
        ("notes\\plan.txt", "full path"),
        ("..\\plan.txt", "full path"),
        ("~\\plan.txt", "full path"),
        ("%USERPROFILE%\\plan.txt", "variables"),
        ("plan.txt", "full path"),
    ],
)
def test_a_path_on_another_computer_a_device_or_not_written_in_full_is_refused(
    text: str, message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        checked_path(text)


def test_a_path_is_resolved_before_it_is_checked(tmp_path: Path, opened: list[Path]) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "setup.exe").write_text("x")
    sneaky = tmp_path / "sub" / ".." / "setup.exe"

    with pytest.raises(ToolError, match="programs or scripts"):
        open_path_tool(opened.append).run({"path": str(sneaky)})
    assert opened == []


def test_the_owner_is_shown_the_exact_path_and_is_always_asked(tmp_path: Path) -> None:
    tool = open_path_tool(lambda _p: None)
    target = str(tmp_path / "plan.txt")

    assert tool.effect({"path": target}) == f"Open {target} with the program Windows uses for it"
    grants = Grants(enabled=True, tools=frozenset({"open_path"}))
    assert tool.level == "open_local" and decide(tool, grants).decision == "ask"


def test_the_arguments_are_checked_like_any_tool() -> None:
    tool = open_path_tool(lambda _p: None)

    with pytest.raises(ArgumentError, match="does not take: args"):
        validate_arguments(tool, {"path": "C:\\a.txt", "args": "--run"})
    with pytest.raises(ArgumentError, match="too long"):
        validate_arguments(tool, {"path": "C:\\" + "a" * 500})
    with pytest.raises(ArgumentError, match="control characters"):
        validate_arguments(tool, {"path": "C:\\a\x00.txt"})


def test_the_blocked_types_include_everything_windows_would_run() -> None:
    for suffix in (".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msi", ".lnk", ".reg", ".dll"):
        assert suffix in BLOCKED_SUFFIXES


# --- open_app --------------------------------------------------------------------------------


def test_a_listed_program_is_started_alone() -> None:
    started: list[str] = []
    tool = open_app_tool(started.append)

    result = tool.run(validate_arguments(tool, {"app": "calculator"}))

    assert started == ["calc.exe"] and result == "Started Calculator."


@pytest.mark.parametrize(
    "app", ["cmd", "powershell", "calc.exe", "notepad.exe C:\\x.txt", "Notepad", "", "../x"]
)
def test_a_program_that_is_not_on_the_list_is_refused_before_anything_starts(app: str) -> None:
    started: list[str] = []
    tool = open_app_tool(started.append)

    with pytest.raises(ArgumentError):
        validate_arguments(tool, {"app": app})
    assert started == []


def test_no_argument_can_be_given_to_a_program() -> None:
    tool = open_app_tool(lambda _c: None)

    with pytest.raises(ArgumentError, match="does not take: arguments, file"):
        validate_arguments(tool, {"app": "notepad", "file": "C:\\x.txt", "arguments": "/p"})


def test_the_list_holds_only_harmless_programs_and_the_model_is_told_which() -> None:
    assert set(APPS) == {"notepad", "calculator", "paint", "file_explorer"}
    tool = open_app_tool(lambda _c: None)

    assert all(name in tool.description for name in APPS)
    assert "cmd" not in {command.split(".")[0] for _label, command in APPS.values()}


def test_starting_a_program_asks_every_time_and_says_which() -> None:
    tool = open_app_tool(lambda _c: None)
    grants = Grants(enabled=True, tools=frozenset({"open_app"}))

    assert tool.effect({"app": "paint"}) == "Start Paint"
    assert decide(tool, grants).decision == "ask"
    assert decide(tool, Grants(enabled=True)).decision == "deny"  # not switched on
    assert decide(tool, Grants()).decision == "deny"  # the agent is off


def test_a_program_that_cannot_start_is_a_tool_error_the_owner_hears_about() -> None:
    def failing(_command: str) -> None:
        raise ToolError("calc.exe could not be started (FileNotFoundError).")

    with pytest.raises(ToolError, match="could not be started"):
        open_app_tool(failing).run({"app": "calculator"})
