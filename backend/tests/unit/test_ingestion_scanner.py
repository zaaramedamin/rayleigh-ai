import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.knowledge.ingestion.scanner import is_within_allowed, resolve_roots, scan_allowed_folders


def _make_dir_link(link: Path, target: Path) -> None:
    """Create a directory symlink, falling back to a junction on Windows."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        if result.returncode == 0:
            return
    pytest.skip("cannot create a symlink or junction in this environment")


def _make_file_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink in this environment")


def _names(files: list[Path]) -> set[str]:
    return {f.name for f in files}


@pytest.fixture
def allowed(tmp_path: Path) -> Path:
    root = tmp_path / "allowed"
    root.mkdir()
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "secret.txt").write_text("secret")
    return folder


def test_finds_supported_files_recursively(allowed: Path) -> None:
    (allowed / "a.txt").write_text("a")
    (allowed / "sub").mkdir()
    (allowed / "sub" / "b.md").write_text("b")
    (allowed / "c.markdown").write_text("c")
    (allowed / "UPPER.TXT").write_text("u")

    result = scan_allowed_folders([allowed])

    assert _names(result.files) == {"a.txt", "b.md", "c.markdown", "UPPER.TXT"}


def test_skips_unsupported_hidden_and_ignored(allowed: Path) -> None:
    (allowed / "doc.pdf").write_text("x")
    (allowed / "run.exe").write_text("x")
    (allowed / ".secret.txt").write_text("x")
    for name in (".git", ".obsidian", "node_modules"):
        (allowed / name).mkdir()
        (allowed / name / "inner.txt").write_text("x")
    (allowed / "keep.txt").write_text("x")

    result = scan_allowed_folders([allowed])

    assert _names(result.files) == {"keep.txt"}
    assert result.skipped_unsupported == 2


def test_empty_allow_list_scans_nothing(outside: Path) -> None:
    result = scan_allowed_folders([])

    assert result.files == []


def test_missing_folder_is_skipped_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    result = scan_allowed_folders([missing])

    assert result.files == []
    assert result.folders_missing == 1
    assert not missing.exists()


def test_file_outside_allow_list_is_not_allowed(allowed: Path, outside: Path) -> None:
    roots = resolve_roots([allowed])

    assert not is_within_allowed(outside / "secret.txt", roots)


def test_dotdot_paths_are_resolved_before_checking(allowed: Path, outside: Path) -> None:
    roots = resolve_roots([allowed])

    sneaky = allowed / ".." / "outside" / "secret.txt"

    assert not is_within_allowed(sneaky, roots)


def test_allow_list_entry_with_dotdot_does_not_widen_scope(
    tmp_path: Path, allowed: Path, outside: Path
) -> None:
    (allowed / "in.txt").write_text("in")

    result = scan_allowed_folders([allowed / ".." / "allowed"])

    assert _names(result.files) == {"in.txt"}


def test_directory_link_pointing_outside_is_skipped(allowed: Path, outside: Path) -> None:
    (allowed / "in.txt").write_text("in")
    _make_dir_link(allowed / "escape", outside)

    result = scan_allowed_folders([allowed])

    assert _names(result.files) == {"in.txt"}
    assert result.skipped_outside_allowlist >= 1


def test_file_symlink_pointing_outside_is_skipped(allowed: Path, outside: Path) -> None:
    (allowed / "in.txt").write_text("in")
    _make_file_link(allowed / "link.txt", outside / "secret.txt")

    result = scan_allowed_folders([allowed])

    assert _names(result.files) == {"in.txt"}
    assert result.skipped_outside_allowlist == 1


def test_nested_allow_list_entries_do_not_duplicate_files(allowed: Path) -> None:
    (allowed / "sub").mkdir()
    (allowed / "sub" / "a.txt").write_text("a")

    result = scan_allowed_folders([allowed, allowed / "sub"])

    assert len(result.files) == 1
