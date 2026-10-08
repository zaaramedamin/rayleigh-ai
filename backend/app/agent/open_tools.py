"""Tools that open things on this computer. The owner is asked before every one.

- open_path: a file or a folder, opened with the program Windows already uses for it (a document in
  its editor, a folder in File Explorer). It never runs a program: files that Windows would run or
  that can carry a script (.exe, .bat, .ps1, .lnk, .msi, ...) are refused, so is a path on another
  computer or a device, and the path must exist and be written in full.
- open_app: a program from a short fixed list (Notepad, Calculator, Paint, File Explorer), started
  alone, with no arguments the model could use to make it do something else.

What actually happens in the world is passed in (`opener`, `launcher`), so the tests never open
anything. The defaults are the real ones.
"""

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.agent.tools import Param, Tool, ToolError

MAX_PATH_CHARS = 400

# Types that Windows runs or that can carry a script or a link to something that does. Opening one
# of these is running a program, which this tool never does.
BLOCKED_SUFFIXES = frozenset(
    {
        ".exe", ".com", ".scr", ".pif", ".bat", ".cmd", ".ps1", ".psm1", ".psd1", ".ps1xml",
        ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".msi", ".msp", ".mst", ".msc",
        ".lnk", ".url", ".reg", ".hta", ".cpl", ".jar", ".dll", ".sys", ".drv", ".ocx",
        ".inf", ".gadget", ".appx", ".msix", ".appinstaller", ".application", ".xbap",
        ".py", ".pyw", ".pyz", ".sh", ".iso", ".vhd", ".vhdx", ".settingcontent-ms",
        ".library-ms", ".search-ms", ".theme", ".diagcab", ".chm",
        # Office files that carry macros, and add-ins that run code when opened.
        ".docm", ".dotm", ".xlsm", ".xltm", ".xlam", ".xlsb", ".xll", ".wll", ".pptm", ".potm",
        ".ppam", ".ppsm", ".sldm", ".iqy", ".slk", ".vsto", ".one", ".onepkg",
        # Pages and images that run script in the browser, and remote connections.
        ".html", ".htm", ".xhtml", ".shtml", ".mht", ".mhtml", ".svg", ".svgz", ".rdp", ".jnlp",
        ".website", ".webloc", ".scf", ".psc1", ".msh", ".msh1", ".msh2", ".mshxml", ".ade",
        ".adp", ".mde", ".mdb", ".accdb",
    }
)  # fmt: skip

# Programs that can be started by name. Each starts with no arguments.
APPS: Mapping[str, tuple[str, str]] = {
    "notepad": ("Notepad", "notepad.exe"),
    "calculator": ("Calculator", "calc.exe"),
    "paint": ("Paint", "mspaint.exe"),
    "file_explorer": ("File Explorer", "explorer.exe"),
}

Opener = Callable[[Path], None]
Launcher = Callable[[str], None]


def _real_opener(path: Path) -> None:
    startfile = getattr(os, "startfile", None)  # Windows only
    if startfile is None:
        raise ToolError("Opening files is only available on Windows.")
    startfile(str(path))


def _real_launcher(command: str) -> None:
    try:
        subprocess.Popen([command], shell=False)  # noqa: S603 - the command is from APPS only
    except OSError as exc:
        raise ToolError(f"{command} could not be started ({exc.__class__.__name__}).") from exc


def checked_path(text: str) -> Path:
    """The file or folder `text` names, if it is safe to open. Raises ToolError with the reason."""
    if text.startswith(("\\\\", "//")):
        raise ToolError("That is a path on another computer or a device, which is never opened.")
    if "%" in text:
        raise ToolError("Write the full path, without % variables.")
    path = Path(text)
    if not path.is_absolute():
        raise ToolError("Write the full path, starting with the drive, for example C:\\Users\\...")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolError(f"{text} does not exist.") from exc
    if str(resolved).startswith("\\\\"):
        raise ToolError("That leads to another computer or a device, which is never opened.")
    if resolved.is_dir():
        return resolved
    suffix = resolved.suffix.lower()
    if not suffix:
        raise ToolError("A file with no type is not opened: Windows cannot tell what it would do.")
    if suffix in BLOCKED_SUFFIXES:
        raise ToolError(
            f"{suffix} files are programs or scripts. Opening them would run them, and that is "
            "never done."
        )
    return resolved


def open_path_tool(opener: Opener = _real_opener) -> Tool:
    def run(args: Mapping[str, Any]) -> str:
        path = checked_path(args["path"])
        opener(path)
        return f"Opened {'the folder' if path.is_dir() else 'the file'} {path}."

    def describe(args: Mapping[str, Any]) -> str:
        return f"Open {args['path']} with the program Windows uses for it"

    return Tool(
        name="open_path",
        description=(
            "Open a file or a folder on this computer with its usual program (a document in its "
            "editor, a folder in File Explorer). Give the full path. It cannot open programs or "
            "scripts. The user is asked to approve each time."
        ),
        level="open_local",
        params={
            "path": Param(
                "string",
                "The full path, such as C:\\Users\\me\\notes\\plan.txt.",
                max_length=MAX_PATH_CHARS,
            )
        },
        run=run,
        describe=describe,
    )


def open_app_tool(launcher: Launcher = _real_launcher) -> Tool:
    def run(args: Mapping[str, Any]) -> str:
        label, command = APPS[args["app"]]
        launcher(command)
        return f"Started {label}."

    def describe(args: Mapping[str, Any]) -> str:
        return f"Start {APPS[args['app']][0]}"

    return Tool(
        name="open_app",
        description=(
            "Start one of these programs on this computer: "
            + ", ".join(f"{name} ({label})" for name, (label, _) in APPS.items())
            + ". It starts empty. The user is asked to approve each time."
        ),
        level="open_local",
        params={"app": Param("string", "Which program to start.", choices=tuple(APPS))},
        run=run,
        describe=describe,
    )
