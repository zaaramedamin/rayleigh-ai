"""What the owner has allowed the agent to do, and what happens to each request because of it.

Two things live here: the grants (DATA_DIR/agent.json: a master switch, which tools are switched
on, and whether the web may be read) and the policy that turns a tool's level plus the grants into
one of three decisions: allow, ask the owner, or deny.

Rules that never bend:
- Everything is off until the owner switches it on. A missing or damaged grants file reads as off.
- Only a read-only tool is ever allowed without asking. Anything that opens, writes or reaches the
  internet asks the owner every time, with no "always".
- A destructive tool is denied whatever the grants say.
- The decision depends only on the tool's declared level and the grants. Nothing the model writes
  can change it, and approval is an action of the owner (a button, a terminal answer), never text.
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.agent.tools import Tool

logger = logging.getLogger(__name__)

GRANTS_FILENAME = "agent.json"
_LOCK = threading.RLock()

Decision = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class Grants:
    enabled: bool = False  # the master switch
    tools: frozenset[str] = field(default_factory=frozenset)  # tools switched on, by name
    web: bool = False  # whether a tool may reach the internet at all


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str  # in plain words, for the owner


# --- the policy --------------------------------------------------------------------------------


def decide(tool: Tool, grants: Grants) -> Verdict:
    """What to do with a valid request for `tool`."""
    if not grants.enabled:
        return Verdict("deny", "The agent is switched off.")
    if tool.level == "destructive":
        return Verdict("deny", "Destructive actions are never allowed.")
    if tool.name not in grants.tools:
        return Verdict("deny", f"The tool {tool.name} is not switched on.")
    if tool.level == "read_local":
        return Verdict("allow", "It only reads, and you switched it on.")
    if tool.level == "open_local":
        return Verdict(
            "ask", "This opens something on your computer, so I need your approval each time."
        )
    if tool.level == "external_read":
        if not grants.web:
            return Verdict("deny", "Web access is switched off.")
        return Verdict(
            "ask", "This reads a page from the internet, so I need your approval each time."
        )
    if tool.level == "write_local":
        return Verdict(
            "ask", "This changes something on your computer, so I need your approval each time."
        )
    return Verdict("deny", f"Unknown permission level {tool.level!r}.")


# --- storage -----------------------------------------------------------------------------------


def _path(data_dir: Path) -> Path:
    return data_dir / GRANTS_FILENAME


def load_grants(data_dir: Path) -> Grants:
    """The saved grants. A missing, unreadable or malformed file is everything off."""
    try:
        raw = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Grants()
    if not isinstance(raw, dict):
        return Grants()
    names = raw.get("tools")
    return Grants(
        enabled=raw.get("enabled") is True,
        tools=frozenset(n for n in names if isinstance(n, str))
        if isinstance(names, list)
        else frozenset(),
        web=raw.get("web") is True,
    )


def save_grants(data_dir: Path, grants: Grants) -> None:
    with _LOCK:
        target = _path(data_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(
                    {"enabled": grants.enabled, "tools": sorted(grants.tools), "web": grants.web},
                    tmp,
                    indent=1,
                )
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    logger.info(
        "agent grants saved enabled=%s tools=%d web=%s",
        grants.enabled,
        len(grants.tools),
        grants.web,
    )
