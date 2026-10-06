"""What the assistant may do in the application: this list, and nothing else.

The model can only *ask* for an action. Its request is untrusted text, so it is checked here
against the catalogue (a known name, known values, bounded text) before anything happens, and
the interface checks it again before carrying it out. A request that does not fit is dropped.

Deliberately not in the catalogue: deleting documents or folders, deleting the profile or
memories, changing the password, and anything outside the application (files, programs, the
network). Those stay with the owner, on the page made for them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.ai.llm.base import ToolCall, ToolSpec

MAX_ACTIONS_PER_REPLY = 3
MAX_TEXT_ARG_CHARS = 400

PAGES = ("home", "chat", "knowledge", "profile", "modules", "privacy", "settings")
THEMES = ("reactor", "mark3")
OPTIONS = ("interface_sounds", "spoken_replies", "animations", "notes_mode")
SWITCH = ("on", "off")

# Carried out by the server while it answers; every other action is carried out by the interface.
REMEMBER = "remember"

_THEME_NAMES = {"reactor": "Arc Reactor", "mark3": "Mark Three"}
_OPTION_NAMES = {
    "interface_sounds": "interface sounds",
    "spoken_replies": "spoken replies",
    "animations": "animations",
    "notes_mode": "notes mode",
}


@dataclass(frozen=True)
class Param:
    description: str
    choices: tuple[str, ...] = ()  # empty means free text, cut at MAX_TEXT_ARG_CHARS


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    params: Mapping[str, Param] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A request that passed validation. Every value is a plain, bounded string."""

    name: str
    args: Mapping[str, str] = field(default_factory=dict)


CATALOGUE: tuple[ActionSpec, ...] = (
    ActionSpec(
        "open_page",
        "Show one page of the application to the user.",
        {
            "page": Param(
                "home: dashboard. chat: the full conversation. knowledge: the library of "
                "folders and documents. profile: what the assistant knows and remembers about "
                "the user. modules: planned features. privacy: how data is protected. "
                "settings: theme, sound, voice, password.",
                PAGES,
            )
        },
    ),
    ActionSpec(
        "set_theme",
        "Change the colours of the interface.",
        {"theme": Param("reactor: cyan on deep blue. mark3: gold and red.", THEMES)},
    ),
    ActionSpec(
        "set_option",
        "Switch one behaviour of the application on or off.",
        {
            "option": Param(
                "interface_sounds: beeps and effects. spoken_replies: the assistant's voice. "
                "animations: moving decoration. notes_mode: answer every message only from "
                "the user's notes.",
                OPTIONS,
            ),
            "value": Param("on or off", SWITCH),
        },
    ),
    ActionSpec(
        "ask_notes",
        "Search the user's saved notes and documents and answer from them, with sources. Use it "
        "only when the user asks what their notes, files or documents say. Never use it for "
        "facts already in the user's profile or in your memories, or for general knowledge.",
        {"question": Param("The question to answer from the notes, as a full sentence.")},
    ),
    ActionSpec(
        "update_library",
        "Scan the folders the user chose for new files and make them searchable.",
    ),
    ActionSpec(
        "report_status",
        "Read out the state of the system: the backend, the local model and the library.",
    ),
    ActionSpec("clear_conversation", "Erase the messages on screen and start a new conversation."),
    ActionSpec("lock_app", "Lock the application. The password is needed to open it again."),
    ActionSpec(
        REMEMBER,
        "Save a lasting fact about the user for future conversations. Use it when the user "
        "asks you to remember something, or tells you a lasting fact or preference about "
        "themselves.",
        {"fact": Param("The fact, as one short sentence about the user.")},
    ),
)

_BY_NAME = {spec.name: spec for spec in CATALOGUE}


def tool_specs(*, with_memory: bool = True) -> list[ToolSpec]:
    """The catalogue in the form the model is given."""
    specs: list[ToolSpec] = []
    for spec in CATALOGUE:
        if spec.name == REMEMBER and not with_memory:
            continue
        properties: dict[str, Any] = {}
        for name, param in spec.params.items():
            properties[name] = {"type": "string", "description": param.description}
            if param.choices:
                properties[name]["enum"] = list(param.choices)
        specs.append(
            ToolSpec(
                spec.name,
                spec.description,
                {"type": "object", "properties": properties, "required": list(spec.params)},
            )
        )
    return specs


def validate_call(call: ToolCall) -> Action | None:
    """The action a model's request stands for, or None if it is not exactly a known one."""
    spec = _BY_NAME.get(call.name)
    if spec is None:
        return None
    args: dict[str, str] = {}
    for name, param in spec.params.items():
        value = call.arguments.get(name)
        if not isinstance(value, str):
            return None
        text = " ".join(value.split())
        if param.choices:
            text = text.lower()
            if text not in param.choices:
                return None
        else:
            text = text[:MAX_TEXT_ARG_CHARS].strip()
            if not text:
                return None
        args[name] = text
    return Action(spec.name, args)  # arguments the catalogue does not list are dropped


def validate_calls(calls: Sequence[ToolCall]) -> list[Action]:
    """The valid, distinct actions among `calls`, in order, at most MAX_ACTIONS_PER_REPLY."""
    actions: list[Action] = []
    for call in calls:
        action = validate_call(call)
        if action is not None and action not in actions:
            actions.append(action)
        if len(actions) == MAX_ACTIONS_PER_REPLY:
            break
    return actions


def _describe(action: Action) -> str:
    """What to say when the model asked for an action and said nothing itself."""
    args = action.args
    match action.name:
        case "open_page":
            return f"Opening the {args['page']} page"
        case "set_theme":
            return f"Switching to the {_THEME_NAMES[args['theme']]} theme"
        case "set_option":
            return f"Turning {_OPTION_NAMES[args['option']]} {args['value']}"
        case "ask_notes":
            return "Searching your notes"
        case "update_library":
            return "Updating the library"
        case "report_status":
            return "Checking the systems"
        case "clear_conversation":
            return "Starting a new conversation"
        case "lock_app":
            return "Locking the application"
        case "remember":
            return "I will remember that"
        case _:
            return "Done"


def confirmation(actions: Sequence[Action], address: str) -> str:
    """One short spoken sentence for `actions`, such as "Opening the settings page, sir." """
    said = ". ".join(_describe(action) for action in actions)
    return f"{said}, {address}." if address else f"{said}."
