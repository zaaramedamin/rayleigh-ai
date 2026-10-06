from typing import Any

import pytest

from app.ai.llm.base import ToolCall
from app.assistant.actions import (
    CATALOGUE,
    MAX_ACTIONS_PER_REPLY,
    MAX_TEXT_ARG_CHARS,
    Action,
    confirmation,
    tool_specs,
    validate_call,
    validate_calls,
)


def test_the_model_is_offered_exactly_the_catalogue() -> None:
    specs = tool_specs()

    assert [spec.name for spec in specs] == [spec.name for spec in CATALOGUE]
    page = next(spec for spec in specs if spec.name == "open_page").parameters
    assert page["required"] == ["page"]
    assert "settings" in page["properties"]["page"]["enum"]
    assert all(spec.parameters["type"] == "object" for spec in specs)


def test_remembering_is_not_offered_when_memory_is_off() -> None:
    assert "remember" in {spec.name for spec in tool_specs()}
    assert "remember" not in {spec.name for spec in tool_specs(with_memory=False)}


def test_nothing_destructive_or_outside_the_application_is_in_the_catalogue() -> None:
    names = " ".join(spec.name for spec in CATALOGUE)

    for word in ("delete", "remove", "password", "file", "run", "shell", "http", "forget"):
        assert word not in names


def test_a_valid_request_becomes_an_action() -> None:
    assert validate_call(ToolCall("open_page", {"page": "settings"})) == Action(
        "open_page", {"page": "settings"}
    )
    assert validate_call(ToolCall("lock_app", {})) == Action("lock_app", {})
    # Case and stray spaces in a choice are forgiven; unknown arguments are dropped.
    assert validate_call(
        ToolCall("set_option", {"option": " Notes_Mode ", "value": "ON", "extra": "x"})
    ) == Action("set_option", {"option": "notes_mode", "value": "on"})


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("delete_everything", {}),
        ToolCall("open_page", {}),
        ToolCall("open_page", {"page": "C:\\Windows"}),
        ToolCall("open_page", {"page": ["settings"]}),
        ToolCall("open_page", {"page": None}),
        ToolCall("set_theme", {"theme": "pink"}),
        ToolCall("set_option", {"option": "interface_sounds"}),
        ToolCall("set_option", {"option": "demo", "value": "on"}),
        ToolCall("set_option", {"option": "animations", "value": "maybe"}),
        ToolCall("ask_notes", {"question": "   "}),
        ToolCall("remember", {"fact": 42}),
    ],
)
def test_anything_that_is_not_exactly_a_known_action_is_dropped(call: ToolCall) -> None:
    assert validate_call(call) is None


def test_free_text_is_flattened_and_bounded() -> None:
    action = validate_call(ToolCall("remember", {"fact": "likes\n\ntea\t" + "x" * 1000}))

    assert action is not None
    assert action.args["fact"].startswith("likes tea x")
    assert len(action.args["fact"]) == MAX_TEXT_ARG_CHARS


def test_only_a_few_distinct_valid_actions_are_kept_in_order() -> None:
    calls: list[ToolCall] = [
        ToolCall("open_page", {"page": "chat"}),
        ToolCall("nonsense", {}),
        ToolCall("open_page", {"page": "chat"}),  # repeated
        ToolCall("set_theme", {"theme": "mark3"}),
        ToolCall("lock_app", {}),
        ToolCall("update_library", {}),
    ]

    actions = validate_calls(calls)

    assert [a.name for a in actions] == ["open_page", "set_theme", "lock_app"]
    assert len(actions) == MAX_ACTIONS_PER_REPLY


@pytest.mark.parametrize(
    ("name", "args", "said"),
    [
        ("open_page", {"page": "settings"}, "Opening the settings page, sir."),
        ("set_theme", {"theme": "mark3"}, "Switching to the Mark Three theme, sir."),
        (
            "set_option",
            {"option": "spoken_replies", "value": "off"},
            "Turning spoken replies off, sir.",
        ),
        ("ask_notes", {"question": "When is my flight?"}, "Searching your notes, sir."),
        ("remember", {"fact": "likes tea"}, "I will remember that, sir."),
        ("lock_app", {}, "Locking the application, sir."),
    ],
)
def test_every_action_has_something_to_say(name: str, args: dict[str, Any], said: str) -> None:
    assert confirmation([Action(name, args)], "sir") == said


def test_the_confirmation_covers_several_actions_and_no_form_of_address() -> None:
    actions = [Action("open_page", {"page": "chat"}), Action("clear_conversation", {})]

    assert confirmation(actions, "") == "Opening the chat page. Starting a new conversation."
    assert all(confirmation([Action(spec.name, _args(spec.name))], "") for spec in CATALOGUE)


def _args(name: str) -> dict[str, str]:
    spec = next(spec for spec in CATALOGUE if spec.name == name)
    return {
        key: (param.choices[0] if param.choices else "text") for key, param in spec.params.items()
    }
