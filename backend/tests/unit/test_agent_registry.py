"""The default set of agent tools."""

from pathlib import Path

from app.agent.registry import default_registry


def build(**kwargs: object):  # type: ignore[no-untyped-def]
    return default_registry(search=lambda _query, _count: [], min_score=0.3, **kwargs)  # type: ignore[arg-type]


def test_the_default_tools_include_the_search_when_there_is_one() -> None:
    assert build().names() == [
        "calculator",
        "current_time",
        "search_notes",
        "open_path",
        "open_app",
    ]


def test_without_a_search_the_agent_has_no_notes_to_look_in() -> None:
    registry = default_registry(search=None, min_score=0.3)

    assert registry.names() == ["calculator", "current_time", "open_path", "open_app"]
    assert registry.get("search_notes") is None


def test_the_reading_tools_only_read_and_the_opening_tools_always_ask() -> None:
    registry = build()

    levels = {name: registry.get(name).level for name in registry.names()}  # type: ignore[union-attr]
    assert levels == {
        "calculator": "read_local",
        "current_time": "read_local",
        "search_notes": "read_local",
        "open_path": "open_local",
        "open_app": "open_local",
    }


def test_what_the_tools_open_goes_through_what_the_caller_passed_in(tmp_path: Path) -> None:
    opened: list[Path] = []
    started: list[str] = []
    note = tmp_path / "plan.txt"
    note.write_text("x")
    registry = build(opener=opened.append, launcher=started.append)

    registry.get("open_path").run({"path": str(note)})  # type: ignore[union-attr]
    registry.get("open_app").run({"app": "paint"})  # type: ignore[union-attr]

    assert opened == [note.resolve()] and started == ["mspaint.exe"]
