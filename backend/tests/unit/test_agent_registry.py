"""The default set of agent tools."""

from app.agent.registry import default_registry


def test_the_default_tools_include_the_search_when_there_is_one() -> None:
    registry = default_registry(search=lambda _query, _count: [], min_score=0.3)

    assert registry.names() == ["calculator", "current_time", "search_notes"]


def test_without_a_search_the_agent_has_no_notes_to_look_in() -> None:
    registry = default_registry(search=None, min_score=0.3)

    assert registry.names() == ["calculator", "current_time"]
    assert registry.get("search_notes") is None


def test_every_default_tool_only_reads() -> None:
    registry = default_registry(search=lambda _query, _count: [], min_score=0.3)

    assert {registry.get(name).level for name in registry.names()} == {"read_local"}  # type: ignore[union-attr]
