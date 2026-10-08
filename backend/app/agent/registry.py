"""The tools the agent has, put together in one place for the command line and the API.

What a tool needs from the outside world (the search, later the places it may open) is passed in, so
the same list is built the same way wherever the agent runs, and tests can build it without any
real search or computer.
"""

from app.agent.builtin import Searcher, calculator_tool, current_time_tool, search_notes_tool
from app.agent.tools import ToolRegistry


def default_registry(*, search: Searcher | None, min_score: float) -> ToolRegistry:
    """Every tool the agent has. Without a `search`, the agent has no notes to look in."""
    registry = ToolRegistry([calculator_tool(), current_time_tool()])
    if search is not None:
        registry.add(search_notes_tool(search, min_score))
    return registry
