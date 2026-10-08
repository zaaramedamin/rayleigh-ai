"""The tools the agent has, put together in one place for the command line and the API.

What a tool needs from the outside world (the search, the way things are opened) is passed in, so
the same list is built the same way wherever the agent runs, and tests can build it without any
real search or computer.
"""

from app.agent.builtin import Searcher, calculator_tool, current_time_tool, search_notes_tool
from app.agent.open_tools import Launcher, Opener, open_app_tool, open_path_tool
from app.agent.tools import ToolRegistry


def default_registry(
    *,
    search: Searcher | None,
    min_score: float,
    opener: Opener | None = None,
    launcher: Launcher | None = None,
) -> ToolRegistry:
    """Every tool the agent has. Without a `search`, the agent has no notes to look in. Without an
    `opener` or `launcher` the tools that open things use the real ones."""
    registry = ToolRegistry([calculator_tool(), current_time_tool()])
    if search is not None:
        registry.add(search_notes_tool(search, min_score))
    registry.add(open_path_tool(opener) if opener else open_path_tool())
    registry.add(open_app_tool(launcher) if launcher else open_app_tool())
    return registry
