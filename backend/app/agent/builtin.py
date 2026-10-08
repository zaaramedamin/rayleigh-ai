"""The first tools: they only read, so once the owner switches them on they run without asking.

- calculator: arithmetic, read by a small parser that understands numbers and a few operators and
  functions and nothing else (never `eval`), with limits so no expression can use up the computer.
- current_time: the date and time on this computer.
- search_notes: the library's own search, as a tool. What it brings back is the owner's notes, so
  like every tool result it is treated as untrusted data by whatever reads it.

Anything that touches the outside world is passed in (the clock, the search), so tests never depend
on the real ones.
"""

import ast
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from app.agent.tools import Param, Tool, ToolError
from app.knowledge.retrieval.service import RetrievedChunk

# --- calculator --------------------------------------------------------------------------------

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 1000
MAX_NODES = 60
MAX_INTEGER_DIGITS = 300

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class _Evaluator:
    def __init__(self) -> None:
        self.nodes = 0

    def value(self, node: ast.AST) -> int | float:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ToolError("That expression is too long to calculate.")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ToolError("Only numbers can be calculated.")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ToolError(f"I do not know {node.id!r}. Only pi and e are known names.")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self.value(node.operand))  # type: ignore[no-any-return]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError(f"Exponents above {MAX_EXPONENT} are not calculated.")
            return _BINARY[type(node.op)](left, right)  # type: ignore[no-any-return]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            function = _FUNCTIONS.get(node.func.id)
            if function is None:
                raise ToolError(f"I do not know the function {node.func.id!r}.")
            if not 1 <= len(node.args) <= 4:
                raise ToolError(f"{node.func.id} takes between one and four numbers.")
            return function(*(self.value(arg) for arg in node.args))  # type: ignore[no-any-return]
        raise ToolError(
            "I can only do arithmetic: + - * / // % ** and ( ), with sqrt, abs, round, min, max, "
            "floor and ceil."
        )


def calculate(expression: str) -> str:
    """The value of an arithmetic expression, as text. Raises ToolError for anything else."""
    text = expression.strip()
    if "^" in text:
        raise ToolError("Write powers with ** (for example 2**10): ^ means something else here.")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ToolError("That expression is too long to calculate.")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ToolError("I cannot read that as an arithmetic expression.") from exc
    try:
        result = _Evaluator().value(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("That divides by zero.") from exc
    except (OverflowError, ValueError, TypeError, RecursionError) as exc:
        raise ToolError(
            "That cannot be calculated (the number is too large or not defined)."
        ) from exc
    return _format_number(result)


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        if len(str(abs(value))) > MAX_INTEGER_DIGITS:
            raise ToolError("The result is too large to show.")
        return str(value)
    if value != value or value in (math.inf, -math.inf):
        raise ToolError("That cannot be calculated (the number is too large or not defined).")
    return f"{value:.12g}"


def calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description=(
            "Calculate an arithmetic expression exactly, for example 12.5*(3+4) or sqrt(2)/2 or "
            "2**10. Use it for any sum, percentage or conversion instead of working it out "
            "yourself."
        ),
        level="read_local",
        params={
            "expression": Param(
                "string", "The expression to calculate.", max_length=MAX_EXPRESSION_CHARS
            )
        },
        run=lambda args: calculate(args["expression"]),
        describe=lambda args: f"Calculate {args['expression']}",
    )


# --- current time ------------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now().astimezone()


def current_time_tool(now: Callable[[], datetime] = _now) -> Tool:
    def run(_args: Mapping[str, Any]) -> str:
        moment = now()
        offset = moment.strftime("%z")
        zone = f"UTC{offset[:3]}:{offset[3:]}" if offset else "local time"
        return f"{moment.strftime('%A %d %B %Y, %H:%M')} ({zone})"

    return Tool(
        name="current_time",
        description="The current date and time on this computer.",
        level="read_local",
        params={},
        run=run,
        describe=lambda _args: "Read the current date and time",
    )


# --- searching the notes -----------------------------------------------------------------------

MAX_NOTES = 8
NOTE_CHARS = 500

Searcher = Callable[[str, int], Sequence[RetrievedChunk]]


def _label(note: RetrievedChunk) -> str:
    where = f"{note.source} > {note.heading_path}" if note.heading_path else note.source
    if note.start_page is not None:
        pages = f"page {note.start_page}"
        if note.end_page and note.end_page != note.start_page:
            pages = f"pages {note.start_page}-{note.end_page}"
        return f"{where}, {pages}"
    return f"{where}, lines {note.start_line}-{note.end_line}"


def search_notes_tool(search: Searcher, min_score: float) -> Tool:
    def run(args: Mapping[str, Any]) -> str:
        count = int(args.get("top_k", 4))
        found = [n for n in search(args["query"], count) if n.score >= min_score][:count]
        if not found:
            return "No note in the library matches that."
        blocks = [
            f"[{number}] {_label(note)}\n{note.text.strip()[:NOTE_CHARS]}"
            for number, note in enumerate(found, start=1)
        ]
        return "\n\n".join(blocks)

    return Tool(
        name="search_notes",
        description=(
            "Search the owner's saved notes and documents and return the passages that match, with "
            "where each one comes from. Use it when you need what the owner's own files say."
        ),
        level="read_local",
        params={
            "query": Param(
                "string", "What to look for, as a short phrase or question.", max_length=300
            ),
            "top_k": Param(
                "integer",
                "How many passages to return.",
                required=False,
                minimum=1,
                maximum=MAX_NOTES,
            ),
        },
        run=run,
        describe=lambda args: f"Search your notes for {args['query']!r}",
        max_result_chars=MAX_NOTES * (NOTE_CHARS + 200),
    )
