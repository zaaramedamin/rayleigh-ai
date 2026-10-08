"""Reading what the owner means, before the agent does anything.

People type fast: typos, missing words, two languages in one line, "open it" with nothing to point
at. A small model that is handed that text and a set of tools tends to do one of two things: guess
(and the guess becomes an action for the owner to refuse), or answer in the wrong language. So the
first thing the agent does is read the message on its own, with no tools, and write down four
things:

- the language the owner wrote in (so the reply is in it);
- what they most likely mean, as one clear sentence with the typos fixed (shown to them, so they
  can see how they were understood);
- whether anything essential is missing, and
- if so, the one question to ask, so the agent asks it instead of choosing for them.

The reading is a help and never a gate on safety. If the model's reply cannot be read, or the model
cannot be reached, the task simply goes on with what the owner typed. The result is only text that
is shown and passed on as part of the first message: it can never grant a permission or run
anything, and the owner's message is treated as data to interpret, not as instructions to this
step.
"""

import json
import logging
import re
from dataclasses import dataclass

from app.ai.llm.base import LLMError, LLMProvider
from app.knowledge.answering.prompts import INTERPRET

logger = logging.getLogger(__name__)

MAX_REQUEST_CHARS = 600
MAX_QUESTION_CHARS = 300
_LANGUAGE = re.compile(r"^[A-Za-z][A-Za-z \-()]{1,29}$")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Understanding:
    language: str  # as the model named it, in English: "French"
    request: str  # one clear sentence, in the owner's language
    clear: bool  # False when something essential is missing
    question: str  # what to ask when it is not clear; empty when it is


def _one_line(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = _WHITESPACE.sub(" ", value).strip()
    return text[:limit].rstrip() if text else None


def parse_understanding(reply: str) -> Understanding | None:
    """The reading in a model's reply, or None if it is not exactly what was asked for."""
    text = _FENCE.sub("", reply.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    language = _one_line(data.get("language"), 100)  # too long is refused, not cut
    request = _one_line(data.get("request"), MAX_REQUEST_CHARS)
    clear = data.get("clear")
    if isinstance(clear, str) and clear.strip().lower() in ("true", "false"):
        clear = clear.strip().lower() == "true"
    if language is None or not _LANGUAGE.match(language) or request is None:
        return None
    if not isinstance(clear, bool):
        return None
    question = _one_line(data.get("question", ""), MAX_QUESTION_CHARS) or ""
    # Not clear but nothing to ask: there is no question to put to the owner, so go on.
    return Understanding(language, request, clear or not question, question if not clear else "")


def understand(llm: LLMProvider, goal: str, context: str = "") -> Understanding | None:
    """What `goal` means, or None when the model's reading cannot be used. Never raises for a model
    error: the task goes on with what the owner typed."""
    facts = context.strip()
    prompt = (
        f"{facts}\n\nThe person's request:\n{goal}" if facts else f"The person's request:\n{goal}"
    )
    try:
        reply = llm.generate(INTERPRET.system, prompt)
    except LLMError:
        logger.info("request not interpreted: the model could not be reached")
        return None
    reading = parse_understanding(reply)
    if reading is None:
        logger.info("request not interpreted: the model's reply could not be read")
        return None
    logger.info("request interpreted clear=%s", reading.clear)
    return reading
