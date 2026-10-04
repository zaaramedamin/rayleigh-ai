"""General conversation with the local model: no notes, no tools, nothing leaves this machine.

This is the assistant's "talk normally" mode. The model answers from what it already knows, so
a reply here is not backed by the owner's notes and carries no citations; the interface labels
it that way. Answering from the notes is a separate path (app.knowledge.answering) with its own
safeguards, and nothing here reads the library.
"""

import logging
from collections.abc import Sequence
from datetime import date

from app.ai.llm.base import ChatMessage, ChatReply, LLMProvider

logger = logging.getLogger(__name__)

# Longer than a search question: this mode is also for "rewrite this" and "explain this".
MAX_MESSAGE_CHARS = 8000
# Earlier conversation sent along with a new message. Enough to follow a thread, and with the
# longest message it still leaves the model's context window room for the reply.
MAX_HISTORY_CHARS = 8000
# Answers from notes use 0 so they are repeatable. Conversation reads better with some variety.
TEMPERATURE = 0.6

SYSTEM_PROMPT = """You are Reyleight, a private assistant running entirely on the user's own \
computer.

Rules:
- Answer from your own general knowledge, helpfully and honestly. If you are not sure about \
something, say so instead of guessing.
- In this mode you cannot see the user's notes or files, and you have no internet access and \
no tools. If the user asks about their own notes, documents or personal details that they have \
not told you in this conversation, say that you cannot see their notes in this mode and that \
switching on "My notes" makes you search them. Never invent what their notes contain.
- Be concise unless the user asks for more detail. Write plain text, without Markdown markup \
such as ** or #.
- Reply in the language the user writes in.
- Today's date is {today}."""


def system_prompt(today: date) -> str:
    return SYSTEM_PROMPT.format(today=f"{today:%A} {today.day} {today:%B %Y}")


def select_history(
    history: Sequence[ChatMessage], budget: int = MAX_HISTORY_CHARS
) -> list[ChatMessage]:
    """The most recent turns that fit in `budget` characters, oldest first.

    The result always starts with something the user said: a reply whose question no longer
    fits is dropped with it.
    """
    kept: list[ChatMessage] = []
    used = 0
    for message in reversed(history):
        content = message.content.strip()
        if not content:
            continue
        if used + len(content) > budget:
            break
        kept.append(ChatMessage(message.role, content))
        used += len(content)
    kept.reverse()
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def build_messages(message: str, history: Sequence[ChatMessage] = ()) -> list[ChatMessage]:
    """The conversation to send: recent history, then the new message. Raises ValueError."""
    text = message.strip()
    if not text:
        raise ValueError("the message is empty")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"the message is longer than {MAX_MESSAGE_CHARS} characters")
    return [*select_history(history), ChatMessage("user", text)]


def reply_to(
    llm: LLMProvider,
    message: str,
    history: Sequence[ChatMessage] = (),
    *,
    today: date | None = None,
) -> ChatReply:
    """Answer `message` in the context of `history`. Raises ValueError for a bad message."""
    messages = build_messages(message, history)
    reply = llm.chat(system_prompt(today or date.today()), messages, temperature=TEMPERATURE)
    # Sizes only: what was said is never logged.
    logger.info("chat finished turns=%d truncated=%s", len(messages), reply.truncated)
    return reply
