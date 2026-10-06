"""The assistant's long-term memory: short facts kept from one conversation to the next.

A memory is saved in two ways only: the owner types it on the Profile page, or the assistant
saves it during a conversation with its `remember` action (the interface says so each time).
Every memory is listed on the Profile page and can be deleted there. Memories are stored in the
library database, encrypted when the library is, and are never logged.
"""

import re
from collections.abc import Sequence
from typing import Literal

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.storage.models import Memory

Origin = Literal["owner", "assistant"]

MAX_MEMORIES = 200
MAX_MEMORY_CHARS = 400
# Characters of memories placed in one prompt. The newest are kept when there are more.
MAX_PROMPT_CHARS = 2000

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class MemoryFullError(ValueError):
    """The memory holds as many entries as it may; something must be deleted first."""


def clean_memory(text: str) -> str:
    """One plain line: no control characters, no line breaks, bounded length."""
    return " ".join(_CONTROL.sub(" ", text).split())[:MAX_MEMORY_CHARS].strip()


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def list_memories(session: Session) -> list[Memory]:
    """Every memory, oldest first."""
    return list(session.scalars(select(Memory).order_by(Memory.id)))


def add_memory(session: Session, text: str, origin: Origin) -> tuple[Memory, bool]:
    """Save a memory. Returns (memory, created); `created` is False if it was already known.

    Raises ValueError for an empty text and MemoryFullError when the memory is full.
    """
    cleaned = clean_memory(text)
    if not cleaned:
        raise ValueError("there is nothing to remember")
    # Compared here, not in SQL: an encrypted column never matches an equality filter.
    wanted = _key(cleaned)
    existing = list_memories(session)
    for memory in existing:
        if _key(memory.text) == wanted:
            return memory, False
    if len(existing) >= MAX_MEMORIES:
        raise MemoryFullError(
            f"the memory is full ({MAX_MEMORIES} entries); delete some on the Profile page"
        )
    memory = Memory(text=cleaned, origin=origin)
    session.add(memory)
    session.commit()
    return memory, True


def delete_memory(session: Session, memory_id: int) -> bool:
    memory = session.get(Memory, memory_id)
    if memory is None:
        return False
    session.delete(memory)
    session.commit()
    return True


def clear_memories(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(Memory)) or 0
    session.execute(delete(Memory))
    session.commit()
    return count


def memories_for_prompt(texts: Sequence[str], budget: int = MAX_PROMPT_CHARS) -> list[str]:
    """The newest memories that fit in `budget` characters, oldest first."""
    kept: list[str] = []
    used = 0
    for text in reversed(texts):
        if used + len(text) > budget:
            break
        kept.append(text)
        used += len(text)
    kept.reverse()
    return kept
