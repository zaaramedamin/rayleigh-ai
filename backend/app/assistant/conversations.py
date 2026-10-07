"""Stored conversations: reopen a conversation later, rename it, delete it.

Everything in a conversation is made of the owner's own words (and of answers built from their
notes), so it is treated like note text: it lives in the library database, in encrypted columns when
the library is encrypted, and it is never logged. Deleting a conversation deletes its messages
with it, and the database overwrites deleted rows (`secure_delete`). Removing a *document* from the
library also removes what a conversation kept of it: an answer that quoted that document is replaced
by a short notice, so a removed note does not live on inside a conversation.

An optional retention period (`CONVERSATION_RETENTION_DAYS`) deletes conversations that were not
touched for that long. By default nothing is deleted automatically.
"""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import delete, func, inspect, select
from sqlalchemy.orm import Session

from app.storage.models import Conversation, Message

Role = Literal["user", "assistant"]
Mode = Literal["general", "notes"]

MAX_CONVERSATIONS = 500
MAX_MESSAGES = 400  # in one conversation
MAX_TITLE_CHARS = 200
MAX_CONTENT_CHARS = 16000
MAX_PAYLOAD_CHARS = 120_000
TITLE_FROM_MESSAGE_CHARS = 60

REMOVED_NOTICE = (
    "This answer used a note that was removed from the library, so it was removed here too."
)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class ConversationFullError(ValueError):
    """The conversation holds as many messages as it may. Start a new one."""


class TooManyConversationsError(ValueError):
    """As many conversations are kept as may be. Delete some first."""


@dataclass(frozen=True)
class NewMessage:
    role: Role
    mode: Mode
    content: str
    # What the interface shows beside an answer (sources and so on). Plain JSON data.
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConversationSummary:
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


def clean_title(text: str) -> str:
    """One plain line of bounded length."""
    return " ".join(_CONTROL.sub(" ", text).split())[:MAX_TITLE_CHARS].strip()


def title_from(message: str) -> str:
    """A short title made from the first thing the owner said."""
    line = clean_title(message)
    if len(line) <= TITLE_FROM_MESSAGE_CHARS:
        return line
    return line[: TITLE_FROM_MESSAGE_CHARS - 3].rstrip() + "..."


def _now() -> datetime:
    return datetime.now(UTC)


def _payload_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("the details of a message must be plain JSON data") from exc
    if len(text) > MAX_PAYLOAD_CHARS:
        raise ValueError("the details of a message are too large")
    return text


def cited_documents(payload: dict[str, Any] | None) -> str:
    """ ",12,40," for the sources named in a payload, or an empty string."""
    sources = payload.get("sources") if payload else None
    if not isinstance(sources, list):
        return ""
    ids = {
        source["document_id"]
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("document_id"), int)
        and not isinstance(source.get("document_id"), bool)
    }
    return f",{','.join(str(i) for i in sorted(ids))}," if ids else ""


def count_conversations(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Conversation)) or 0


def create_conversation(session: Session, title: str = "") -> Conversation:
    """Start a conversation. Raises TooManyConversationsError when the limit is reached."""
    if count_conversations(session) >= MAX_CONVERSATIONS:
        raise TooManyConversationsError(
            f"{MAX_CONVERSATIONS} conversations are kept; delete some to start a new one"
        )
    conversation = Conversation(title=clean_title(title))
    session.add(conversation)
    session.commit()
    return conversation


def list_conversations(
    session: Session, *, limit: int = 100, offset: int = 0
) -> list[ConversationSummary]:
    """The conversations, the most recently used first."""
    counts = (
        select(Message.conversation_id, func.count().label("n"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = session.execute(
        select(Conversation, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [ConversationSummary(c.id, c.title, c.created_at, c.updated_at, int(n)) for c, n in rows]


def summary_of(session: Session, conversation: Conversation) -> ConversationSummary:
    count = session.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    )
    return ConversationSummary(
        conversation.id,
        conversation.title,
        conversation.created_at,
        conversation.updated_at,
        int(count or 0),
    )


def get_conversation(session: Session, conversation_id: int) -> Conversation | None:
    return session.get(Conversation, conversation_id)


def messages_of(session: Session, conversation_id: int) -> list[Message]:
    return list(
        session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.position)
        )
    )


def append_messages(
    session: Session, conversation: Conversation, messages: Sequence[NewMessage]
) -> list[Message]:
    """Add turns to the end of a conversation.

    Raises ValueError for a message that is empty or too large, and ConversationFullError when the
    conversation would grow past its limit. Nothing is saved if any message is refused.
    """
    prepared: list[tuple[NewMessage, str, str | None, str]] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            raise ValueError("a message is empty")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"a message is longer than {MAX_CONTENT_CHARS} characters")
        prepared.append(
            (message, content, _payload_json(message.payload), cited_documents(message.payload))
        )

    last = session.scalar(
        select(func.max(Message.position)).where(Message.conversation_id == conversation.id)
    )
    next_position = 0 if last is None else last + 1
    if next_position + len(prepared) > MAX_MESSAGES:
        raise ConversationFullError(
            f"this conversation holds {MAX_MESSAGES} messages, the most it can; start a new one"
        )

    created = [
        Message(
            conversation_id=conversation.id,
            position=next_position + offset,
            role=message.role,
            mode=message.mode,
            content=content,
            payload=payload,
            cited_documents=cited,
        )
        for offset, (message, content, payload, cited) in enumerate(prepared)
    ]
    session.add_all(created)
    if not conversation.title:
        first_user = next((m.content for m in messages if m.role == "user"), "")
        conversation.title = title_from(first_user)
    conversation.updated_at = _now()
    session.commit()
    return created


def rename_conversation(session: Session, conversation: Conversation, title: str) -> None:
    """Give a conversation a new title. Raises ValueError if it would be empty."""
    cleaned = clean_title(title)
    if not cleaned:
        raise ValueError("a conversation needs a title")
    conversation.title = cleaned
    session.commit()


def _delete_conversations(session: Session, ids: Iterable[int]) -> int:
    chosen = list(ids)
    if not chosen:
        return 0
    # The messages go explicitly as well as through the foreign key, so nothing depends on it.
    for start in range(0, len(chosen), 500):
        batch = chosen[start : start + 500]
        session.execute(delete(Message).where(Message.conversation_id.in_(batch)))
        session.execute(delete(Conversation).where(Conversation.id.in_(batch)))
    session.commit()
    session.expire_all()  # objects of the deleted rows must not be used again
    return len(chosen)


def delete_conversation(session: Session, conversation_id: int) -> bool:
    """Delete a conversation and every message in it. False if there is no such conversation."""
    if session.get(Conversation, conversation_id) is None:
        return False
    _delete_conversations(session, [conversation_id])
    return True


def delete_all_conversations(session: Session) -> int:
    return _delete_conversations(session, session.scalars(select(Conversation.id)).all())


def purge_older_than(session: Session, days: int) -> int:
    """Delete conversations not touched for `days` days. Zero or less keeps everything."""
    if days <= 0:
        return 0
    cutoff = _now() - timedelta(days=days)
    expired = session.scalars(select(Conversation.id).where(Conversation.updated_at < cutoff)).all()
    return _delete_conversations(session, expired)


def forget_document_passages(session: Session, document_ids: Iterable[int]) -> int:
    """Replace the answers that quoted these documents by a notice. The caller commits.

    The words of an answer can repeat a note, so removing a note from the library must remove them
    here too. Messages are found through `cited_documents`, a plain column of document numbers.
    """
    if not inspect(session.connection()).has_table("messages"):
        return 0  # a library not upgraded yet has no conversations
    scrubbed = 0
    for document_id in set(document_ids):
        found = session.scalars(
            select(Message).where(Message.cited_documents.like(f"%,{int(document_id)},%"))
        ).all()
        for message in found:
            message.content = REMOVED_NOTICE
            message.payload = json.dumps({"removed": True})
            message.cited_documents = ""
            scrubbed += 1
    return scrubbed
