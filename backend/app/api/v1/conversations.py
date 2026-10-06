import json
import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import SessionDep, SettingsDep
from app.assistant.conversations import (
    MAX_CONTENT_CHARS,
    MAX_CONVERSATIONS,
    MAX_MESSAGES,
    MAX_TITLE_CHARS,
    ConversationFullError,
    ConversationSummary,
    NewMessage,
    TooManyConversationsError,
    append_messages,
    count_conversations,
    create_conversation,
    delete_all_conversations,
    delete_conversation,
    get_conversation,
    list_conversations,
    messages_of,
    purge_older_than,
    rename_conversation,
    summary_of,
)
from app.storage.models import Conversation, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])

# Turns that may be added in one request: a question and its answer, with some room to spare.
MAX_MESSAGES_PER_REQUEST = 10


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationList(BaseModel):
    conversations: list[ConversationOut]
    total: int = Field(description="All stored conversations, before paging.")
    limit: int = Field(description="The most conversations that can be kept.")
    retention_days: int = Field(
        description="Conversations untouched for this many days are deleted. 0: kept until deleted."
    )


class MessageOut(BaseModel):
    id: int
    position: int
    role: Literal["user", "assistant"]
    mode: Literal["general", "notes"]
    content: str
    payload: dict[str, Any] | None = Field(
        description="What the interface shows beside an answer (sources and so on), as saved."
    )
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
    message_limit: int = Field(description="The most messages one conversation can hold.")


class NewConversation(BaseModel):
    title: str = Field(default="", max_length=MAX_TITLE_CHARS)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)


class MessageIn(BaseModel):
    role: Literal["user", "assistant"]
    mode: Literal["general", "notes"]
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    payload: dict[str, Any] | None = None


class NewMessages(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=MAX_MESSAGES_PER_REQUEST)


class ClearedConversations(BaseModel):
    deleted: int


def _out(summary: ConversationSummary) -> ConversationOut:
    return ConversationOut(
        id=summary.id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        message_count=summary.message_count,
    )


def _message_out(message: Message) -> MessageOut:
    payload: dict[str, Any] | None = None
    if message.payload:
        try:
            decoded = json.loads(message.payload)
        except ValueError:
            decoded = None
        payload = decoded if isinstance(decoded, dict) else None
    return MessageOut(
        id=message.id,
        position=message.position,
        role="assistant" if message.role == "assistant" else "user",
        mode="notes" if message.mode == "notes" else "general",
        content=message.content,
        payload=payload,
        created_at=message.created_at,
    )


def _existing(session: Session, conversation_id: int) -> Conversation:
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That conversation does not exist.")
    return conversation


@router.get("")
def get_conversations(
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=100, ge=1, le=MAX_CONVERSATIONS),
    offset: int = Query(default=0, ge=0),
) -> ConversationList:
    """The stored conversations, most recently used first."""
    # Expired conversations are removed when the list is read, so no background job is needed.
    purged = purge_older_than(session, settings.conversation_retention_days)
    if purged:
        logger.info("conversations past their retention period deleted count=%d", purged)
    return ConversationList(
        conversations=[_out(s) for s in list_conversations(session, limit=limit, offset=offset)],
        total=count_conversations(session),
        limit=MAX_CONVERSATIONS,
        retention_days=settings.conversation_retention_days,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def post_conversation(body: NewConversation, session: SessionDep) -> ConversationOut:
    """Start a new, empty conversation. Its title comes from the first message when left empty."""
    try:
        conversation = create_conversation(session, body.title)
    except TooManyConversationsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _out(summary_of(session, conversation))


@router.get("/{conversation_id}")
def get_one_conversation(conversation_id: int, session: SessionDep) -> ConversationDetail:
    conversation = _existing(session, conversation_id)
    summary = summary_of(session, conversation)
    return ConversationDetail(
        **_out(summary).model_dump(),
        messages=[_message_out(m) for m in messages_of(session, conversation_id)],
        message_limit=MAX_MESSAGES,
    )


@router.put("/{conversation_id}")
def put_conversation(conversation_id: int, body: Rename, session: SessionDep) -> ConversationOut:
    """Rename a conversation."""
    conversation = _existing(session, conversation_id)
    try:
        rename_conversation(session, conversation, body.title)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _out(summary_of(session, conversation))


@router.post("/{conversation_id}/messages", status_code=status.HTTP_201_CREATED)
def post_messages(conversation_id: int, body: NewMessages, session: SessionDep) -> ConversationOut:
    """Add turns (a question and its answer) to the end of a conversation."""
    conversation = _existing(session, conversation_id)
    try:
        append_messages(
            session,
            conversation,
            [NewMessage(m.role, m.mode, m.content, m.payload) for m in body.messages],
        )
    except ConversationFullError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _out(summary_of(session, conversation))


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_conversation(conversation_id: int, session: SessionDep) -> None:
    """Delete a conversation and every message in it, for good."""
    if not delete_conversation(session, conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That conversation does not exist.")


@router.delete("")
def remove_all_conversations(session: SessionDep) -> ClearedConversations:
    """Delete every stored conversation, for good."""
    return ClearedConversations(deleted=delete_all_conversations(session))
