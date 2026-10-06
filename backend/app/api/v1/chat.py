import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.llm.base import (
    ChatMessage,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.api.deps import LLMDep, OptionalSessionDep, SettingsDep
from app.assistant.chat import MAX_MESSAGE_CHARS, AssistantContext, Remember, reply_to
from app.assistant.identity import load_identity
from app.assistant.memory import add_memory, list_memories
from app.core.config import Settings
from app.knowledge.profile.service import FIELDS, load_profile
from app.security.errors import SecurityError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Bounds on what a client may send. The service then keeps only the recent turns that fit.
MAX_HISTORY_TURNS = 40
MAX_TURN_CHARS = 16000


class ChatTurn(BaseModel):
    # Only what was said. Instructions to the model ("system") are never taken from a request.
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Earlier turns of this conversation, oldest first.",
    )


class ChatAction(BaseModel):
    name: str
    args: dict[str, str]


class ChatResponse(BaseModel):
    answer: str
    model: str = Field(description="The local model that wrote the reply.")
    truncated: bool = Field(description="True when the reply was cut off at the length limit.")
    actions: list[ChatAction] = Field(
        default_factory=list,
        description="What the assistant asks the interface to do, already validated. "
        '"remember" is listed for information: the server has saved it.',
    )


def _profile_facts(session: Session, settings: Settings) -> list[tuple[str, str]]:
    """What the Profile page says, as (heading, text). Empty if it cannot be read."""
    try:
        values, _document = load_profile(session, settings.data_dir)
    except (OSError, ValueError, SecurityError):
        logger.warning("the profile note could not be read; chatting without it")
        return []
    return [(f.label, values[f.key]) for f in FIELDS if values.get(f.key, "").strip()]


def _assistant_context(session: Session | None, settings: Settings) -> AssistantContext:
    """Who the assistant is and what it knows. Defaults when the library cannot be opened."""
    if session is None:
        return AssistantContext()
    try:
        identity = load_identity(session)
        return AssistantContext(
            identity=identity,
            profile=_profile_facts(session, settings) if identity.use_profile else (),
            memories=[m.text for m in list_memories(session)] if identity.use_memory else (),
        )
    except (SQLAlchemyError, SecurityError):
        logger.warning("the assistant's settings could not be read; chatting with the defaults")
        session.rollback()
        return AssistantContext()


def _remember_in(session: Session) -> Remember:
    def remember(fact: str) -> None:
        try:
            add_memory(session, fact, "assistant")
        except SQLAlchemyError as exc:
            session.rollback()
            raise ValueError("the memory could not be written") from exc

    return remember


@router.post("/chat")
def chat(
    request: ChatRequest, llm: LLMDep, session: OptionalSessionDep, settings: SettingsDep
) -> ChatResponse:
    """Talk to the local model directly. The reply is not based on the notes and cites nothing.

    The model is told who it is and, if you allow it, what your Profile page and its memories
    say. No other note is read, so this works whatever state the library is in. To get an answer
    from the notes, with sources, use `/ask`.
    """
    history = [ChatMessage(turn.role, turn.content) for turn in request.history]
    context = _assistant_context(session, settings)
    can_remember = session is not None and context.identity.use_memory
    try:
        reply = reply_to(
            llm,
            request.message,
            history,
            context=context,
            remember=_remember_in(session) if session is not None and can_remember else None,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(exc)) from exc
    except (LLMUnavailableError, LLMModelNotFoundError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return ChatResponse(
        answer=reply.text,
        model=llm.model_name,
        truncated=reply.truncated,
        actions=[ChatAction(name=a.name, args=dict(a.args)) for a in reply.actions],
    )
