from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.ai.llm.base import (
    ChatMessage,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.api.deps import LLMDep
from app.assistant.chat import MAX_MESSAGE_CHARS, reply_to

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


class ChatResponse(BaseModel):
    answer: str
    model: str = Field(description="The local model that wrote the reply.")
    truncated: bool = Field(description="True when the reply was cut off at the length limit.")


@router.post("/chat")
def chat(request: ChatRequest, llm: LLMDep) -> ChatResponse:
    """Talk to the local model directly. The reply is not based on the notes and cites nothing.

    Nothing is read from the library, so this works whatever state the library is in. To get an
    answer from the notes, with sources, use `/ask`.
    """
    history = [ChatMessage(turn.role, turn.content) for turn in request.history]
    try:
        reply = reply_to(llm, request.message, history)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(exc)) from exc
    except (LLMUnavailableError, LLMModelNotFoundError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return ChatResponse(answer=reply.text, model=llm.model_name, truncated=reply.truncated)
