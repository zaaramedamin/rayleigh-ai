from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import SessionDep
from app.assistant.identity import (
    DEFAULT_ROLE,
    MAX_ADDRESS_CHARS,
    MAX_NAME_CHARS,
    MAX_ROLE_CHARS,
    Identity,
    load_identity,
    save_identity,
)
from app.assistant.memory import (
    MAX_MEMORIES,
    MAX_MEMORY_CHARS,
    MemoryFullError,
    add_memory,
    clear_memories,
    delete_memory,
    list_memories,
)
from app.storage.models import Memory

router = APIRouter(prefix="/assistant", tags=["assistant"])


class IdentityValues(BaseModel):
    name: str = Field(default="", max_length=MAX_NAME_CHARS)
    address: str = Field(
        default="", max_length=MAX_ADDRESS_CHARS, description="How the assistant addresses you."
    )
    role: str = Field(default="", max_length=MAX_ROLE_CHARS)
    use_profile: bool = Field(
        default=True, description="Tell the model what the Profile page says about you."
    )
    use_memory: bool = Field(
        default=True, description="Tell the model its memories, and let it save new ones."
    )


class IdentityOut(IdentityValues):
    default_role: str = Field(description="The role used when none is written.")


class MemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MEMORY_CHARS)


class MemoryOut(BaseModel):
    id: int
    text: str
    origin: Literal["owner", "assistant"]
    created_at: datetime


class MemoryList(BaseModel):
    memories: list[MemoryOut]
    limit: int = Field(description="The most memories that can be kept.")


class ClearedMemories(BaseModel):
    deleted: int


def _identity_out(identity: Identity) -> IdentityOut:
    return IdentityOut(
        name=identity.name,
        address=identity.address,
        role=identity.role,
        use_profile=identity.use_profile,
        use_memory=identity.use_memory,
        default_role=DEFAULT_ROLE,
    )


def _memory_out(memory: Memory) -> MemoryOut:
    return MemoryOut(
        id=memory.id,
        text=memory.text,
        origin="assistant" if memory.origin == "assistant" else "owner",
        created_at=memory.created_at,
    )


@router.get("")
def get_identity(session: SessionDep) -> IdentityOut:
    """Who the assistant is: its name, how it addresses you, and the role you gave it."""
    return _identity_out(load_identity(session))


@router.put("")
def put_identity(values: IdentityValues, session: SessionDep) -> IdentityOut:
    """Save the assistant's identity. An empty name or role goes back to the default."""
    saved = save_identity(
        session,
        Identity(
            name=values.name,
            address=values.address,
            role=values.role,
            use_profile=values.use_profile,
            use_memory=values.use_memory,
        ),
    )
    return _identity_out(saved)


@router.get("/memories")
def get_memories(session: SessionDep) -> MemoryList:
    """Everything the assistant remembers, oldest first."""
    return MemoryList(
        memories=[_memory_out(memory) for memory in list_memories(session)], limit=MAX_MEMORIES
    )


@router.post("/memories", status_code=status.HTTP_201_CREATED)
def post_memory(body: MemoryIn, session: SessionDep) -> MemoryOut:
    """Add a memory yourself. Adding one that is already there returns the existing one."""
    try:
        memory, _created = add_memory(session, body.text, "owner")
    except MemoryFullError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _memory_out(memory)


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_memory(memory_id: int, session: SessionDep) -> None:
    if not delete_memory(session, memory_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That memory does not exist.")


@router.delete("/memories")
def remove_all_memories(session: SessionDep) -> ClearedMemories:
    """Forget everything."""
    return ClearedMemories(deleted=clear_memories(session))
