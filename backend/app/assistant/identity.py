"""Who the assistant is: its name, how it addresses the owner, and the role the owner gave it.

The owner writes these on the Profile page. They are stored in the library database (encrypted
when the library is), and they become part of the instructions the local model is given in
general chat. Nothing here is ever taken from a note or from the model's own output.
"""

import re
from dataclasses import asdict, dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models import AssistantSetting

MAX_NAME_CHARS = 40
MAX_ADDRESS_CHARS = 40
MAX_ROLE_CHARS = 2000

DEFAULT_ROLE = (
    "You are my personal assistant, in the spirit of a capable, calm and loyal butler. "
    "You help me find things in my notes, answer my questions, remember what matters to me, "
    "and operate this application when I ask."
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Identity:
    name: str = "Reyleight"
    # How the assistant addresses the owner ("sir", "ma'am", a first name, ...). May be empty.
    address: str = "sir"
    role: str = DEFAULT_ROLE
    # Tell the model what the Profile page says about the owner, in every general conversation.
    use_profile: bool = True
    # Tell the model what it was asked to remember, and let it save new memories.
    use_memory: bool = True


_TEXT_LIMITS = {"name": MAX_NAME_CHARS, "address": MAX_ADDRESS_CHARS, "role": MAX_ROLE_CHARS}
_FLAGS = ("use_profile", "use_memory")


def _one_line(value: str) -> str:
    return " ".join(_CONTROL.sub("", value).split())


def clean_identity(identity: Identity) -> Identity:
    """Trimmed, bounded, without control characters. An empty name or role becomes the default."""
    role = _CONTROL.sub("", identity.role).replace("\r\n", "\n").replace("\r", "\n").strip()
    return replace(
        identity,
        name=_one_line(identity.name)[:MAX_NAME_CHARS] or Identity.name,
        address=_one_line(identity.address)[:MAX_ADDRESS_CHARS],
        role=role[:MAX_ROLE_CHARS] or DEFAULT_ROLE,
    )


def load_identity(session: Session) -> Identity:
    stored = {row.key: row.value for row in session.scalars(select(AssistantSetting))}
    values: dict[str, str | bool] = {}
    for key in _TEXT_LIMITS:
        if key in stored:
            values[key] = stored[key]
    for key in _FLAGS:
        if key in stored:
            values[key] = stored[key] == "true"
    return clean_identity(replace(Identity(), **values))  # type: ignore[arg-type]


def save_identity(session: Session, identity: Identity) -> Identity:
    cleaned = clean_identity(identity)
    rows = {row.key: row for row in session.scalars(select(AssistantSetting))}
    for key, value in asdict(cleaned).items():
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        if key in rows:
            rows[key].value = text
        else:
            session.add(AssistantSetting(key=key, value=text))
    session.commit()
    return cleaned
