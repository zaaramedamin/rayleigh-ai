"""The owner's profile, stored as the note "My profile.md" inside the library.

Keeping it as a normal note means it is chunked, indexed, encrypted and cited like any other
note: when a question is about you, the answer cites "My profile.md" and says so. It is used
only when a question matches it, and it is visible and deletable in the library. Nothing about
the owner is injected into prompts behind their back.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.chunking.service import chunk_document
from app.storage.files import read_file, store_file
from app.storage.models import KIND_PROFILE, Document

PROFILE_FILENAME = "My profile.md"
TITLE = "About me"
MAX_FIELD_CHARS = 2000


@dataclass(frozen=True)
class ProfileField:
    key: str
    label: str  # the heading in the note, so questions like "where do I live" match it
    hint: str


FIELDS: tuple[ProfileField, ...] = (
    ProfileField("name", "My name", "What should it call you?"),
    ProfileField("location", "Where I live", "City, country, time zone"),
    ProfileField("occupation", "What I do", "Work, studies, projects"),
    ProfileField("languages", "Languages I speak", "And which one you want answers in"),
    ProfileField("interests", "My interests", "Hobbies, topics you care about"),
    ProfileField("preferences", "How I like answers", "Short or detailed, tone, format"),
    ProfileField("about", "More about me", "Anything else it should know"),
)
_BY_LABEL = {f.label: f.key for f in FIELDS}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)


def clean_value(value: str) -> str:
    """Plain text only: no control characters, and no line that could pass for a heading."""
    text = _CONTROL.sub("", value).replace("\r\n", "\n").replace("\r", "\n")
    return _HEADING.sub("", text).strip()[:MAX_FIELD_CHARS]


def render_profile(values: dict[str, str]) -> str:
    sections = []
    for field in FIELDS:
        value = clean_value(values.get(field.key, ""))
        if value:
            sections.append(f"## {field.label}\n\n{value}\n")
    return f"# {TITLE}\n\n" + "\n".join(sections) if sections else ""


def parse_profile(text: str) -> dict[str, str]:
    """Read the fields back from a profile note written by `render_profile`."""
    values: dict[str, str] = {}
    key: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if key is not None:
            values[key] = "\n".join(buffer).strip()

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            key = _BY_LABEL.get(line[3:].strip())
            buffer = []
        elif key is not None:
            buffer.append(line)
    flush()
    return values


def find_profile(session: Session) -> Document | None:
    # By the plain `kind` column: the file name is encrypted in an encrypted library, and an
    # encrypted value can never be matched by an equality test.
    return session.scalars(
        select(Document).where(Document.kind == KIND_PROFILE).order_by(Document.id.desc())
    ).first()


def load_profile(session: Session, data_dir: Path) -> tuple[dict[str, str], Document | None]:
    document = find_profile(session)
    if document is None:
        return {key: "" for key in _BY_LABEL.values()}, None
    parsed = parse_profile(read_file(data_dir, document).decode("utf-8", errors="replace"))
    return {field.key: parsed.get(field.key, "") for field in FIELDS}, document


def save_profile(
    session: Session,
    data_dir: Path,
    values: dict[str, str],
    chunk_size: int,
    overlap: int,
) -> tuple[Document | None, Document | None]:
    """Write the profile note. Returns (new document or None if empty, replaced old document).

    The old note is not deleted here: the caller removes it (and its vectors) once the new one
    exists, so a failure never leaves the owner without a profile.
    """
    old = find_profile(session)
    body = render_profile(values)
    if not body:
        return None, old
    new, created = store_file(session, data_dir, body.encode("utf-8"), PROFILE_FILENAME)
    if new.kind != KIND_PROFILE:
        new.kind = KIND_PROFILE
        session.commit()
    if created:
        chunk_document(session, data_dir, new, chunk_size, overlap)
    return new, (old if old is not None and old.id != new.id else None)
