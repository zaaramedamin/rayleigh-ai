from datetime import datetime

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.api import library_sync
from app.api.deps import SessionDep, SettingsDep
from app.knowledge.library.documents import delete_document
from app.knowledge.profile.service import (
    FIELDS,
    MAX_FIELD_CHARS,
    PROFILE_FILENAME,
    load_profile,
    save_profile,
)

router = APIRouter(prefix="/profile", tags=["profile"])


class ProfileFieldInfo(BaseModel):
    key: str
    label: str
    hint: str


class ProfileValues(BaseModel):
    name: str = Field(default="", max_length=MAX_FIELD_CHARS)
    location: str = Field(default="", max_length=MAX_FIELD_CHARS)
    occupation: str = Field(default="", max_length=MAX_FIELD_CHARS)
    languages: str = Field(default="", max_length=MAX_FIELD_CHARS)
    interests: str = Field(default="", max_length=MAX_FIELD_CHARS)
    preferences: str = Field(default="", max_length=MAX_FIELD_CHARS)
    about: str = Field(default="", max_length=MAX_FIELD_CHARS)


class ProfileOut(BaseModel):
    values: ProfileValues
    fields: list[ProfileFieldInfo]
    saved_as: str = Field(description="The library note the profile is kept in.")
    updated_at: datetime | None
    searchable: bool = Field(description="The profile is indexed, so the assistant can use it.")


def _out(values: dict[str, str], document: object, searchable: bool) -> ProfileOut:
    updated = getattr(document, "created_at", None)
    return ProfileOut(
        values=ProfileValues(**values),
        fields=[ProfileFieldInfo(key=f.key, label=f.label, hint=f.hint) for f in FIELDS],
        saved_as=PROFILE_FILENAME,
        updated_at=updated,
        searchable=searchable,
    )


@router.get("")
def get_profile(session: SessionDep, settings: SettingsDep) -> ProfileOut:
    values, document = load_profile(session, settings.data_dir)
    searchable = document is not None and document.indexed_model == settings.embedding_model
    return _out(values, document, searchable)


@router.put("")
def put_profile(values: ProfileValues, session: SessionDep, settings: SettingsDep) -> ProfileOut:
    """Save what the assistant should know about you. It becomes a searchable, citable note."""
    new, old = save_profile(
        session,
        settings.data_dir,
        values.model_dump(),
        settings.chunk_size_chars,
        settings.chunk_overlap_chars,
    )
    searchable = False
    if new is not None:
        searchable = new.indexed_model == settings.embedding_model or library_sync.make_searchable(
            settings, session, [new.id]
        )
        session.refresh(new)
    if old is not None:
        library_sync.delete_vectors(settings, [old.id])
        delete_document(session, settings.data_dir, old, exclude=False)
    saved_values, document = load_profile(session, settings.data_dir)
    return _out(saved_values, document, searchable)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_profile(session: SessionDep, settings: SettingsDep) -> None:
    _new, old = save_profile(
        session, settings.data_dir, {}, settings.chunk_size_chars, settings.chunk_overlap_chars
    )
    if old is not None:
        library_sync.delete_vectors(settings, [old.id])
        delete_document(session, settings.data_dir, old, exclude=False)
