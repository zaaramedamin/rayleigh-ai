"""Transparent column encryption for SQLAlchemy.

`EncryptedText` columns are encrypted when written and decrypted when read, so the rest of the
application keeps working with ordinary strings. The key comes from the *engine* the session is
bound to (see `attach_vault`), so a plaintext library and an encrypted library can be open in the
same process without mixing up.

Safety rule: a value that is already encrypted is never returned as text without a key. It raises
LibraryLockedError instead of handing back ciphertext as if it were the note.
"""

from typing import Any

from sqlalchemy import Dialect, String, Text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.types import TypeDecorator

from app.security.errors import LibraryLockedError
from app.security.vault import Vault

_ATTRIBUTE = "reyleight_vault"


def attach_vault(engine: Engine, vault: Vault | None) -> None:
    """Make every encrypted column of sessions bound to `engine` use `vault`."""
    setattr(engine.dialect, _ATTRIBUTE, vault)


def vault_of(engine_or_dialect: Engine | Dialect) -> Vault | None:
    dialect = (
        engine_or_dialect.dialect if isinstance(engine_or_dialect, Engine) else engine_or_dialect
    )
    vault = getattr(dialect, _ATTRIBUTE, None)
    return vault if isinstance(vault, Vault) else None


def vault_of_session(session: Session) -> Vault | None:
    """The vault of the library this session works on, or None for a plaintext library."""
    bind = session.get_bind()
    return vault_of(bind if isinstance(bind, Engine) else bind.engine)


class _EncryptedBase(TypeDecorator[str]):
    cache_ok = True

    def __init__(self, context: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context = context

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        vault = vault_of(dialect)
        return value if vault is None else vault.encrypt_text(value, self.context)

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None or not Vault.is_encrypted_text(value):
            return value  # a plaintext library, or a value not migrated yet
        vault = vault_of(dialect)
        if vault is None:
            raise LibraryLockedError(
                "the library contains encrypted data but no key is available to read it"
            )
        return vault.decrypt_text(value, self.context)


class EncryptedText(_EncryptedBase):
    impl = Text
    cache_ok = True  # must be set on each concrete class; the state is just `context`


class EncryptedString(_EncryptedBase):
    """Same as EncryptedText, with a length for the database schema (SQLite ignores it)."""

    impl = String
    cache_ok = True
