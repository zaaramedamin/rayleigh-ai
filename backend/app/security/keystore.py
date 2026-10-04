"""The key file (DATA_DIR/security.json) and everything about unlocking the library.

One random 256-bit key (the "library key") encrypts the library. It is never stored in the clear.
The key file holds two protected copies of it:

- wrapped by this Windows account (DPAPI): unlocks automatically while you are signed in;
- wrapped by a recovery passphrase (scrypt, then AES-256-GCM): unlocks the library on another
  computer, or if Windows is reinstalled or the account is lost.

The key file is not secret. Without the Windows account or the passphrase it unlocks nothing, so
it is safe to keep in backups; a backup restored on a new computer needs the recovery passphrase.

A library is in one of three states, recorded in the key file: "plaintext" (no key file),
"migrating" (encryption was started and not finished) and "encrypted".
"""

import hashlib
import json
import os
import re
import secrets
import unicodedata
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.security import cng, dpapi
from app.security.errors import (
    DecryptionError,
    KeystoreError,
    LibraryLockedError,
    SecurityError,
    WrongPassphraseError,
)
from app.security.vault import Vault

SECURITY_FILENAME = "security.json"
FORMAT_VERSION = 1
_KDF_N, _KDF_R, _KDF_P = 2**16, 8, 1
_KDF_MAXMEM = 2**28
_KEY_AAD = b"reyleight-library-key-v1"
MIN_PASSPHRASE_LENGTH = 12
_RECOVERY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_RECOVERY_LENGTH = 25
_GENERATED = re.compile(rf"^[A-Za-z2-7]{{{_RECOVERY_LENGTH}}}$")
_GENERATED_DISPLAY = re.compile(r"^[A-Z2-7]{5}(-[A-Z2-7]{5}){4}$")

LibraryState = Literal["plaintext", "migrating", "encrypted"]
UnlockMethod = Literal["windows", "passphrase"]


@dataclass(frozen=True)
class UnlockResult:
    vault: Vault
    method: UnlockMethod
    remembered_for_windows: bool = False  # a typed passphrase was saved for this Windows account


@dataclass(frozen=True)
class KeyFile:
    state: LibraryState
    created_at: str
    salt: bytes
    kdf_n: int
    kdf_r: int
    kdf_p: int
    wrapped_by_passphrase: bytes
    wrapped_by_windows: bytes | None


# --- passphrases ------------------------------------------------------------------------------


def _typed(text: str) -> str:
    """The passphrase as typed: only Unicode-normalized and trimmed at the ends."""
    return unicodedata.normalize("NFKC", text).strip()


def _generated_form(text: str) -> str | None:
    """The canonical form of a generated recovery key typed in any style, else None."""
    squeezed = _typed(text).replace("-", "").replace(" ", "")
    return squeezed.upper() if _GENERATED.match(squeezed) else None


def normalize_passphrase(text: str) -> str:
    """The form a new passphrase is stored under.

    A passphrase is used exactly as typed (so its case and spaces matter), with one exception:
    the dash-grouped key that `generate_recovery_passphrase` prints is stored without dashes.
    """
    typed = _typed(text)
    return typed.replace("-", "") if _GENERATED_DISPLAY.match(typed) else typed


def passphrase_candidates(text: str) -> list[str]:
    """What to try when unlocking: as typed, then (if it looks like a generated key) canonical.

    This lets a generated recovery key be typed in lower case or without dashes, without changing
    what any ordinary passphrase means.
    """
    candidates = [_typed(text)]
    canonical = _generated_form(text)
    if canonical is not None and canonical not in candidates:
        candidates.append(canonical)
    return candidates


def generate_recovery_passphrase() -> str:
    """A random recovery key with 125 bits of strength, such as ABCDE-FGHIJ-KLMNO-PQRST-UVWXY."""
    chars = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(_RECOVERY_LENGTH))
    return "-".join(chars[i : i + 5] for i in range(0, _RECOVERY_LENGTH, 5))


def validate_new_passphrase(passphrase: str) -> str:
    """Return the normalized passphrase, or raise ValueError saying what is wrong."""
    normalized = normalize_passphrase(passphrase)
    if len(normalized) < MIN_PASSPHRASE_LENGTH:
        raise ValueError(f"the passphrase must be at least {MIN_PASSPHRASE_LENGTH} characters")
    if len(set(normalized)) < 6:
        raise ValueError("the passphrase is too repetitive")
    return normalized


def _derive_key_encryption_key(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    """Stretch exactly this passphrase (already in its stored form) into a 256-bit key."""
    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=_KDF_MAXMEM,
        dklen=cng.KEY_BYTES,
    )


def _wrap(kek: bytes, library_key: bytes) -> bytes:
    nonce = os.urandom(cng.NONCE_BYTES)
    return nonce + cng.encrypt(kek, nonce, library_key, _KEY_AAD)


def _unwrap(kek: bytes, wrapped: bytes) -> bytes:
    if len(wrapped) < cng.NONCE_BYTES + cng.TAG_BYTES:
        raise DecryptionError("the wrapped key is damaged")
    return cng.decrypt(kek, wrapped[: cng.NONCE_BYTES], wrapped[cng.NONCE_BYTES :], _KEY_AAD)


# --- the key file -----------------------------------------------------------------------------


def _path(data_dir: Path) -> Path:
    return data_dir / SECURITY_FILENAME


def _encode(data: bytes | None) -> str | None:
    return None if data is None else b64encode(data).decode("ascii")


def _decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise KeystoreError(f"the key file is damaged ({field})")
    try:
        return b64decode(value.encode("ascii"), validate=True)
    except ValueError as exc:
        raise KeystoreError(f"the key file is damaged ({field})") from exc


def read_keyfile(data_dir: Path) -> KeyFile | None:
    path = _path(data_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        kdf = raw["kdf"]
        state = raw["state"]
        if raw["version"] != FORMAT_VERSION or state not in ("migrating", "encrypted"):
            raise KeystoreError("the key file has an unsupported version or state")
        windows = raw.get("wrapped_by_windows")
        return KeyFile(
            state=state,
            created_at=str(raw.get("created_at", "")),
            salt=_decode(kdf["salt"], "salt"),
            kdf_n=int(kdf["n"]),
            kdf_r=int(kdf["r"]),
            kdf_p=int(kdf["p"]),
            wrapped_by_passphrase=_decode(raw["wrapped_by_passphrase"], "passphrase key"),
            wrapped_by_windows=None if windows is None else _decode(windows, "Windows key"),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise KeystoreError("the key file (security.json) is damaged or unreadable") from exc


def _write_keyfile(data_dir: Path, keyfile: KeyFile) -> None:
    document = {
        "version": FORMAT_VERSION,
        "state": keyfile.state,
        "created_at": keyfile.created_at,
        "cipher": "AES-256-GCM",
        "kdf": {
            "name": "scrypt",
            "n": keyfile.kdf_n,
            "r": keyfile.kdf_r,
            "p": keyfile.kdf_p,
            "salt": _encode(keyfile.salt),
        },
        "wrapped_by_passphrase": _encode(keyfile.wrapped_by_passphrase),
        "wrapped_by_windows": _encode(keyfile.wrapped_by_windows),
    }
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def library_state(data_dir: Path) -> LibraryState:
    keyfile = read_keyfile(data_dir)
    return "plaintext" if keyfile is None else keyfile.state


def _protect_for_windows(library_key: bytes) -> bytes | None:
    try:
        return dpapi.protect(library_key)
    except SecurityError:
        return None  # the passphrase still works


# --- creating, unlocking, changing --------------------------------------------------------------


def create_keys(data_dir: Path, passphrase: str, *, state: LibraryState = "migrating") -> Vault:
    """Create a new library key protected by this Windows account and by `passphrase`."""
    if read_keyfile(data_dir) is not None:
        raise KeystoreError("this library already has a key file")
    normalized = validate_new_passphrase(passphrase)
    if state == "plaintext":
        raise KeystoreError("a key file always records an encrypted or migrating library")
    library_key = os.urandom(cng.KEY_BYTES)
    salt = os.urandom(16)
    kek = _derive_key_encryption_key(normalized, salt, _KDF_N, _KDF_R, _KDF_P)
    _write_keyfile(
        data_dir,
        KeyFile(
            state=state,
            created_at=datetime.now(UTC).isoformat(),
            salt=salt,
            kdf_n=_KDF_N,
            kdf_r=_KDF_R,
            kdf_p=_KDF_P,
            wrapped_by_passphrase=_wrap(kek, library_key),
            wrapped_by_windows=_protect_for_windows(library_key),
        ),
    )
    return Vault(library_key)


def _unlock_with_windows(keyfile: KeyFile) -> Vault | None:
    if keyfile.wrapped_by_windows is None:
        return None
    try:
        return Vault(dpapi.unprotect(keyfile.wrapped_by_windows))
    except SecurityError:
        return None


def _unlock_with_passphrase(keyfile: KeyFile, passphrase: str) -> Vault:
    for candidate in passphrase_candidates(passphrase):
        kek = _derive_key_encryption_key(
            candidate, keyfile.salt, keyfile.kdf_n, keyfile.kdf_r, keyfile.kdf_p
        )
        try:
            return Vault(_unwrap(kek, keyfile.wrapped_by_passphrase))
        except DecryptionError:
            continue
    raise WrongPassphraseError("that recovery passphrase does not unlock this library")


def unlock(data_dir: Path, *, passphrase: str | None = None, remember: bool = True) -> UnlockResult:
    """Get the library key: from this Windows account, else from the recovery passphrase.

    A passphrase that works is remembered for this Windows account (unless `remember` is False),
    so later runs unlock automatically. Raises LibraryLockedError when neither is available.
    """
    keyfile = read_keyfile(data_dir)
    if keyfile is None:
        raise KeystoreError("this library is not encrypted")
    vault = _unlock_with_windows(keyfile)
    if vault is not None:
        return UnlockResult(vault, "windows")
    if passphrase is None:
        raise LibraryLockedError(
            "the library is encrypted and Windows could not unlock it for this account"
        )
    vault = _unlock_with_passphrase(keyfile, passphrase)
    remembered = False
    if remember:
        protected = _protect_for_windows(vault.key)
        if protected is not None:
            _write_keyfile(data_dir, _replace(keyfile, wrapped_by_windows=protected))
            remembered = True
    return UnlockResult(vault, "passphrase", remembered)


def _replace(keyfile: KeyFile, **changes: Any) -> KeyFile:
    values = {**keyfile.__dict__, **changes}
    return KeyFile(**values)


def set_state(data_dir: Path, state: LibraryState) -> None:
    keyfile = read_keyfile(data_dir)
    if keyfile is None or state == "plaintext":
        raise KeystoreError("cannot change the state of a library without a key file")
    _write_keyfile(data_dir, _replace(keyfile, state=state))


def change_passphrase(data_dir: Path, vault: Vault, new_passphrase: str) -> None:
    """Protect the library key with a new recovery passphrase. The data is not re-encrypted."""
    keyfile = read_keyfile(data_dir)
    if keyfile is None:
        raise KeystoreError("this library is not encrypted")
    normalized = validate_new_passphrase(new_passphrase)
    salt = os.urandom(16)
    kek = _derive_key_encryption_key(normalized, salt, _KDF_N, _KDF_R, _KDF_P)
    _write_keyfile(
        data_dir,
        _replace(
            keyfile,
            salt=salt,
            kdf_n=_KDF_N,
            kdf_r=_KDF_R,
            kdf_p=_KDF_P,
            wrapped_by_passphrase=_wrap(kek, vault.key),
            wrapped_by_windows=_protect_for_windows(vault.key),
        ),
    )


@dataclass(frozen=True)
class RecoveryCheck:
    passphrase_works: bool
    windows_unlock_works: bool
    same_key_as_windows: bool | None  # None when the Windows copy cannot be read


def check_recovery(data_dir: Path, passphrase: str) -> RecoveryCheck:
    """Does this recovery passphrase unlock the library, and is it the same key Windows holds?"""
    keyfile = read_keyfile(data_dir)
    if keyfile is None:
        raise KeystoreError("this library is not encrypted")
    windows = _unlock_with_windows(keyfile)
    try:
        by_passphrase = _unlock_with_passphrase(keyfile, passphrase)
    except WrongPassphraseError:
        return RecoveryCheck(False, windows is not None, None)
    same = None if windows is None else secrets.compare_digest(windows.key, by_passphrase.key)
    return RecoveryCheck(True, windows is not None, same)


def windows_unlock_works(data_dir: Path) -> bool:
    keyfile = read_keyfile(data_dir)
    return keyfile is not None and _unlock_with_windows(keyfile) is not None
