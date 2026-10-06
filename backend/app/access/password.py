"""The access password. Only a salted scrypt hash is stored, in DATA_DIR/access.json.

The file holds no secret that unlocks anything by itself: without the password it is useless.
Forgetting the password means deleting the file and choosing a new one (the notes are not
touched, because this password protects the application, not the data files).
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import unicodedata
from pathlib import Path

ACCESS_FILENAME = "access.json"
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256
_N, _R, _P = 2**15, 8, 1
_MAXMEM = 2**27
_FORMAT_VERSION = 1


class PasswordError(ValueError):
    """The password is not acceptable (too short or too long)."""


def _path(data_dir: Path) -> Path:
    return data_dir / ACCESS_FILENAME


def _normalise(password: str) -> bytes:
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def _hash(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(_normalise(password), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=32)


def validate_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordError(f"The password must be at most {MAX_PASSWORD_LENGTH} characters.")


def is_configured(data_dir: Path) -> bool:
    return _path(data_dir).is_file()


def set_password(data_dir: Path, password: str) -> None:
    """Choose (or replace) the access password."""
    validate_new_password(password)
    salt = secrets.token_bytes(16)
    record = {
        "version": _FORMAT_VERSION,
        "n": _N,
        "r": _R,
        "p": _P,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(_hash(password, salt, _N, _R, _P)).decode("ascii"),
    }
    target = _path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(record, tmp)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def verify_password(data_dir: Path, password: str) -> bool:
    """True if `password` is the access password. False if wrong or none has been set."""
    try:
        record = json.loads(_path(data_dir).read_text(encoding="utf-8"))
        salt = base64.b64decode(record["salt"])
        expected = base64.b64decode(record["hash"])
        actual = _hash(password, salt, int(record["n"]), int(record["r"]), int(record["p"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)
