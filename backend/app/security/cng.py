"""AES-256-GCM through Windows' own cryptography (CNG, bcrypt.dll), using only the standard library.

Authenticated encryption: decrypting with the wrong key, a different context (`aad`) or modified
data fails; it never returns wrong data. Nothing has to be installed, and because bcrypt.dll is a
signed Windows component there is nothing for Smart App Control to block.

Nonces must never repeat for one key. Callers use a fresh random 96-bit nonce for every message,
which stays safe far beyond the number of values a personal library will ever hold.
"""

import ctypes
import sys
from ctypes import wintypes
from functools import lru_cache
from typing import Any

from app.security.errors import DecryptionError, EncryptionNotSupportedError, SecurityError

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16

_STATUS_AUTH_TAG_MISMATCH = 0xC000A002
_AUTH_MODE_INFO_VERSION = 1


class _AuthInfo(ctypes.Structure):
    """BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO"""

    _fields_ = [
        ("cbSize", wintypes.ULONG),
        ("dwInfoVersion", wintypes.ULONG),
        ("pbNonce", ctypes.c_void_p),
        ("cbNonce", wintypes.ULONG),
        ("pbAuthData", ctypes.c_void_p),
        ("cbAuthData", wintypes.ULONG),
        ("pbTag", ctypes.c_void_p),
        ("cbTag", wintypes.ULONG),
        ("pbMacContext", ctypes.c_void_p),
        ("cbMacContext", wintypes.ULONG),
        ("cbAAD", wintypes.ULONG),
        ("cbData", ctypes.c_ulonglong),
        ("dwFlags", wintypes.ULONG),
    ]


@lru_cache(maxsize=1)
def _library() -> Any:
    if sys.platform != "win32":
        raise EncryptionNotSupportedError(
            "encryption uses Windows' built-in cryptography and is only available on Windows"
        )
    lib = ctypes.WinDLL("bcrypt")
    pointer = ctypes.c_void_p
    lib.BCryptOpenAlgorithmProvider.argtypes = [
        ctypes.POINTER(pointer),
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.ULONG,
    ]
    lib.BCryptSetProperty.argtypes = [
        pointer,
        wintypes.LPCWSTR,
        pointer,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    lib.BCryptGenerateSymmetricKey.argtypes = [
        pointer,
        ctypes.POINTER(pointer),
        pointer,
        wintypes.ULONG,
        pointer,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    for name in ("BCryptEncrypt", "BCryptDecrypt"):
        getattr(lib, name).argtypes = [
            pointer,
            pointer,
            wintypes.ULONG,
            pointer,
            pointer,
            wintypes.ULONG,
            pointer,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.ULONG,
        ]
    lib.BCryptDestroyKey.argtypes = [pointer]
    for name in (
        "BCryptOpenAlgorithmProvider",
        "BCryptSetProperty",
        "BCryptGenerateSymmetricKey",
        "BCryptEncrypt",
        "BCryptDecrypt",
        "BCryptDestroyKey",
    ):
        getattr(lib, name).restype = ctypes.c_long
    return lib


def _check(status: int, what: str) -> None:
    if status != 0:
        raise SecurityError(
            f"Windows cryptography call failed ({what}): 0x{status & 0xFFFFFFFF:08x}"
        )


@lru_cache(maxsize=1)
def _algorithm() -> ctypes.c_void_p:
    lib = _library()
    handle = ctypes.c_void_p()
    _check(lib.BCryptOpenAlgorithmProvider(ctypes.byref(handle), "AES", None, 0), "open AES")
    mode = ctypes.create_unicode_buffer("ChainingModeGCM")
    _check(
        lib.BCryptSetProperty(handle, "ChainingMode", mode, ctypes.sizeof(mode), 0), "select GCM"
    )
    return handle


def _make_key(key: bytes) -> ctypes.c_void_p:
    if len(key) != KEY_BYTES:
        raise SecurityError("the encryption key must be 32 bytes")
    handle = ctypes.c_void_p()
    key_buffer = ctypes.create_string_buffer(key, len(key))
    _check(
        _library().BCryptGenerateSymmetricKey(
            _algorithm(), ctypes.byref(handle), None, 0, key_buffer, len(key), 0
        ),
        "create key",
    )
    return handle


def _auth_info(nonce: bytes, aad: bytes, tag: Any) -> tuple[_AuthInfo, tuple[Any, ...]]:
    if len(nonce) != NONCE_BYTES:
        raise SecurityError("the nonce must be 12 bytes")
    nonce_buffer = ctypes.create_string_buffer(nonce, len(nonce))
    aad_buffer = ctypes.create_string_buffer(aad or b"\0", max(len(aad), 1))
    info = _AuthInfo()
    info.cbSize = ctypes.sizeof(_AuthInfo)
    info.dwInfoVersion = _AUTH_MODE_INFO_VERSION
    info.pbNonce = ctypes.cast(nonce_buffer, ctypes.c_void_p)
    info.cbNonce = len(nonce)
    info.pbAuthData = ctypes.cast(aad_buffer, ctypes.c_void_p)
    info.cbAuthData = len(aad)
    info.pbTag = ctypes.cast(tag, ctypes.c_void_p)
    info.cbTag = TAG_BYTES
    return info, (nonce_buffer, aad_buffer, tag)  # the buffers must outlive the call


def encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """Return ciphertext followed by the 16-byte authentication tag."""
    tag = ctypes.create_string_buffer(TAG_BYTES)
    info, keep = _auth_info(nonce, aad, tag)
    handle = _make_key(key)
    try:
        size = max(len(plaintext), 1)
        data = ctypes.create_string_buffer(plaintext, size)
        out = ctypes.create_string_buffer(size)
        written = wintypes.ULONG()
        _check(
            _library().BCryptEncrypt(
                handle,
                data,
                len(plaintext),
                ctypes.byref(info),
                None,
                0,
                out,
                size,
                ctypes.byref(written),
                0,
            ),
            "encrypt",
        )
    finally:
        _library().BCryptDestroyKey(handle)
    del keep
    return bytes(out.raw[: written.value]) + bytes(tag.raw)


def decrypt(key: bytes, nonce: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    """Decrypt `ciphertext + tag`. Raises DecryptionError for a wrong key or modified data."""
    if len(blob) < TAG_BYTES:
        raise DecryptionError("the encrypted value is too short")
    ciphertext, tag_bytes = blob[:-TAG_BYTES], blob[-TAG_BYTES:]
    tag = ctypes.create_string_buffer(tag_bytes, TAG_BYTES)
    info, keep = _auth_info(nonce, aad, tag)
    handle = _make_key(key)
    try:
        size = max(len(ciphertext), 1)
        data = ctypes.create_string_buffer(ciphertext, size)
        out = ctypes.create_string_buffer(size)
        written = wintypes.ULONG()
        status = _library().BCryptDecrypt(
            handle,
            data,
            len(ciphertext),
            ctypes.byref(info),
            None,
            0,
            out,
            size,
            ctypes.byref(written),
            0,
        )
    finally:
        _library().BCryptDestroyKey(handle)
    del keep
    if status != 0:
        if (status & 0xFFFFFFFF) == _STATUS_AUTH_TAG_MISMATCH:
            raise DecryptionError("wrong key, or the data was modified")
        _check(status, "decrypt")
    return bytes(out.raw[: written.value])
