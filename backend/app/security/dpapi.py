"""Windows' per-account data protection (DPAPI), through the standard library.

A value protected here can be unprotected only by the same Windows account on the same machine
(and only while that account's credentials are intact). Reyleight uses it to keep the library key
available automatically while you are signed in, without storing it in the clear.
"""

import ctypes
import sys
from ctypes import wintypes

from app.security.errors import DecryptionError, EncryptionNotSupportedError, SecurityError

_ENTROPY = b"reyleight-library-key-v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data, max(len(data), 1))
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buffer


def _libraries() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    if sys.platform != "win32":
        raise EncryptionNotSupportedError("Windows data protection is only available on Windows")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _transform(function: str, data: bytes) -> bytes:
    crypt32, kernel32 = _libraries()
    source, source_buffer = _blob(data)
    entropy, entropy_buffer = _blob(_ENTROPY)
    result = _DataBlob()
    call = getattr(crypt32, function)
    call.restype = wintypes.BOOL
    # CryptProtectData and CryptUnprotectData take the same arguments here.
    ok = call(
        ctypes.byref(source),
        None,
        ctypes.byref(entropy),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(result),
    )
    del source_buffer, entropy_buffer
    if not ok:
        code = ctypes.get_last_error()
        if function == "CryptUnprotectData":
            raise DecryptionError(
                f"Windows could not unlock the key for this account (error {code})"
            )
        raise SecurityError(f"Windows could not protect the key (error {code})")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(result.pbData, ctypes.c_void_p))


def protect(data: bytes) -> bytes:
    """Encrypt `data` so that only this Windows account can decrypt it."""
    return _transform("CryptProtectData", data)


def unprotect(blob: bytes) -> bytes:
    """Reverse `protect`. Raises DecryptionError for another account, machine or a damaged blob."""
    return _transform("CryptUnprotectData", blob)
