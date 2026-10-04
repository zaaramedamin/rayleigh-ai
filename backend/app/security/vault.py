"""The Vault holds the library key in memory and encrypts and decrypts values and files.

Formats (version 1):
- bytes (stored files):   b"RLE1" + 12-byte nonce + ciphertext + 16-byte tag
- text (database values): "reyleight-enc-v1:" + base64url(12-byte nonce + ciphertext + tag)

Every value is bound to a *context* (which column, or which file) that is checked on decryption,
so a value moved to another column or another file name does not decrypt. A context cannot tell
two rows of the same column apart, so swapping whole rows is not detected; changing any value is.
"""

import base64
import binascii
import os

from app.security import cng
from app.security.errors import DecryptionError, SecurityError

BLOB_MAGIC = b"RLE1"
TEXT_PREFIX = "reyleight-enc-v1:"
_OVERHEAD_MIN = len(BLOB_MAGIC) + cng.NONCE_BYTES + cng.TAG_BYTES


class Vault:
    def __init__(self, key: bytes) -> None:
        if len(key) != cng.KEY_BYTES:
            raise SecurityError("the encryption key must be 32 bytes")
        self._key = key

    def __repr__(self) -> str:  # never reveal key material in logs or tracebacks
        return "Vault(<key hidden>)"

    @property
    def key(self) -> bytes:
        """The raw key, for wrapping it into the key file. Handle with care."""
        return self._key

    # --- bytes (stored files) -----------------------------------------------------------------

    @staticmethod
    def is_encrypted_blob(data: bytes) -> bool:
        return len(data) >= _OVERHEAD_MIN and data.startswith(BLOB_MAGIC)

    def encrypt_bytes(self, data: bytes, context: bytes) -> bytes:
        nonce = os.urandom(cng.NONCE_BYTES)
        return BLOB_MAGIC + nonce + cng.encrypt(self._key, nonce, data, context)

    def decrypt_bytes(self, blob: bytes, context: bytes) -> bytes:
        if not self.is_encrypted_blob(blob):
            raise DecryptionError("this data is not in the encrypted format")
        body = blob[len(BLOB_MAGIC) :]
        return cng.decrypt(self._key, body[: cng.NONCE_BYTES], body[cng.NONCE_BYTES :], context)

    # --- text (database values) ---------------------------------------------------------------

    @staticmethod
    def is_encrypted_text(value: str) -> bool:
        return value.startswith(TEXT_PREFIX)

    def encrypt_text(self, text: str, context: str) -> str:
        nonce = os.urandom(cng.NONCE_BYTES)
        sealed = cng.encrypt(self._key, nonce, text.encode("utf-8"), context.encode("utf-8"))
        return TEXT_PREFIX + base64.urlsafe_b64encode(nonce + sealed).decode("ascii")

    def decrypt_text(self, value: str, context: str) -> str:
        if not self.is_encrypted_text(value):
            raise DecryptionError("this value is not in the encrypted format")
        try:
            raw = base64.urlsafe_b64decode(value[len(TEXT_PREFIX) :].encode("ascii"))
        except (binascii.Error, UnicodeEncodeError) as exc:
            raise DecryptionError("the encrypted value is damaged") from exc
        if len(raw) < cng.NONCE_BYTES + cng.TAG_BYTES:
            raise DecryptionError("the encrypted value is damaged")
        plain = cng.decrypt(
            self._key, raw[: cng.NONCE_BYTES], raw[cng.NONCE_BYTES :], context.encode("utf-8")
        )
        return plain.decode("utf-8")
