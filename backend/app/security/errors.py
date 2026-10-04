class SecurityError(RuntimeError):
    """Base class for encryption problems. Messages never contain keys or passphrases."""


class EncryptionNotSupportedError(SecurityError):
    """Encryption needs Windows' built-in cryptography, and this system does not have it."""


class DecryptionError(SecurityError):
    """Data could not be decrypted: wrong key, or the data was modified."""


class KeystoreError(SecurityError):
    """The key file (security.json) is missing, damaged or in an unexpected state."""


class LibraryLockedError(SecurityError):
    """The library is encrypted and no key is available: the Windows unlock did not work."""


class WrongPassphraseError(SecurityError):
    """The recovery passphrase does not unlock this library."""


class MigrationIncompleteError(SecurityError):
    """Encrypting the library was started but not finished; it must be resumed."""
