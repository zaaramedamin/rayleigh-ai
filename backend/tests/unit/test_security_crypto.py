import base64
import os

import pytest

from app.security import cng, dpapi
from app.security.errors import DecryptionError, SecurityError
from app.security.vault import BLOB_MAGIC, TEXT_PREFIX, Vault

KEY = bytes(range(32))
NONCE = bytes(range(12))


# --- the cipher (Windows CNG) -----------------------------------------------------------------


def test_aes_256_gcm_matches_the_published_test_vector() -> None:
    # McGrew and Viega, "The Galois/Counter Mode of Operation (GCM)", Test Case 16 (AES-256).
    key = bytes.fromhex("feffe9928665731c6d6a8f9467308308" * 2)
    nonce = bytes.fromhex("cafebabefacedbaddecaf888")
    plaintext = bytes.fromhex(
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
        "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"
    )
    aad = bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddad2")
    expected_ciphertext = bytes.fromhex(
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
        "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662"
    )
    expected_tag = bytes.fromhex("76fc6ece0f4e1768cddf8853bb2d551b")

    sealed = cng.encrypt(key, nonce, plaintext, aad)

    assert sealed[:-16] == expected_ciphertext
    assert sealed[-16:] == expected_tag
    assert cng.decrypt(key, nonce, sealed, aad) == plaintext


def test_round_trip_for_many_sizes() -> None:
    for size in (0, 1, 15, 16, 17, 255, 4096, 100_000):
        data = os.urandom(size)
        sealed = cng.encrypt(KEY, NONCE, data, b"ctx")
        assert len(sealed) == size + cng.TAG_BYTES
        assert cng.decrypt(KEY, NONCE, sealed, b"ctx") == data


def test_the_same_input_with_different_nonces_gives_different_output() -> None:
    first = cng.encrypt(KEY, NONCE, b"same text", b"")
    second = cng.encrypt(KEY, bytes(12), b"same text", b"")

    assert first != second


@pytest.mark.parametrize("what", ["key", "ciphertext", "tag", "context", "nonce"])
def test_anything_wrong_is_rejected_never_decrypted_to_garbage(what: str) -> None:
    sealed = cng.encrypt(KEY, NONCE, b"secret note", b"ctx")
    key, nonce, blob, aad = KEY, NONCE, sealed, b"ctx"
    if what == "key":
        key = bytes(32)
    elif what == "ciphertext":
        blob = bytes([sealed[0] ^ 1]) + sealed[1:]
    elif what == "tag":
        blob = sealed[:-1] + bytes([sealed[-1] ^ 1])
    elif what == "context":
        aad = b"other"
    else:
        nonce = bytes(12)

    with pytest.raises(DecryptionError):
        cng.decrypt(key, nonce, blob, aad)


def test_truncated_data_is_rejected() -> None:
    with pytest.raises(DecryptionError):
        cng.decrypt(KEY, NONCE, b"short", b"")


@pytest.mark.parametrize("size", [0, 16, 31, 33])
def test_keys_must_be_exactly_32_bytes(size: int) -> None:
    with pytest.raises(SecurityError, match="32 bytes"):
        cng.encrypt(bytes(size), NONCE, b"x", b"")


def test_nonces_must_be_exactly_12_bytes() -> None:
    with pytest.raises(SecurityError, match="12 bytes"):
        cng.encrypt(KEY, bytes(8), b"x", b"")


def test_the_cipher_is_safe_to_use_from_several_threads() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def work(i: int) -> bool:
        data = os.urandom(1000 + i)
        nonce = os.urandom(12)
        return cng.decrypt(KEY, nonce, cng.encrypt(KEY, nonce, data, b"t"), b"t") == data

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(work, range(200)))


# --- Windows account protection (DPAPI) -------------------------------------------------------


def test_dpapi_round_trip_and_the_output_is_not_the_input() -> None:
    secret = os.urandom(32)

    protected = dpapi.protect(secret)

    assert secret not in protected
    assert dpapi.unprotect(protected) == secret


def test_dpapi_rejects_a_damaged_blob() -> None:
    protected = dpapi.protect(b"library key material")
    damaged = protected[:-1] + bytes([protected[-1] ^ 1])

    with pytest.raises(DecryptionError):
        dpapi.unprotect(damaged)
    with pytest.raises(DecryptionError):
        dpapi.unprotect(b"not a dpapi blob at all")


def test_dpapi_handles_empty_data() -> None:
    assert dpapi.unprotect(dpapi.protect(b"")) == b""


# --- the vault -------------------------------------------------------------------------------


def test_text_round_trip_including_unicode_and_empty() -> None:
    vault = Vault(KEY)
    for text in ("", "plain", "héllo wörld 日本語 🙂", "line one\nline two\r\n", "x" * 50_000):
        sealed = vault.encrypt_text(text, "chunks.text")
        assert sealed.startswith(TEXT_PREFIX)
        assert Vault.is_encrypted_text(sealed)
        assert vault.decrypt_text(sealed, "chunks.text") == text


def test_encrypted_text_contains_no_trace_of_the_plaintext() -> None:
    sealed = Vault(KEY).encrypt_text("the garage door code is 4821", "chunks.text")

    assert "garage" not in sealed
    assert "4821" not in sealed
    assert "garage" not in base64.urlsafe_b64decode(sealed[len(TEXT_PREFIX) :]).decode("latin-1")


def test_encrypting_the_same_text_twice_gives_different_values() -> None:
    vault = Vault(KEY)

    assert vault.encrypt_text("same", "c") != vault.encrypt_text("same", "c")


def test_a_value_does_not_decrypt_under_another_context() -> None:
    vault = Vault(KEY)
    sealed = vault.encrypt_text("note text", "chunks.text")

    with pytest.raises(DecryptionError):
        vault.decrypt_text(sealed, "documents.original_filename")


def test_a_different_key_cannot_decrypt() -> None:
    sealed = Vault(KEY).encrypt_text("secret", "c")

    with pytest.raises(DecryptionError):
        Vault(bytes(32)).decrypt_text(sealed, "c")


@pytest.mark.parametrize(
    "value", [TEXT_PREFIX + "!!!not base64!!!", TEXT_PREFIX + "QUJD", TEXT_PREFIX]
)
def test_damaged_encrypted_text_is_rejected(value: str) -> None:
    with pytest.raises(DecryptionError):
        Vault(KEY).decrypt_text(value, "c")


def test_decrypting_a_plain_value_is_an_error_not_a_pass_through() -> None:
    with pytest.raises(DecryptionError, match="not in the encrypted format"):
        Vault(KEY).decrypt_text("just text", "c")
    with pytest.raises(DecryptionError, match="not in the encrypted format"):
        Vault(KEY).decrypt_bytes(b"plain bytes", b"c")


def test_bytes_round_trip_and_format() -> None:
    vault = Vault(KEY)
    data = os.urandom(10_000)

    sealed = vault.encrypt_bytes(data, b"file:abc")

    assert sealed.startswith(BLOB_MAGIC)
    assert Vault.is_encrypted_blob(sealed)
    assert not Vault.is_encrypted_blob(data)
    assert not Vault.is_encrypted_blob(BLOB_MAGIC)  # too short to be real
    assert vault.decrypt_bytes(sealed, b"file:abc") == data


def test_a_stored_file_does_not_decrypt_under_another_file_name() -> None:
    vault = Vault(KEY)
    sealed = vault.encrypt_bytes(b"file contents", b"file:aaa")

    with pytest.raises(DecryptionError):
        vault.decrypt_bytes(sealed, b"file:bbb")


def test_a_modified_stored_file_is_detected() -> None:
    vault = Vault(KEY)
    sealed = bytearray(vault.encrypt_bytes(b"file contents", b"c"))
    sealed[20] ^= 1

    with pytest.raises(DecryptionError):
        vault.decrypt_bytes(bytes(sealed), b"c")


def test_the_vault_never_reveals_its_key_when_printed() -> None:
    vault = Vault(KEY)

    assert repr(vault) == "Vault(<key hidden>)"
    assert str(KEY.hex()) not in repr(vault)
    assert str(vault) == "Vault(<key hidden>)"


def test_a_vault_needs_a_32_byte_key() -> None:
    with pytest.raises(SecurityError):
        Vault(b"too short")
