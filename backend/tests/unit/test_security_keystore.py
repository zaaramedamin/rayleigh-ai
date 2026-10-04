import json
import os
from pathlib import Path

import pytest

from app.security import keystore
from app.security.errors import KeystoreError, LibraryLockedError, WrongPassphraseError
from app.security.keystore import (
    SECURITY_FILENAME,
    change_passphrase,
    check_recovery,
    create_keys,
    generate_recovery_passphrase,
    library_state,
    normalize_passphrase,
    passphrase_candidates,
    read_keyfile,
    set_state,
    unlock,
    validate_new_passphrase,
    windows_unlock_works,
)

PASSPHRASE = "correct horse battery staple"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


def _drop_windows_copy(folder: Path) -> None:
    """Make the library look like it was copied to another computer or Windows account."""
    keyfile = read_keyfile(folder)
    assert keyfile is not None
    keystore._write_keyfile(folder, keystore._replace(keyfile, wrapped_by_windows=None))


# --- passphrases -----------------------------------------------------------------------------


def test_a_generated_recovery_key_has_the_documented_shape_and_is_random() -> None:
    first, second = generate_recovery_passphrase(), generate_recovery_passphrase()

    assert first != second
    groups = first.split("-")
    assert [len(g) for g in groups] == [5, 5, 5, 5, 5]
    assert set("".join(groups)) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


def test_a_generated_key_can_be_typed_in_any_reasonable_way() -> None:
    key = "ABCDE-FGHIJ-KLMNO-PQRST-UVWXY"

    for typed in (key, key.lower(), key.replace("-", ""), key.replace("-", " "), f"  {key}  "):
        assert "ABCDEFGHIJKLMNOPQRSTUVWXY" in passphrase_candidates(typed)
    # and it is stored without the dashes
    assert normalize_passphrase(key) == "ABCDEFGHIJKLMNOPQRSTUVWXY"


def test_an_ordinary_passphrase_is_used_exactly_as_typed() -> None:
    # Even a 25-letter phrase must keep its case and spaces: it is not a generated key.
    phrase = "correct horse battery staple"

    assert normalize_passphrase(phrase) == phrase
    assert normalize_passphrase("Correct Horse Battery Staple") != normalize_passphrase(phrase)
    assert passphrase_candidates(phrase)[0] == phrase
    assert normalize_passphrase("  padded phrase here  ") == "padded phrase here"


def test_unicode_forms_of_the_same_passphrase_are_equal() -> None:
    assert normalize_passphrase("café au lait long enough") == normalize_passphrase(
        "café au lait long enough"
    )


@pytest.mark.parametrize("weak", ["", "short", "elevenchars", "aaaaaaaaaaaaaaaa", "1111111111111"])
def test_weak_passphrases_are_refused_with_a_reason(weak: str) -> None:
    with pytest.raises(ValueError, match="passphrase"):
        validate_new_passphrase(weak)


def test_a_good_passphrase_is_accepted() -> None:
    assert validate_new_passphrase(PASSPHRASE) == PASSPHRASE


# --- the key file -----------------------------------------------------------------------------


def test_a_library_without_a_key_file_is_plaintext(folder: Path) -> None:
    assert library_state(folder) == "plaintext"
    assert read_keyfile(folder) is None


def test_creating_keys_writes_a_file_that_never_contains_the_key(folder: Path) -> None:
    vault = create_keys(folder, PASSPHRASE)

    text = (folder / SECURITY_FILENAME).read_text(encoding="utf-8")
    assert vault.key.hex() not in text
    assert PASSPHRASE not in text
    document = json.loads(text)
    assert document["state"] == "migrating"
    assert document["cipher"] == "AES-256-GCM"
    assert document["kdf"]["name"] == "scrypt"
    assert document["kdf"]["n"] == 65536
    assert document["wrapped_by_passphrase"] and document["wrapped_by_windows"]
    assert not list(folder.glob("*.tmp"))  # the atomic write left nothing behind


def test_keys_cannot_be_created_twice(folder: Path) -> None:
    create_keys(folder, PASSPHRASE)

    with pytest.raises(KeystoreError, match="already has a key file"):
        create_keys(folder, "another long passphrase")


def test_a_weak_passphrase_creates_nothing(folder: Path) -> None:
    with pytest.raises(ValueError):
        create_keys(folder, "weak")

    assert not (folder / SECURITY_FILENAME).exists()


def test_the_state_can_move_from_migrating_to_encrypted(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="migrating")
    assert library_state(folder) == "migrating"

    set_state(folder, "encrypted")

    assert library_state(folder) == "encrypted"


def test_a_key_file_cannot_claim_the_library_is_plaintext(folder: Path) -> None:
    with pytest.raises(KeystoreError):
        create_keys(folder, PASSPHRASE, state="plaintext")


# --- unlocking --------------------------------------------------------------------------------


def test_windows_unlocks_the_library_automatically(folder: Path) -> None:
    created = create_keys(folder, PASSPHRASE, state="encrypted")

    result = unlock(folder)

    assert result.method == "windows"
    assert result.vault.key == created.key
    assert windows_unlock_works(folder)


def test_without_the_windows_copy_the_library_is_locked_until_the_passphrase_is_given(
    folder: Path,
) -> None:
    created = create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)

    assert not windows_unlock_works(folder)
    with pytest.raises(LibraryLockedError):
        unlock(folder)

    result = unlock(folder, passphrase=PASSPHRASE)
    assert result.method == "passphrase"
    assert result.vault.key == created.key


def test_a_passphrase_that_works_is_remembered_for_this_windows_account(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)

    first = unlock(folder, passphrase=PASSPHRASE)
    second = unlock(folder)

    assert first.remembered_for_windows
    assert second.method == "windows"


def test_remembering_can_be_switched_off(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)

    result = unlock(folder, passphrase=PASSPHRASE, remember=False)

    assert not result.remembered_for_windows
    assert not windows_unlock_works(folder)


def test_a_wrong_passphrase_is_refused_and_changes_nothing(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)
    before = (folder / SECURITY_FILENAME).read_bytes()

    with pytest.raises(WrongPassphraseError):
        unlock(folder, passphrase="definitely not the passphrase")

    assert (folder / SECURITY_FILENAME).read_bytes() == before


def test_a_generated_recovery_key_unlocks_however_it_is_typed(folder: Path) -> None:
    recovery = generate_recovery_passphrase()
    created = create_keys(folder, recovery, state="encrypted")
    _drop_windows_copy(folder)

    for typed in (recovery, recovery.lower(), recovery.replace("-", "")):
        _drop_windows_copy(folder)
        assert unlock(folder, passphrase=typed).vault.key == created.key


def test_unlocking_a_library_with_no_key_file_is_an_error(folder: Path) -> None:
    with pytest.raises(KeystoreError, match="not encrypted"):
        unlock(folder)


# --- changing the passphrase -----------------------------------------------------------------


def test_changing_the_passphrase_keeps_the_data_key_and_retires_the_old_phrase(
    folder: Path,
) -> None:
    created = create_keys(folder, PASSPHRASE, state="encrypted")

    change_passphrase(folder, created, "a brand new long passphrase")
    _drop_windows_copy(folder)

    assert unlock(folder, passphrase="a brand new long passphrase").vault.key == created.key
    _drop_windows_copy(folder)
    with pytest.raises(WrongPassphraseError):
        unlock(folder, passphrase=PASSPHRASE)


def test_changing_the_passphrase_uses_a_fresh_salt(folder: Path) -> None:
    created = create_keys(folder, PASSPHRASE, state="encrypted")
    before = read_keyfile(folder)
    assert before is not None

    change_passphrase(folder, created, "a brand new long passphrase")

    after = read_keyfile(folder)
    assert after is not None and after.salt != before.salt


def test_a_weak_new_passphrase_changes_nothing(folder: Path) -> None:
    created = create_keys(folder, PASSPHRASE, state="encrypted")
    before = (folder / SECURITY_FILENAME).read_bytes()

    with pytest.raises(ValueError):
        change_passphrase(folder, created, "weak")

    assert (folder / SECURITY_FILENAME).read_bytes() == before


# --- checking the recovery passphrase -----------------------------------------------------------


def test_the_recovery_check_confirms_the_phrase_and_that_it_matches_windows(
    folder: Path,
) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")

    result = check_recovery(folder, PASSPHRASE)

    assert (result.passphrase_works, result.windows_unlock_works, result.same_key_as_windows) == (
        True,
        True,
        True,
    )


def test_the_recovery_check_catches_a_wrong_phrase(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")

    result = check_recovery(folder, "this was written down wrongly")

    assert result.passphrase_works is False


def test_the_recovery_check_catches_two_different_keys(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    keyfile = read_keyfile(folder)
    assert keyfile is not None
    other_windows_copy = keystore.dpapi.protect(os.urandom(32))
    keystore._write_keyfile(
        folder, keystore._replace(keyfile, wrapped_by_windows=other_windows_copy)
    )

    result = check_recovery(folder, PASSPHRASE)

    assert result.passphrase_works is True
    assert result.same_key_as_windows is False


def test_the_recovery_check_works_without_the_windows_copy(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)

    result = check_recovery(folder, PASSPHRASE)

    assert (result.passphrase_works, result.windows_unlock_works, result.same_key_as_windows) == (
        True,
        False,
        None,
    )


# --- damaged key files -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "{ not json",
        "[]",
        "{}",
        '{"version": 99, "state": "encrypted"}',
        '{"version": 1, "state": "plaintext", "kdf": {}}',
    ],
)
def test_a_damaged_key_file_is_a_clear_error_never_a_silent_plaintext_library(
    folder: Path, content: str
) -> None:
    (folder / SECURITY_FILENAME).write_text(content, encoding="utf-8")

    with pytest.raises(KeystoreError, match="damaged|unsupported"):
        library_state(folder)


def test_bad_base64_in_the_key_file_is_reported(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    document = json.loads((folder / SECURITY_FILENAME).read_text(encoding="utf-8"))
    document["wrapped_by_passphrase"] = "!!! not base64 !!!"
    (folder / SECURITY_FILENAME).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(KeystoreError, match="damaged"):
        read_keyfile(folder)


def test_a_modified_wrapped_key_is_never_accepted(folder: Path) -> None:
    create_keys(folder, PASSPHRASE, state="encrypted")
    _drop_windows_copy(folder)
    keyfile = read_keyfile(folder)
    assert keyfile is not None
    damaged = bytes([keyfile.wrapped_by_passphrase[20] ^ 1])
    wrapped = keyfile.wrapped_by_passphrase[:20] + damaged + keyfile.wrapped_by_passphrase[21:]
    keystore._write_keyfile(folder, keystore._replace(keyfile, wrapped_by_passphrase=wrapped))

    with pytest.raises(WrongPassphraseError):
        unlock(folder, passphrase=PASSPHRASE)
