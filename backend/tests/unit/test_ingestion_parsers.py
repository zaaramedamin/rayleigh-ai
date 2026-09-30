import pytest

from app.knowledge.ingestion.parsers import ParseError, parse_text


def test_parse_text_decodes_utf8() -> None:
    assert parse_text("héllo\n# Title".encode()) == "héllo\n# Title"


def test_parse_text_strips_bom() -> None:
    assert parse_text(b"\xef\xbb\xbfhello") == "hello"


def test_parse_text_rejects_non_utf8() -> None:
    with pytest.raises(ParseError, match="not_utf8"):
        parse_text(b"\x80abc")


def test_parse_text_rejects_binary_with_nul_bytes() -> None:
    with pytest.raises(ParseError, match="binary"):
        parse_text(b"abc\x00def")
