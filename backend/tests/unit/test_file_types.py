import re
from pathlib import Path

import pytest

from app.knowledge.ingestion.file_types import (
    FILE_TYPES,
    extract_text,
    file_type_for,
    supported_extensions,
)
from app.knowledge.ingestion.parsers import ParseError

README = Path(__file__).resolve().parents[3] / "README.md"


def test_extensions_are_lowercase_dotted_and_unique() -> None:
    all_extensions = [ext for ft in FILE_TYPES for ext in ft.extensions]

    assert all(re.fullmatch(r"\.[a-z0-9]+", ext) for ext in all_extensions)
    assert len(all_extensions) == len(set(all_extensions))


def test_readme_lists_every_supported_extension() -> None:
    readme = README.read_text(encoding="utf-8")

    missing = [ext for ext in supported_extensions() if f"`{ext}`" not in readme]

    assert missing == []


def test_lookup_is_case_insensitive_and_rejects_unknown() -> None:
    assert file_type_for(".TXT") is not None
    assert file_type_for(".exe") is None
    assert file_type_for("") is None


def test_extract_text_rejects_unsupported_suffix() -> None:
    with pytest.raises(ValueError):
        extract_text(".exe", b"MZ")


def test_csv_is_read_as_text() -> None:
    assert extract_text(".csv", b"name,kcal\noats,389\n") == "name,kcal\noats,389\n"


def test_valid_json_is_returned_unchanged() -> None:
    assert extract_text(".json", b'{"a": [1, 2]}') == '{"a": [1, 2]}'


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ParseError, match="invalid_json"):
        extract_text(".json", b'{"a": ')


def test_deeply_nested_json_is_rejected_not_crashing() -> None:
    with pytest.raises(ParseError, match="invalid_json"):
        extract_text(".json", b"[" * 100_000)


def test_html_keeps_visible_text_and_drops_scripts_and_styles() -> None:
    html = (
        b"<html><head><style>p{color:red}</style><script>alert('x')</script></head>"
        b"<body><h1>Title</h1><p>Fish &amp; chips</p><noscript>enable js</noscript></body></html>"
    )

    text = extract_text(".html", html)

    assert "Title" in text
    assert "Fish & chips" in text
    assert "alert" not in text
    assert "color" not in text
    assert "enable js" not in text


def test_html_block_tags_become_line_breaks() -> None:
    text = extract_text(".htm", b"<p>one</p><p>two</p>")

    assert text.splitlines() == ["one", "", "two"]


def test_html_with_no_visible_text_yields_blank() -> None:
    assert extract_text(".html", b"<script>var x=1</script>").strip() == ""


def test_non_utf8_html_is_rejected() -> None:
    with pytest.raises(ParseError, match="not_utf8"):
        extract_text(".html", b"<p>\x80</p>")
