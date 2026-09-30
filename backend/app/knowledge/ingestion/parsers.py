class ParseError(ValueError):
    """Raised when a file's bytes cannot be treated as text. The message is a short reason code."""


def parse_text(data: bytes) -> str:
    """Decode TXT/Markdown bytes as UTF-8 (a leading BOM is dropped).

    Markdown is returned unchanged; headings are needed later for chunking.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError("not_utf8") from exc
    if "\x00" in text:
        raise ParseError("binary")
    return text
