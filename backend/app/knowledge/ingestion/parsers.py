import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser


class ParseError(ValueError):
    """Raised when a file's bytes cannot be treated as text. The message is a short reason code."""


@dataclass(frozen=True)
class Extracted:
    """The text of a file, and where its pages begin when it has pages.

    `page_starts` holds, for each page in order, the character offset in `text` where that page
    begins (so the first is 0). It is None for formats without pages, such as plain text.
    """

    text: str
    page_starts: tuple[int, ...] | None = None


def parse_text(data: bytes) -> str:
    """Decode text bytes as UTF-8 (a leading BOM is dropped).

    Markdown and other text formats are returned unchanged; structure is needed later.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError("not_utf8") from exc
    if "\x00" in text:
        raise ParseError("binary")
    return text


def parse_json(data: bytes) -> str:
    """Return the JSON text unchanged, after checking that it is valid JSON."""
    text = parse_text(data)
    try:
        json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ParseError("invalid_json") from exc
    return text


_SKIPPED_HTML_TAGS = frozenset({"script", "style", "noscript", "template"})
_BLOCK_HTML_TAGS = frozenset(
    {
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "header", "footer",
    }
)  # fmt: skip


class _HtmlTextExtractor(HTMLParser):
    """Collect visible text, dropping scripts and styles. Never executes or fetches anything."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_HTML_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_HTML_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


def parse_html(data: bytes) -> str:
    """Extract the visible text of an HTML document."""
    text = parse_text(data)
    extractor = _HtmlTextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
    except Exception as exc:
        raise ParseError("invalid_html") from exc
    return extractor.text()
