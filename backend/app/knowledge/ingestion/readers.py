"""Readers for formats that need a real parser. They run inside the sandbox (see sandbox.py).

Every reader takes a file's bytes and returns `Extracted` (text, and where each page begins), or
raises ParseError with a short reason code:

- ``corrupt``         the file is damaged or is not what its extension says
- ``encrypted``       it needs a password
- ``no_text``         there is no text to read (a scan: it would need OCR)
- ``too_many_pages``  more pages than PDF_MAX_PAGES

Options (from the settings) reach the worker as one JSON environment variable, see `option`.
"""

import io
import json
import math
import os
import re
from collections import Counter
from typing import Any

from app.knowledge.ingestion.parsers import Extracted, ParseError

OPTIONS_ENV = "REYLEIGHT_READER_OPTIONS"
DEFAULT_PDF_MAX_PAGES = 2000
# Less text than this in a whole file means there is nothing to read.
MIN_TEXT_CHARS = 20
# A line that repeats at the top or bottom of at least this share of the pages (and on at least
# three of them) is a running header or footer, not content.
REPEATED_LINE_SHARE = 0.6
REPEATED_LINE_MIN_PAGES = 4

_LIGATURES = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        "­": "",  # soft hyphen
    }
)
# A word cut at the end of a line: "informa-" then "tion". Only when both sides are lower case, so
# "well-known" and "Anglo-Saxon" keep their hyphen.
_HYPHENATED = re.compile(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def option(name: str, default: Any) -> Any:
    """A setting handed to the worker by the program that started it."""
    try:
        options = json.loads(os.environ.get(OPTIONS_ENV, "{}"))
    except ValueError:
        return default
    return options.get(name, default) if isinstance(options, dict) else default


# --- PDF ----------------------------------------------------------------------------------------


def _normalise_line(line: str) -> str:
    """A line with its numbers masked, so "Page 3 of 9" and "Page 4 of 9" compare equal."""
    return _SPACES.sub(" ", _DIGITS.sub("#", line.strip().lower()))


def drop_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers and footers: lines at the edge of most pages that repeat."""
    if len(pages) < REPEATED_LINE_MIN_PAGES:
        return pages
    split = [text.split("\n") for text in pages]
    edges: list[dict[int, str]] = []
    seen: Counter[str] = Counter()
    for lines in split:
        filled = [i for i, line in enumerate(lines) if line.strip()]
        keys = {i: _normalise_line(lines[i]) for i in {*filled[:2], *filled[-2:]}}
        keys = {i: key for i, key in keys.items() if len(key) >= 3}
        edges.append(keys)
        seen.update(set(keys.values()))
    threshold = max(3, math.ceil(REPEATED_LINE_SHARE * len(pages)))
    repeated = {key for key, count in seen.items() if count >= threshold}
    if not repeated:
        return pages
    return [
        "\n".join(line for i, line in enumerate(lines) if keys.get(i) not in repeated)
        for lines, keys in zip(split, edges, strict=True)
    ]


def clean_page_text(text: str) -> str:
    text = text.translate(_LIGATURES).replace("\r\n", "\n").replace("\r", "\n")
    return _HYPHENATED.sub("", text).strip()


def read_pdf(data: bytes) -> Extracted:
    """The text of a PDF, page by page, so a citation can say which page."""
    from pypdf import PdfReader
    from pypdf.errors import DependencyError

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception as exc:  # a damaged file can break the parser in many ways
        raise ParseError("corrupt") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty password only to limit printing or editing.
        # Those open without a password; anything that asks for one cannot be read here.
        try:
            opened = reader.decrypt("")
        except DependencyError as exc:  # the cipher needs a library that is not installed
            raise ParseError("encrypted") from exc
        except Exception as exc:
            raise ParseError("encrypted") from exc
        if not opened:
            raise ParseError("encrypted")

    try:
        count = len(reader.pages)
    except Exception as exc:
        raise ParseError("corrupt") from exc
    if count == 0:
        raise ParseError("no_text")
    if count > int(option("pdf_max_pages", DEFAULT_PDF_MAX_PAGES)):
        raise ParseError("too_many_pages")

    texts: list[str] = []
    for page in reader.pages:
        try:
            texts.append(clean_page_text(page.extract_text() or ""))
        except Exception:  # one unreadable page should not lose the whole file
            texts.append("")
    if not option("pdf_keep_headers_footers", False):
        texts = [text.strip() for text in drop_repeated_lines(texts)]
    if len(_SPACES.sub("", "".join(texts))) < MIN_TEXT_CHARS:
        raise ParseError("no_text")

    starts: list[int] = []
    position = 0
    for text in texts:
        starts.append(position)
        position += len(text) + 2  # the blank line between pages
    return Extracted("\n\n".join(texts), tuple(starts))
