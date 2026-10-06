"""Registry of file types the ingestion pipeline can read.

This is the single source of truth: the scanner, the CLI (`--list-types`), the API
(`GET /api/v1/ingestion/file-types`) and the README all derive from it.

A reader takes a file's bytes and returns `Extracted`: the text, and where its pages begin when
the format has pages (so a citation can say "page 12"). Readers for complicated formats run in a
separate, time-limited process (see sandbox.py) so a hostile or broken file cannot hang or crash
the application.
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import get_settings
from app.knowledge.ingestion.parsers import Extracted, parse_html, parse_json, parse_text
from app.knowledge.ingestion.sandbox import run_in_sandbox

Reader = Callable[[bytes], Extracted]


@dataclass(frozen=True)
class FileType:
    name: str
    extensions: tuple[str, ...]
    description: str
    extract: Reader
    # True when the reader runs in its own process, with a time limit.
    sandboxed: bool = False
    # True when the text has Markdown headings (`#`), which chunks use as their heading path.
    markdown: bool = False


def _plain(parser: Callable[[bytes], str]) -> Reader:
    """Wrap a reader that returns only text, for formats without pages."""

    def read(data: bytes) -> Extracted:
        return Extracted(parser(data))

    return read


def _in_sandbox(reader: str) -> Reader:
    """A reader that runs in its own process, with the time limit and options from the settings."""

    def read(data: bytes) -> Extracted:
        settings = get_settings()
        return run_in_sandbox(
            reader,
            data,
            timeout=settings.parser_timeout_seconds,
            options={
                "pdf_max_pages": settings.pdf_max_pages,
                "pdf_keep_headers_footers": settings.pdf_keep_headers_footers,
            },
        )

    return read


FILE_TYPES: tuple[FileType, ...] = (
    FileType("Plain text", (".txt", ".log"), "UTF-8 text", _plain(parse_text)),
    FileType(
        "Markdown",
        (".md", ".markdown"),
        "Markdown; headings kept for chunking",
        _plain(parse_text),
        markdown=True,
    ),
    FileType("reStructuredText", (".rst",), "reStructuredText, read as text", _plain(parse_text)),
    FileType("CSV / TSV", (".csv", ".tsv"), "Delimited text, read as text", _plain(parse_text)),
    FileType("JSON", (".json",), "Must be valid JSON", _plain(parse_json)),
    FileType("YAML", (".yaml", ".yml"), "YAML, read as text", _plain(parse_text)),
    FileType(
        "HTML",
        (".html", ".htm"),
        "Visible text only; scripts and styles dropped",
        _plain(parse_html),
    ),
    FileType(
        "PDF",
        (".pdf",),
        "Text of PDFs, cited by page. Scans (pictures of text) have no text to read",
        _in_sandbox("app.knowledge.ingestion.readers:read_pdf"),
        sandboxed=True,
    ),
    FileType(
        "Word",
        (".docx",),
        "Word documents: headings, lists and tables. The old .doc format is not read",
        _in_sandbox("app.knowledge.ingestion.office:read_docx"),
        sandboxed=True,
        markdown=True,
    ),
    FileType(
        "Excel",
        (".xlsx",),
        "Spreadsheets: each row as column-and-value pairs. Formulas are not run",
        _in_sandbox("app.knowledge.ingestion.office:read_xlsx"),
        sandboxed=True,
        markdown=True,
    ),
    FileType(
        "PowerPoint",
        (".pptx",),
        "Slides: titles, text, tables and speaker notes",
        _in_sandbox("app.knowledge.ingestion.office:read_pptx"),
        sandboxed=True,
        markdown=True,
    ),
)

_BY_EXTENSION: dict[str, FileType] = {ext: ft for ft in FILE_TYPES for ext in ft.extensions}


def supported_extensions() -> frozenset[str]:
    return frozenset(_BY_EXTENSION)


def file_type_for(suffix: str) -> FileType | None:
    return _BY_EXTENSION.get(suffix.lower())


def is_markdown(suffix: str) -> bool:
    """True if text read from this type has Markdown headings the chunker should use."""
    file_type = file_type_for(suffix)
    return file_type is not None and file_type.markdown


def extract_document(suffix: str, data: bytes) -> Extracted:
    """Read file bytes with the reader registered for `suffix`: text and page starts."""
    file_type = file_type_for(suffix)
    if file_type is None:
        raise ValueError(f"unsupported file type: {suffix!r}")
    return file_type.extract(data)


def extract_text(suffix: str, data: bytes) -> str:
    """Just the text of a file."""
    return extract_document(suffix, data).text
