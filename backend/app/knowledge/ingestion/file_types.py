"""Registry of file types the ingestion pipeline can read.

This is the single source of truth: the scanner, the CLI (`--list-types`), the API
(`GET /api/v1/ingestion/file-types`) and the README all derive from it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.knowledge.ingestion.parsers import parse_html, parse_json, parse_text


@dataclass(frozen=True)
class FileType:
    name: str
    extensions: tuple[str, ...]
    description: str
    extract: Callable[[bytes], str]


FILE_TYPES: tuple[FileType, ...] = (
    FileType("Plain text", (".txt", ".log"), "UTF-8 text", parse_text),
    FileType("Markdown", (".md", ".markdown"), "Markdown; headings kept for chunking", parse_text),
    FileType("reStructuredText", (".rst",), "reStructuredText, read as text", parse_text),
    FileType("CSV / TSV", (".csv", ".tsv"), "Delimited text, read as text", parse_text),
    FileType("JSON", (".json",), "Must be valid JSON", parse_json),
    FileType("YAML", (".yaml", ".yml"), "YAML, read as text", parse_text),
    FileType(
        "HTML", (".html", ".htm"), "Visible text only; scripts and styles dropped", parse_html
    ),
)

_BY_EXTENSION: dict[str, FileType] = {ext: ft for ft in FILE_TYPES for ext in ft.extensions}


def supported_extensions() -> frozenset[str]:
    return frozenset(_BY_EXTENSION)


def file_type_for(suffix: str) -> FileType | None:
    return _BY_EXTENSION.get(suffix.lower())


def extract_text(suffix: str, data: bytes) -> str:
    """Extract text from file bytes using the reader registered for `suffix`."""
    file_type = file_type_for(suffix)
    if file_type is None:
        raise ValueError(f"unsupported file type: {suffix!r}")
    return file_type.extract(data)
