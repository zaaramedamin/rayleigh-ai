"""Readers for Word (.docx), Excel (.xlsx) and PowerPoint (.pptx), using only the standard library.

These files are zip archives of XML. Reading them directly (rather than with a library that
needs a compiled XML package) keeps the program free of files Windows might block, and lets the
safety rules be enforced here:

- the archive is never extracted to disk; only the few parts that hold text are read in memory;
- limits on the number of entries, the size of each part, the total size read and the compression
  ratio, so a zip bomb is refused instead of expanded;
- an entry whose name could escape a folder (an absolute path, ``..``, a drive letter, a backslash)
  makes the whole archive unsafe;
- XML with a DOCTYPE or entity declaration is refused (Office files never have one, and it is how
  entity-expansion attacks work);
- macros are never read or run, links are never followed, and formulas are never evaluated (the
  value the file saved with each cell is read).

The readers run in the sandbox and report these reasons: ``corrupt``, ``unsafe_archive``,
``encrypted``, ``old_format`` (the old binary .doc/.xls/.ppt, or a password-protected file) and
``no_text``.
"""

import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import PurePosixPath
from xml.etree import ElementTree

from app.knowledge.ingestion.parsers import Extracted, ParseError

MAX_ENTRIES = 5000
MAX_PART_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
# A part this much bigger than it is stored, and bigger than a megabyte, is a bomb.
MAX_COMPRESSION_RATIO = 200
MAX_SHEET_ROWS = 5000
MAX_SHEET_COLUMNS = 200
MIN_TEXT_CHARS = 1

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

_OLE_HEADER = b"\xd0\xcf\x11\xe0"
_DTD = re.compile(rb"<!(?:DOCTYPE|ENTITY)", re.IGNORECASE)
_DRIVE = re.compile(r"^[A-Za-z]:")
_HEADING_STYLE_ID = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_BLANKS = re.compile(r"[ \t\r\f\v]+")


def _q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


# --- the archive ----------------------------------------------------------------------------------


class Package:
    """An Office file opened for reading, with its safety limits checked up front."""

    def __init__(self, data: bytes) -> None:
        if data[:4] == _OLE_HEADER:
            raise ParseError("old_format")
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise ParseError("corrupt") from exc
        infos = self._zip.infolist()
        if len(infos) > MAX_ENTRIES:
            raise ParseError("unsafe_archive")
        self._infos: dict[str, zipfile.ZipInfo] = {}
        declared = 0
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name).parts
            if name.startswith("/") or "\\" in name or ".." in parts or _DRIVE.match(name):
                raise ParseError("unsafe_archive")
            if info.flag_bits & 0x1:
                raise ParseError("encrypted")
            declared += info.file_size
            too_big = info.file_size > MAX_PART_BYTES
            ratio = info.file_size / max(1, info.compress_size)
            if too_big or (info.file_size > 1_000_000 and ratio > MAX_COMPRESSION_RATIO):
                raise ParseError("unsafe_archive")
            self._infos[name] = info
        if declared > MAX_TOTAL_BYTES:
            raise ParseError("unsafe_archive")
        self._read_so_far = 0

    def names(self) -> list[str]:
        return list(self._infos)

    def has(self, name: str) -> bool:
        return name in self._infos

    def read(self, name: str) -> bytes | None:
        """The bytes of one part, or None if the archive has no such part."""
        info = self._infos.get(name)
        if info is None:
            return None
        try:
            with self._zip.open(info) as part:
                data = part.read(MAX_PART_BYTES + 1)  # the declared size may be a lie
        except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, EOFError) as exc:
            raise ParseError("corrupt") from exc
        self._read_so_far += len(data)
        if len(data) > MAX_PART_BYTES or self._read_so_far > MAX_TOTAL_BYTES:
            raise ParseError("unsafe_archive")
        return data

    def xml(self, name: str) -> ElementTree.Element | None:
        data = self.read(name)
        return None if data is None else parse_xml(data)

    def relationships(self, part: str) -> dict[str, str]:
        """Relationship id -> full part name, for the relationships file that belongs to `part`."""
        folder, _, leaf = part.rpartition("/")
        root = self.xml(f"{folder}/_rels/{leaf}.rels" if folder else f"_rels/{leaf}.rels")
        if root is None:
            return {}
        found = {}
        for item in root.iter(_q(REL, "Relationship")):
            rid, target = item.get("Id"), item.get("Target")
            if not rid or not target or item.get("TargetMode") == "External":
                continue  # nothing outside the file is ever followed
            found[rid] = _resolve(folder, target)
        return found


def _resolve(folder: str, target: str) -> str:
    """A relationship target as a part name inside the archive."""
    if target.startswith("/"):
        return target.lstrip("/")
    parts: list[str] = []
    for piece in f"{folder}/{target}".split("/") if folder else target.split("/"):
        if piece == "..":
            if parts:
                parts.pop()
        elif piece and piece != ".":
            parts.append(piece)
    return "/".join(parts)


def parse_xml(data: bytes) -> ElementTree.Element:
    # Office files never have a DTD. One means an attempt at entity expansion; do not parse it.
    if _DTD.search(data):
        raise ParseError("unsafe_archive")
    try:
        return ElementTree.fromstring(data)
    except (ElementTree.ParseError, ValueError) as exc:
        raise ParseError("corrupt") from exc


def _clean(text: str) -> str:
    return _BLANKS.sub(" ", text).strip()


def _finish(blocks: list[str], page_starts: tuple[int, ...] | None = None) -> Extracted:
    text = "\n\n".join(block for block in blocks if block.strip())
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ParseError("no_text")
    return Extracted(text, page_starts)


# --- Word -----------------------------------------------------------------------------------------


def _text_of(element: ElementTree.Element, text_tag: str, break_tags: tuple[str, ...]) -> str:
    """The text under `element` in reading order. Text that exists only as a fallback copy of a
    drawing (marked-up twice by Word) is read once."""
    pieces: list[str] = []

    def walk(node: ElementTree.Element) -> None:
        for child in node:
            if child.tag == _q(MC, "Fallback"):
                continue
            if child.tag == text_tag:
                pieces.append(child.text or "")
            elif child.tag in break_tags:
                pieces.append("\n")
            elif child.tag == _q(W, "tab"):
                pieces.append("\t")
            else:
                walk(child)

    walk(element)
    return "".join(pieces)


TITLE_LEVEL = 0


def _word_styles(package: Package) -> dict[str, int]:
    """Style id -> heading level (1 to 6) for the styles that are headings; 0 for a title."""
    levels: dict[str, int] = {}
    root = package.xml("word/styles.xml")
    if root is None:
        return levels
    for style in root.iter(_q(W, "style")):
        style_id = style.get(_q(W, "styleId"))
        name_element = style.find(_q(W, "name"))
        name = (name_element.get(_q(W, "val")) if name_element is not None else "") or ""
        if not style_id:
            continue
        match = _HEADING_STYLE_ID.match(name.strip()) or _HEADING_STYLE_ID.match(style_id)
        if match:
            levels[style_id] = min(6, max(1, int(match.group(1))))
        elif name.strip().lower() == "title" or style_id.lower() == "title":
            levels[style_id] = TITLE_LEVEL
    return levels


def _word_paragraph(
    paragraph: ElementTree.Element, headings: dict[str, int]
) -> tuple[int | None, str]:
    """(heading level, text). The level is None for ordinary text; a list item starts with '- '."""
    text = _clean(_text_of(paragraph, _q(W, "t"), (_q(W, "br"), _q(W, "cr"))))
    if not text:
        return None, ""
    properties = paragraph.find(_q(W, "pPr"))
    if properties is not None:
        style = properties.find(_q(W, "pStyle"))
        style_id = style.get(_q(W, "val")) if style is not None else None
        if style_id in headings:
            return headings[style_id], text.replace("\n", " ")
        if properties.find(_q(W, "numPr")) is not None:
            return None, f"- {text}"
    return None, text


def _table_lines(rows: list[list[str]]) -> str:
    """A table as lines of 'column: value' pairs, so a row is understood without its header."""
    rows = [[cell for cell in row] for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(cell for cell in rows[0] if cell)
    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [
            f"{header[i] if i < len(header) and header[i] else f'column {i + 1}'}: {cell}"
            for i, cell in enumerate(row)
            if cell
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _word_table(table: ElementTree.Element) -> str:
    rows = []
    for row in table.findall(_q(W, "tr")):
        cells = []
        for cell in row.findall(_q(W, "tc")):
            texts = [_word_paragraph(p, {})[1] for p in cell.iter(_q(W, "p"))]
            cells.append(_clean(" ".join(t.removeprefix("- ") for t in texts if t)))
        rows.append(cells)
    return _table_lines(rows)


def _word_blocks(
    parent: ElementTree.Element, headings: dict[str, int]
) -> Iterator[tuple[int | None, str]]:
    for child in parent:
        if child.tag == _q(W, "p"):
            yield _word_paragraph(child, headings)
        elif child.tag == _q(W, "tbl"):
            yield None, _word_table(child)
        elif child.tag == _q(W, "sdt"):  # a content control wraps ordinary paragraphs
            content = child.find(_q(W, "sdtContent"))
            if content is not None:
                yield from _word_blocks(content, headings)


def read_docx(data: bytes) -> Extracted:
    """Paragraphs, headings (as Markdown, so chunks get heading paths), lists and tables."""
    package = Package(data)
    root = package.xml("word/document.xml")
    if root is None:
        raise ParseError("corrupt")
    body = root.find(_q(W, "body"))
    if body is None:
        raise ParseError("corrupt")
    blocks = list(_word_blocks(body, _word_styles(package)))
    # A title is the top level, so when there is one the headings nest below it.
    shift = 1 if any(level == TITLE_LEVEL for level, _ in blocks) else 0
    return _finish(
        [
            text
            if level is None
            else f"{'#' * (1 if level == TITLE_LEVEL else min(6, level + shift))} {text}"
            for level, text in blocks
        ]
    )


# --- Excel ----------------------------------------------------------------------------------------


def _column_index(reference: str) -> int:
    """'A1' -> 0, 'B7' -> 1, 'AA3' -> 26."""
    letters = re.match(r"[A-Za-z]+", reference)
    if letters is None:
        return 0
    index = 0
    for char in letters.group(0).upper():
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _shared_strings(package: Package) -> list[str]:
    root = package.xml("xl/sharedStrings.xml")
    if root is None:
        return []
    strings = []
    for item in root.findall(_q(S, "si")):
        parts = []
        for node in item.iter():
            if node.tag == _q(S, "t"):
                parts.append(node.text or "")
        strings.append("".join(parts))
    return strings


def _cell_value(cell: ElementTree.Element, strings: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        inline = cell.find(_q(S, "is"))
        return (
            _clean("".join(t.text or "" for t in inline.iter(_q(S, "t"))))
            if inline is not None
            else ""
        )
    value = cell.find(_q(S, "v"))
    if value is None or value.text is None:
        return ""
    text = value.text
    if kind == "s":
        try:
            return _clean(strings[int(text)])
        except (ValueError, IndexError):
            return ""
    if kind == "b":
        return "yes" if text.strip() == "1" else "no"
    if kind == "e":
        return ""  # a formula error such as #DIV/0! says nothing about the content
    return _clean(text)


def _sheet_rows(data: bytes, strings: list[str]) -> tuple[list[list[str]], bool]:
    """The non-empty rows of one sheet (cells by column), and whether it was cut at the row cap."""
    if _DTD.search(data):
        raise ParseError("unsafe_archive")
    rows: list[list[str]] = []
    truncated = False
    try:
        for _event, element in ElementTree.iterparse(io.BytesIO(data), events=("end",)):
            if element.tag != _q(S, "row"):
                continue
            cells: dict[int, str] = {}
            for cell in element.findall(_q(S, "c")):
                column = _column_index(cell.get("r") or "")
                if column >= MAX_SHEET_COLUMNS:
                    continue
                value = _cell_value(cell, strings)
                if value:
                    cells[column] = value
            element.clear()
            if not cells:
                continue
            if len(rows) >= MAX_SHEET_ROWS:
                truncated = True
                break
            width = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(width)])
    except ElementTree.ParseError as exc:
        raise ParseError("corrupt") from exc
    return rows, truncated


def read_xlsx(data: bytes) -> Extracted:
    """Each sheet is a section; each row is 'column: value' pairs under the first row's headings.

    Formulas are not evaluated: the value the file saved with the cell is read.
    """
    package = Package(data)
    workbook = package.xml("xl/workbook.xml")
    if workbook is None:
        raise ParseError("corrupt")
    targets = package.relationships("xl/workbook.xml")
    strings = _shared_strings(package)
    blocks: list[str] = []
    for sheet in workbook.iter(_q(S, "sheet")):
        if sheet.get("state") == "veryHidden":
            continue
        name = _clean(sheet.get("name") or "Sheet")
        part = targets.get(sheet.get(_q(R, "id")) or "")
        sheet_data = package.read(part) if part else None
        if sheet_data is None:
            continue
        rows, truncated = _sheet_rows(sheet_data, strings)
        body = _table_lines(rows)
        if not body:
            continue
        note = f"\n(only the first {MAX_SHEET_ROWS} rows are read)" if truncated else ""
        blocks.append(f"## {name}\n\n{body}{note}")
    return _finish(blocks)


# --- PowerPoint -----------------------------------------------------------------------------------


def _slide_number(part: str) -> int:
    match = re.search(r"(\d+)\.xml$", part)
    return int(match.group(1)) if match else 0


def _slides_in_order(package: Package) -> list[str]:
    """The slide parts in the order of the presentation (files keep old numbers when slides are
    moved, so the numbers in the file names are only a fallback)."""
    on_disk = [n for n in package.names() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    presentation = package.xml("ppt/presentation.xml")
    if presentation is not None:
        targets = package.relationships("ppt/presentation.xml")
        ordered = [
            targets[rid]
            for item in presentation.iter(_q(P, "sldId"))
            if (rid := item.get(_q(R, "id"))) in targets
        ]
        ordered = [part for part in ordered if package.has(part)]
        if ordered:
            return ordered
    return sorted(on_disk, key=_slide_number)


def _drawing_paragraphs(root: ElementTree.Element) -> list[str]:
    paragraphs = []
    for paragraph in root.iter(_q(A, "p")):
        text = _clean(_text_of(paragraph, _q(A, "t"), (_q(A, "br"),)))
        if text:
            paragraphs.append(text)
    return paragraphs


def _slide_title(root: ElementTree.Element) -> list[str]:
    """The paragraphs of the slide's title placeholder, if it has one."""
    for shape in root.iter(_q(P, "sp")):
        placeholder = shape.find(f"{_q(P, 'nvSpPr')}/{_q(P, 'nvPr')}/{_q(P, 'ph')}")
        if placeholder is not None and placeholder.get("type") in ("title", "ctrTitle"):
            return _drawing_paragraphs(shape)
    return []


def _slide_tables(root: ElementTree.Element) -> list[str]:
    tables = []
    for table in root.iter(_q(A, "tbl")):
        rows = [
            [_clean(" ".join(_drawing_paragraphs(cell))) for cell in row.findall(_q(A, "tc"))]
            for row in table.findall(_q(A, "tr"))
        ]
        text = _table_lines(rows)
        if text:
            tables.append(text)
    return tables


def read_pptx(data: bytes) -> Extracted:
    """One section per slide (its title becomes the heading), with the speaker notes."""
    package = Package(data)
    slides = _slides_in_order(package)
    if not slides:
        raise ParseError("corrupt")
    blocks: list[str] = []
    for position, part in enumerate(slides, start=1):
        root = package.xml(part)
        if root is None:
            continue
        title_parts = _slide_title(root)
        title = " ".join(title_parts)
        heading = f"# Slide {position}" + (f"\n\n## {title}" if title else "")
        body = [p for p in _drawing_paragraphs_outside_tables(root) if p not in title_parts]
        body += _slide_tables(root)
        notes = ""
        for target in package.relationships(part).values():
            if "notesSlides/" in target:
                notes_root = package.xml(target)
                if notes_root is not None:
                    notes = " ".join(p for p in _drawing_paragraphs(notes_root) if not p.isdigit())
        pieces = [heading, *body]
        if notes:
            pieces.append(f"Notes: {notes}")
        blocks.append("\n\n".join(pieces))
    return _finish(blocks)


def _drawing_paragraphs_outside_tables(root: ElementTree.Element) -> list[str]:
    """Slide text from shapes, not from tables (those are read as rows)."""
    inside_tables = {id(p) for table in root.iter(_q(A, "tbl")) for p in table.iter(_q(A, "p"))}
    paragraphs = []
    for paragraph in root.iter(_q(A, "p")):
        if id(paragraph) in inside_tables:
            continue
        text = _clean(_text_of(paragraph, _q(A, "t"), (_q(A, "br"),)))
        if text:
            paragraphs.append(text)
    return paragraphs
