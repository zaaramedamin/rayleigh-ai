"""Split document text into chunks that keep provenance (heading path and line range).

Pure functions: no database or filesystem access.

Rules:
- Markdown is first split into sections at headings (headings inside code fences are ignored).
  Chunks never cross a section boundary, so there is no overlap across headings.
- Inside a section, text is packed paragraph by paragraph up to `chunk_size` characters.
  Normal chunks are an exact, contiguous slice of the source lines.
- Code fences are never split. A fence larger than `chunk_size` becomes one oversized chunk.
- A paragraph larger than `chunk_size` is split line by line; a single line larger than
  `chunk_size` is split at whitespace (hard cut if there is none) into fragments that overlap.
- When a section is split, each new chunk starts with up to `overlap` characters of whole
  lines from the end of the previous chunk.
"""

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

_NEWLINE = re.compile(r"\r\n|\r|\n")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class ChunkData:
    index: int
    text: str
    heading_path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive


@dataclass
class _Section:
    heading_path: str
    first_line_no: int  # 1-based number of lines[0]
    lines: list[str]


@dataclass(frozen=True)
class _Unit:
    start: int  # index into section.lines, inclusive
    end: int  # inclusive
    atomic: bool = False  # code fence: never split
    fragments: tuple[str, ...] = ()  # set for pieces of one over-long line


def chunk_text(
    text: str,
    *,
    markdown: bool,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkData]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    lines = _NEWLINE.split(text)
    sections = _split_sections(lines) if markdown else [_Section("", 1, lines)]

    chunks: list[ChunkData] = []
    for raw_section in sections:
        section = _trim(raw_section)
        if section is None or _is_heading_only(section, markdown):
            continue
        for start_line, end_line, chunk_text_ in _chunk_section(
            section, chunk_size, overlap, markdown
        ):
            chunks.append(
                ChunkData(
                    index=len(chunks),
                    text=chunk_text_,
                    heading_path=section.heading_path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    return chunks


# --- sections ---------------------------------------------------------------------------------


def _fence_closes(line: str, marker: str) -> bool:
    return bool(re.match(rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$", line))


def _split_sections(lines: list[str]) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section("", 1, [])
    stack: list[tuple[int, str]] = []
    fence: str | None = None

    for offset, line in enumerate(lines):
        if fence is not None:
            if _fence_closes(line, fence):
                fence = None
            current.lines.append(line)
            continue

        opened = _FENCE_OPEN.match(line)
        if opened:
            fence = opened.group(1)
            current.lines.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(current)
            current = _Section(" > ".join(t for _, t in stack), offset + 1, [line])
            continue

        current.lines.append(line)

    sections.append(current)
    return sections


def _trim(section: _Section) -> _Section | None:
    lines = section.lines
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    if start == end:
        return None
    return _Section(section.heading_path, section.first_line_no + start, lines[start:end])


def _is_heading_only(section: _Section, markdown: bool) -> bool:
    """A heading with no body adds nothing; its text survives in child chunks' heading_path."""
    return markdown and len(section.lines) == 1 and bool(_HEADING.match(section.lines[0]))


# --- units ------------------------------------------------------------------------------------


def _paragraph_units(lines: list[str], markdown: bool) -> list[_Unit]:
    units: list[_Unit] = []
    para_start: int | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        opened = _FENCE_OPEN.match(line) if markdown else None
        if opened:
            if para_start is not None:
                units.append(_Unit(para_start, i - 1))
                para_start = None
            j = i + 1
            while j < len(lines) and not _fence_closes(lines[j], opened.group(1)):
                j += 1
            end = min(j, len(lines) - 1)
            units.append(_Unit(i, end, atomic=True))
            i = end + 1
            continue
        if line.strip():
            if para_start is None:
                para_start = i
        elif para_start is not None:
            units.append(_Unit(para_start, i - 1))
            para_start = None
        i += 1
    if para_start is not None:
        units.append(_Unit(para_start, len(lines) - 1))
    return units


def _split_long_line(text: str, size: int, overlap: int) -> list[str]:
    """Split one over-long line at whitespace where possible, with `overlap` chars of overlap."""
    pieces: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        end = min(pos + size, length)
        if end < length:
            cut = max(text.rfind(" ", pos + size // 2, end), text.rfind("\t", pos + size // 2, end))
            if cut > pos:
                end = cut
        piece = text[pos:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        next_pos = end - overlap if end - overlap > pos else end
        while next_pos < length and text[next_pos].isspace():
            next_pos += 1
        pos = next_pos
    return pieces


# --- packing ----------------------------------------------------------------------------------


def _chunk_section(
    section: _Section, size: int, overlap: int, markdown: bool
) -> list[tuple[int, int, str]]:
    """Return (start_line, end_line, text) triples for one section."""
    lines = section.lines
    base = section.first_line_no

    def slice_len(a: int, b: int) -> int:
        return sum(len(lines[k]) for k in range(a, b + 1)) + (b - a)

    def slice_text(a: int, b: int) -> str:
        return "\n".join(lines[a : b + 1])

    # Expand oversize paragraphs into line units, and oversize lines into fragments.
    units: list[_Unit] = []
    for unit in _paragraph_units(lines, markdown):
        if unit.atomic or slice_len(unit.start, unit.end) <= size:
            units.append(unit)
            continue
        for k in range(unit.start, unit.end + 1):
            if len(lines[k]) <= size:
                units.append(_Unit(k, k))
            else:
                units.append(
                    _Unit(k, k, fragments=tuple(_split_long_line(lines[k], size, overlap)))
                )

    out: list[tuple[int, int, str]] = []
    cur: tuple[int, int] | None = None  # (start, end) line indexes of the chunk being built
    prev: tuple[int, int] | None = None  # last emitted line-based chunk, for overlap

    def flush() -> None:
        nonlocal cur, prev
        if cur is not None:
            out.append((base + cur[0], base + cur[1], slice_text(*cur)))
            prev = cur
            cur = None

    def overlap_start(unit_start: int) -> int:
        """Earliest line (within the previous chunk) giving <= `overlap` chars of overlap."""
        if prev is None or overlap == 0:
            return unit_start
        prev_start, prev_end = prev
        best = unit_start
        for k in range(prev_end, prev_start, -1):  # never re-emit the whole previous chunk
            if slice_len(k, unit_start - 1) > overlap:
                break
            best = k
        while best < unit_start and not lines[best].strip():
            best += 1
        return best

    for unit in units:
        if unit.fragments:
            flush()
            prev = None
            for fragment in unit.fragments:
                out.append((base + unit.start, base + unit.start, fragment))
            continue

        if cur is not None and slice_len(cur[0], unit.end) <= size:
            cur = (cur[0], unit.end)
            continue

        flush()
        start = overlap_start(unit.start)
        if start != unit.start and slice_len(start, unit.end) > size:
            start = unit.start
        cur = (start, unit.end)
        if unit.atomic and slice_len(unit.start, unit.end) > size:
            flush()
            prev = None  # no overlap into or out of an oversized code block

    flush()
    return out
