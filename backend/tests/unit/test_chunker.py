import re

import pytest

from app.knowledge.chunking.chunker import ChunkData, chunk_text

NEWLINE = re.compile(r"\r\n|\r|\n")


def _source_slice(text: str, chunk: ChunkData) -> str:
    return "\n".join(NEWLINE.split(text)[chunk.start_line - 1 : chunk.end_line])


def _paths(chunks: list[ChunkData]) -> list[str]:
    return [c.heading_path for c in chunks]


# --- headings and provenance ------------------------------------------------------------------

NESTED_MD = """Intro line before any heading.

# Project

Project overview.

## Setup

Install it.

### Windows

Use PowerShell.

## Usage

Run it.

# Appendix

Extra notes.
"""


def test_heading_paths_follow_nesting() -> None:
    chunks = chunk_text(NESTED_MD, markdown=True, chunk_size=1000, overlap=100)

    assert _paths(chunks) == [
        "",
        "Project",
        "Project > Setup",
        "Project > Setup > Windows",
        "Project > Usage",
        "Appendix",
    ]


def test_sibling_heading_resets_deeper_levels() -> None:
    text = "# A\n\n## B\n\nb text\n\n## C\n\nc text\n"

    chunks = chunk_text(text, markdown=True)

    assert _paths(chunks) == ["A > B", "A > C"]


def test_chunk_line_numbers_point_at_the_source() -> None:
    chunks = chunk_text(NESTED_MD, markdown=True, chunk_size=1000, overlap=100)

    for chunk in chunks:
        assert chunk.text == _source_slice(NESTED_MD, chunk)
    by_path = {c.heading_path: c for c in chunks}
    assert by_path["Project > Setup"].start_line == 7
    assert by_path["Project > Setup"].end_line == 9
    assert by_path["Project > Setup"].text.startswith("## Setup")


def test_indexes_are_sequential_from_zero() -> None:
    chunks = chunk_text(NESTED_MD, markdown=True)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_heading_with_no_body_is_dropped_but_kept_in_child_paths() -> None:
    text = "# Parent\n\n## Child\n\nbody\n"

    chunks = chunk_text(text, markdown=True)

    assert _paths(chunks) == ["Parent > Child"]


def test_closing_hashes_and_hash_in_title() -> None:
    text = "## C# Notes ##\n\nbody\n"

    chunks = chunk_text(text, markdown=True)

    assert chunks[0].heading_path == "C# Notes"


def test_plain_text_ignores_markdown_syntax() -> None:
    text = "# not a heading\n\n```\nnot a fence\n```\n"

    chunks = chunk_text(text, markdown=False)

    assert _paths(chunks) == [""]
    assert chunks[0].text == text.strip("\n")


def test_line_numbers_handle_crlf_and_cr() -> None:
    text = "# T\r\n\r\nline a\r\nline b\r\n"

    chunks = chunk_text(text, markdown=True)

    assert (chunks[0].start_line, chunks[0].end_line) == (1, 4)
    assert chunks[0].text == "# T\n\nline a\nline b"


# --- size and overlap -------------------------------------------------------------------------


def _long_section(paragraphs: int = 12, width: int = 60) -> str:
    body = "\n\n".join(f"Paragraph {i} " + "word " * (width // 5) for i in range(paragraphs))
    return f"# Big\n\n{body}\n"


def test_chunks_respect_size_limit() -> None:
    text = _long_section()

    chunks = chunk_text(text, markdown=True, chunk_size=300, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.heading_path == "Big" for c in chunks)


def test_split_chunks_overlap_within_a_section() -> None:
    text = _long_section()

    chunks = chunk_text(text, markdown=True, chunk_size=300, overlap=80)

    assert len(chunks) > 2
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start_line <= earlier.end_line, "consecutive chunks should overlap"
        assert later.start_line > earlier.start_line, "but must not repeat the whole chunk"
        assert later.end_line > earlier.end_line, "and must make progress"
    for chunk in chunks:
        assert chunk.text == _source_slice(text, chunk)


def test_overlap_zero_means_no_shared_lines() -> None:
    text = _long_section()

    chunks = chunk_text(text, markdown=True, chunk_size=300, overlap=0)

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start_line > earlier.end_line


def test_no_overlap_across_headings() -> None:
    text = "# One\n\nalpha alpha\n\n# Two\n\nbeta beta\n"

    chunks = chunk_text(text, markdown=True, chunk_size=1000, overlap=500)

    assert len(chunks) == 2
    assert chunks[0].end_line < chunks[1].start_line
    assert "alpha" not in chunks[1].text


def test_small_documents_become_one_chunk() -> None:
    text = "a short note\nwith two lines\n"

    chunks = chunk_text(text, markdown=False)

    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 2)


def test_oversize_paragraph_is_split_by_lines() -> None:
    text = "\n".join(f"row {i},{'x' * 20}" for i in range(40))  # one paragraph, no blank lines

    chunks = chunk_text(text, markdown=False, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    for chunk in chunks:
        assert chunk.text == _source_slice(text, chunk)
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    assert covered == set(range(1, 41))


def test_single_overlong_line_is_split_at_whitespace_with_overlap() -> None:
    line = " ".join(f"word{i}" for i in range(200))

    chunks = chunk_text(line, markdown=False, chunk_size=100, overlap=20)

    assert len(chunks) > 5
    assert all(len(c.text) <= 100 for c in chunks)
    assert all((c.start_line, c.end_line) == (1, 1) for c in chunks)
    assert all(not c.text.startswith("ord") for c in chunks), "cuts should fall on word boundaries"
    # Fragments overlap: each one starts with words that ended the previous fragment.
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        first_word = later.text.split()[0]
        assert first_word in earlier.text


def test_overlong_line_without_whitespace_is_hard_cut_and_loses_no_text() -> None:
    blob = "a" * 950

    chunks = chunk_text(blob, markdown=False, chunk_size=100, overlap=10)

    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].text + chunks[-1].text  # non-empty
    assert "".join(c.text for c in chunks).count("a") >= 950


# --- code fences ------------------------------------------------------------------------------


def test_heading_inside_code_fence_is_not_a_heading() -> None:
    text = "# Real\n\n```bash\n# just a comment\necho hi\n```\n"

    chunks = chunk_text(text, markdown=True)

    assert _paths(chunks) == ["Real"]
    assert "# just a comment" in chunks[0].text


def test_tilde_fence_and_longer_closing_fence() -> None:
    text = "# T\n\n~~~\n# not heading\n~~~~\n\n## After\n\nbody\n"

    chunks = chunk_text(text, markdown=True)

    assert _paths(chunks) == ["T", "T > After"]


def test_code_fence_is_never_split_even_when_oversized() -> None:
    fence = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(50)) + "\n```"
    text = f"# Code\n\nintro\n\n{fence}\n\noutro\n"

    chunks = chunk_text(text, markdown=True, chunk_size=200, overlap=40)

    holders = [c for c in chunks if "x0 = 0" in c.text]
    assert len(holders) == 1
    assert fence in holders[0].text
    assert len(holders[0].text) > 200


def test_small_code_fence_stays_with_neighbours() -> None:
    text = "# T\n\nRun:\n\n```\nls\n```\n\nDone.\n"

    chunks = chunk_text(text, markdown=True)

    assert len(chunks) == 1
    assert "```\nls\n```" in chunks[0].text


def test_unclosed_fence_runs_to_end_of_section_without_crashing() -> None:
    text = "# T\n\n```\nnever closed\nstill code\n"

    chunks = chunk_text(text, markdown=True)

    assert len(chunks) == 1
    assert chunks[0].text.endswith("still code")


# --- edge cases and validation ----------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "\n\n\n", "   \n\t\n"])
def test_blank_input_produces_no_chunks(text: str) -> None:
    assert chunk_text(text, markdown=False) == []
    assert chunk_text(text, markdown=True) == []


def test_chunking_is_deterministic() -> None:
    assert chunk_text(NESTED_MD, markdown=True) == chunk_text(NESTED_MD, markdown=True)


def test_every_non_blank_source_line_is_covered() -> None:
    text = _long_section(paragraphs=30)

    chunks = chunk_text(text, markdown=True, chunk_size=250, overlap=50)

    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    non_blank = {i + 1 for i, line in enumerate(NEWLINE.split(text)) if line.strip()}
    assert non_blank <= covered


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_size_or_overlap_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", markdown=False, chunk_size=size, overlap=overlap)
