"""Small Word, Excel and PowerPoint packages built on the fly (zip files of XML), good and hostile."""

import io
import zipfile
from collections.abc import Sequence
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def package(parts: dict[str, str | bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return out.getvalue()


def relationships(items: dict[str, str]) -> str:
    rows = "".join(
        f'<Relationship Id="{rid}" Type="{REL}/x" Target="{target}"/>'
        for rid, target in items.items()
    )
    return f'{XML}<Relationships xmlns="{REL}">{rows}</Relationships>'


# --- Word -----------------------------------------------------------------------------------------

STYLES = (
    f'{XML}<w:styles xmlns:w="{W}">'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Titre3"><w:name w:val="heading 3"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>'
    "</w:styles>"
)


def run(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(text: str, style: str | None = None, bullet: bool = False, raw: str | None = None) -> str:
    properties = ""
    if style:
        properties += f'<w:pStyle w:val="{style}"/>'
    if bullet:
        properties += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
    ppr = f"<w:pPr>{properties}</w:pPr>" if properties else ""
    return f"<w:p>{ppr}{raw if raw is not None else run(text)}</w:p>"


def table(rows: Sequence[Sequence[str]]) -> str:
    body = "".join(
        "<w:tr>" + "".join(f"<w:tc>{para(cell)}</w:tc>" for cell in row) + "</w:tr>" for row in rows
    )
    return f"<w:tbl>{body}</w:tbl>"


def docx(*blocks: str, styles: str | None = STYLES) -> bytes:
    document = (
        f'{XML}<w:document xmlns:w="{W}" xmlns:mc="{MC}"><w:body>{"".join(blocks)}'
        "<w:sectPr/></w:body></w:document>"
    )
    parts: dict[str, str | bytes] = {"word/document.xml": document}
    if styles is not None:
        parts["word/styles.xml"] = styles
    return package(parts)


# --- Excel ----------------------------------------------------------------------------------------


def column_letters(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


def xlsx(
    sheets: dict[str, Sequence[Sequence[object]]], states: dict[str, str] | None = None
) -> bytes:
    """Strings are stored as shared strings, True/False as booleans, `("f", 7)` as a cached
    formula result, `("i", "text")` as an inline string, other numbers as numbers."""
    strings: list[str] = []

    def shared(text: str) -> int:
        if text not in strings:
            strings.append(text)
        return strings.index(text)

    sheet_tags, rel_items, parts = [], {}, {}
    for number, (name, rows) in enumerate(sheets.items(), start=1):
        xml_rows = []
        for row_number, row in enumerate(rows, start=1):
            cells = []
            for column, value in enumerate(row):
                if value is None or value == "":
                    continue
                ref = f"{column_letters(column)}{row_number}"
                if isinstance(value, bool):
                    cells.append(f'<c r="{ref}" t="b"><v>{int(value)}</v></c>')
                elif isinstance(value, tuple) and value[0] == "f":
                    cells.append(f'<c r="{ref}"><f>SUM(A1:A2)</f><v>{value[1]}</v></c>')
                elif isinstance(value, tuple) and value[0] == "i":
                    cells.append(
                        f'<c r="{ref}" t="inlineStr"><is><t>{escape(value[1])}</t></is></c>'
                    )
                elif isinstance(value, str):
                    cells.append(f'<c r="{ref}" t="s"><v>{shared(value)}</v></c>')
                else:
                    cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
        parts[f"xl/worksheets/sheet{number}.xml"] = (
            f'{XML}<worksheet xmlns="{S}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
        )
        state = f' state="{states[name]}"' if states and name in states else ""
        sheet_tags.append(
            f'<sheet name="{escape(name)}" sheetId="{number}"{state} r:id="rId{number}"/>'
        )
        rel_items[f"rId{number}"] = f"worksheets/sheet{number}.xml"
    parts["xl/workbook.xml"] = (
        f'{XML}<workbook xmlns="{S}" xmlns:r="{R}"><sheets>{"".join(sheet_tags)}</sheets></workbook>'
    )
    parts["xl/_rels/workbook.xml.rels"] = relationships(rel_items)
    shared_xml = "".join(
        f"<si><r><t>{escape(text[: len(text) // 2])}</t></r><r><t>{escape(text[len(text) // 2 :])}</t></r></si>"
        for text in strings
    )  # rich text: every string is stored in two runs
    parts["xl/sharedStrings.xml"] = f'{XML}<sst xmlns="{S}">{shared_xml}</sst>'
    return package(parts)


# --- PowerPoint -----------------------------------------------------------------------------------


def _shape(paragraphs: Sequence[str], placeholder: str | None = None) -> str:
    ph = f'<p:ph type="{placeholder}"/>' if placeholder else ""
    text = "".join(f"<a:p><a:r><a:t>{escape(t)}</a:t></a:r></a:p>" for t in paragraphs)
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="s"/><p:cNvSpPr/><p:nvPr>{ph}</p:nvPr></p:nvSpPr>'
        f"<p:spPr/><p:txBody><a:bodyPr/>{text}</p:txBody></p:sp>"
    )


def _slide_table(rows: Sequence[Sequence[str]]) -> str:
    body = "".join(
        "<a:tr>"
        + "".join(
            f"<a:tc><a:txBody><a:p><a:r><a:t>{escape(c)}</a:t></a:r></a:p></a:txBody></a:tc>"
            for c in row
        )
        + "</a:tr>"
        for row in rows
    )
    return f"<p:graphicFrame><a:graphic><a:graphicData><a:tbl>{body}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"


def pptx(slides: Sequence[dict[str, object]], order: Sequence[int] | None = None) -> bytes:
    """Each slide: title, body (list of text), table (rows), notes. `order` lists which slide
    files come first in the presentation (to prove the file numbers are not the order)."""
    parts: dict[str, str | bytes] = {}
    for number, slide in enumerate(slides, start=1):
        shapes = ""
        if slide.get("title"):
            shapes += _shape([str(slide["title"])], "title")
        if slide.get("body"):
            shapes += _shape([str(t) for t in slide["body"]], "body")  # type: ignore[attr-defined]
        if slide.get("table"):
            shapes += _slide_table(slide["table"])  # type: ignore[arg-type]
        parts[f"ppt/slides/slide{number}.xml"] = (
            f'{XML}<p:sld xmlns:a="{A}" xmlns:p="{P}" xmlns:r="{R}"><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>'
        )
        if slide.get("notes"):
            parts[f"ppt/slides/_rels/slide{number}.xml.rels"] = relationships(
                {"rId1": f"../notesSlides/notesSlide{number}.xml"}
            )
            parts[f"ppt/notesSlides/notesSlide{number}.xml"] = (
                f'{XML}<p:notes xmlns:a="{A}" xmlns:p="{P}"><p:cSld><p:spTree>'
                f"{_shape([str(slide['notes']), str(number)])}</p:spTree></p:cSld></p:notes>"
            )
    sequence = list(order) if order is not None else list(range(1, len(slides) + 1))
    ids = "".join(f'<p:sldId id="{255 + n}" r:id="rId{n}"/>' for n in sequence)
    parts["ppt/presentation.xml"] = (
        f'{XML}<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>'
    )
    parts["ppt/_rels/presentation.xml.rels"] = relationships(
        {f"rId{n}": f"slides/slide{n}.xml" for n in range(1, len(slides) + 1)}
    )
    return package(parts)


# --- hostile files --------------------------------------------------------------------------------


def zip_bomb(size_mb: int = 20) -> bytes:
    """A small file that unpacks to size_mb megabytes of zeros."""
    return package({"word/document.xml": b"\0" * (size_mb * 1024 * 1024)})


def with_entry(name: str, content: bytes = b"x") -> bytes:
    """A package holding one entry under a chosen (possibly dangerous) name."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(zipfile.ZipInfo(name), content)
    return out.getvalue()


def with_doctype() -> bytes:
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>'
        f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>&lol2;</w:t></w:r></w:p></w:body></w:document>'
    )
    return package({"word/document.xml": bomb})


def marked_encrypted() -> bytes:
    """A package whose only entry says it is password-protected (the zip module cannot write that
    flag, so the bytes are changed in both headers)."""
    data = bytearray(package({"word/document.xml": b"x"}))
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        data[data.find(signature) + offset] |= 0x01
    return bytes(data)
