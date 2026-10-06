"""Tiny PDFs made on the fly, so the tests need no binary files and no PDF library to write text."""

import io
from collections.abc import Sequence

from pypdf import PdfReader, PdfWriter


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """A PDF with one page per entry; each entry is the lines of text on that page."""
    objects: list[bytes] = []

    def add(body: str | bytes) -> int:
        objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objects)  # object numbers start at 1

    catalog = add("")  # filled in below, once the page objects are known
    pages_root = add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_numbers = []
    for lines in pages:
        content = (
            "BT /F1 12 Tf 72 740 Td 16 TL\n"
            + "".join(f"({_escape(line)}) Tj T*\n" for line in lines)
            + "ET"
        )
        stream = add(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        page_numbers.append(
            add(
                f"<< /Type /Page /Parent {pages_root} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>"
            )
        )
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_root} 0 R >>".encode("latin-1")
    objects[pages_root - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode("latin-1")
    )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
    out.write((trailer + f"startxref\n{xref}\n%%EOF\n").encode("latin-1"))
    return out.getvalue()


def blank_pdf(page_count: int = 2) -> bytes:
    """A PDF whose pages hold no text at all (what a scan looks like to a text reader)."""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(612, 792)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def encrypted(plain: bytes, password: str) -> bytes:
    """The same PDF protected by `password` (RC4, which needs no extra library to write)."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(plain)))
    writer.encrypt(password, algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
