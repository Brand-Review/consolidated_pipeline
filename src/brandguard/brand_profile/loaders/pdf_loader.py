"""PDF loader — extracts plaintext + per-page offsets + font-size-based headings."""

from __future__ import annotations
import io
from typing import List, Union

from .base import RawDocument, Section, Page


def load_pdf(source: Union[str, bytes], filename: str = "") -> RawDocument:
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from e

    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(source)

    plaintext_parts: List[str] = []
    pages: List[Page] = []
    sections: List[Section] = []

    cursor = 0
    heading_sizes = _detect_heading_sizes(doc)

    for i, page in enumerate(doc):
        page_start = cursor
        page_text = page.get_text("text")
        plaintext_parts.append(page_text)

        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                sizes = [sp.get("size", 0) for sp in spans]
                max_size = max(sizes) if sizes else 0
                text = "".join(sp.get("text", "") for sp in spans).strip()
                if not text:
                    continue
                level = _heading_level_for_size(max_size, heading_sizes)
                if level > 0:
                    idx = page_text.find(text)
                    abs_start = page_start + (idx if idx >= 0 else 0)
                    sections.append(Section(
                        heading=text,
                        level=level,
                        char_start=abs_start,
                        char_end=abs_start + len(text),
                        page=i + 1,
                    ))

        cursor += len(page_text)
        pages.append(Page(page_number=i + 1, char_start=page_start, char_end=cursor))

    doc.close()

    plaintext = "".join(plaintext_parts)
    for idx, s in enumerate(sections):
        next_start = sections[idx + 1].char_start if idx + 1 < len(sections) else len(plaintext)
        s.char_end = max(s.char_end, next_start)

    return RawDocument(
        plaintext=plaintext,
        sections=sections,
        pages=pages,
        source_filename=filename,
        mime_type="application/pdf",
    )


def _detect_heading_sizes(doc) -> List[float]:
    sizes = []
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    sz = sp.get("size", 0)
                    if sz > 0:
                        sizes.append(sz)
    if not sizes:
        return []
    sizes.sort()
    median = sizes[len(sizes) // 2]
    unique = sorted({round(s, 1) for s in sizes if s > median * 1.15}, reverse=True)
    return unique[:3]


def _heading_level_for_size(size: float, heading_sizes: List[float]) -> int:
    for level, hs in enumerate(heading_sizes, start=1):
        if size >= hs - 0.05:
            return level
    return 0
