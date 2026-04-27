"""HTML loader — BeautifulSoup-based plaintext + heading extraction."""

from __future__ import annotations
import re
from typing import List, Union

from .base import RawDocument, Section, Page


def load_html(source: Union[str, bytes], filename: str = "") -> RawDocument:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from e

    if isinstance(source, (bytes, bytearray)):
        html = source.decode("utf-8", errors="replace")
    else:
        with open(source, "r", encoding="utf-8") as f:
            html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    plaintext_parts: List[str] = []
    sections: List[Section] = []
    cursor = 0

    for el in soup.descendants:
        if getattr(el, "name", None) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = el.get_text(" ", strip=True)
            if heading:
                level = int(el.name[1])
                sections.append(Section(
                    heading=heading,
                    level=level,
                    char_start=cursor,
                    char_end=cursor + len(heading),
                    page=1,
                ))
                plaintext_parts.append(heading + "\n")
                cursor += len(heading) + 1
        elif getattr(el, "name", None) in {"p", "li", "td", "th", "blockquote"}:
            text = el.get_text(" ", strip=True)
            if text:
                plaintext_parts.append(text + "\n")
                cursor += len(text) + 1

    plaintext = "".join(plaintext_parts)
    if not plaintext.strip():
        plaintext = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    for idx, s in enumerate(sections):
        s.char_end = sections[idx + 1].char_start if idx + 1 < len(sections) else len(plaintext)

    return RawDocument(
        plaintext=plaintext,
        sections=sections,
        pages=[Page(page_number=1, char_start=0, char_end=len(plaintext))],
        source_filename=filename,
        mime_type="text/html",
    )
