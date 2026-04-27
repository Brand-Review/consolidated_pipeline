"""Markdown loader — uses markdown-it-py AST to extract plaintext + section headings."""

from __future__ import annotations
from typing import List, Union

from .base import RawDocument, Section, Page


def load_markdown(source: Union[str, bytes], filename: str = "") -> RawDocument:
    if isinstance(source, (bytes, bytearray)):
        text = source.decode("utf-8", errors="replace")
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

    sections = _extract_sections(text)
    plaintext = text

    return RawDocument(
        plaintext=plaintext,
        sections=sections,
        pages=[Page(page_number=1, char_start=0, char_end=len(plaintext))],
        source_filename=filename,
        mime_type="text/markdown",
    )


def _extract_sections(text: str) -> List[Section]:
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return _fallback_heading_scan(text)

    md = MarkdownIt()
    tokens = md.parse(text)

    sections: List[Section] = []
    lines = text.split("\n")
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            heading = inline.content.strip() if inline else ""
            if tok.map:
                start_line = tok.map[0]
                char_start = line_offsets[start_line] if start_line < len(line_offsets) else 0
            else:
                char_start = 0
            sections.append(Section(
                heading=heading,
                level=level,
                char_start=char_start,
                char_end=char_start + len(heading),
                page=1,
            ))
        i += 1

    for idx, s in enumerate(sections):
        next_start = sections[idx + 1].char_start if idx + 1 < len(sections) else len(text)
        s.char_end = next_start
    return sections


def _fallback_heading_scan(text: str) -> List[Section]:
    sections: List[Section] = []
    cursor = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[level:].strip()
            if heading and 1 <= level <= 6:
                sections.append(Section(
                    heading=heading,
                    level=level,
                    char_start=cursor,
                    char_end=cursor + len(line),
                    page=1,
                ))
        cursor += len(line) + 1
    for idx, s in enumerate(sections):
        s.char_end = sections[idx + 1].char_start if idx + 1 < len(sections) else len(text)
    return sections
