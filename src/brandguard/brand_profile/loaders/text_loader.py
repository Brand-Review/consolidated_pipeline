"""Plain text loader — single section, single page."""

from __future__ import annotations
from typing import Union

from .base import RawDocument, Section, Page


def load_text(source: Union[str, bytes], filename: str = "") -> RawDocument:
    if isinstance(source, (bytes, bytearray)):
        text = source.decode("utf-8", errors="replace")
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

    return RawDocument(
        plaintext=text,
        sections=[Section(heading="General", level=1, char_start=0, char_end=len(text), page=1)],
        pages=[Page(page_number=1, char_start=0, char_end=len(text))],
        source_filename=filename,
        mime_type="text/plain",
    )
