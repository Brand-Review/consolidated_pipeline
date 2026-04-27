"""
RawDocument dataclass and shared helpers for document loaders.

Loaders convert raw uploaded bytes into a normalized RawDocument with plaintext,
section metadata (heading hierarchy), and page offsets so chunkers can attribute
every chunk to a section + page.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Section:
    heading: str
    level: int
    char_start: int
    char_end: int
    page: int


@dataclass
class Page:
    page_number: int
    char_start: int
    char_end: int


@dataclass
class RawDocument:
    plaintext: str
    sections: List[Section] = field(default_factory=list)
    pages: List[Page] = field(default_factory=list)
    source_filename: str = ""
    mime_type: str = ""
    raw_s3_key: Optional[str] = None

    def page_for_char(self, char_idx: int) -> int:
        for p in self.pages:
            if p.char_start <= char_idx < p.char_end:
                return p.page_number
        return self.pages[-1].page_number if self.pages else 1

    def section_for_char(self, char_idx: int) -> str:
        current = "General"
        for s in self.sections:
            if s.char_start <= char_idx:
                current = s.heading
            else:
                break
        return current
