"""Recursive character splitter — respects document sections, then descends separators."""

from __future__ import annotations
from typing import Dict, List

from .base import Chunk, Chunker
from ..loaders.base import RawDocument


class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, config: Dict):
        self.chunk_size = int(config.get("chunk_size", 800))
        self.min_chunk_size = int(config.get("min_chunk_size", 100))
        self.separators = list(config.get("separators", ["\n\n", "\n", ". ", " "]))

    def split(self, doc: RawDocument) -> List[Chunk]:
        text = doc.plaintext
        if not text:
            return []

        spans = self._section_spans(doc)
        chunks: List[Chunk] = []
        idx = 0
        for start, end, section, page in spans:
            segment = text[start:end].strip()
            if not segment:
                continue
            for piece in self._recursive_split(segment, self.separators):
                piece = piece.strip()
                if not piece or len(piece) < self.min_chunk_size and len(segment) > self.min_chunk_size:
                    continue
                chunks.append(Chunk(
                    text=piece,
                    section=section,
                    page=page,
                    char_count=len(piece),
                    strategy=self.name,
                    chunk_index=idx,
                    source_filename=doc.source_filename,
                ))
                idx += 1
        return chunks

    def _section_spans(self, doc: RawDocument):
        if not doc.sections:
            yield 0, len(doc.plaintext), "General", 1
            return
        sorted_secs = sorted(doc.sections, key=lambda s: s.char_start)
        first_start = sorted_secs[0].char_start
        if first_start > 0:
            yield 0, first_start, "General", doc.page_for_char(0)
        for i, s in enumerate(sorted_secs):
            end = sorted_secs[i + 1].char_start if i + 1 < len(sorted_secs) else len(doc.plaintext)
            yield s.char_start, end, s.heading, s.page

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, rest = separators[0], separators[1:]
        parts = text.split(sep) if sep else [text]
        out: List[str] = []
        buf = ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= self.chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(part) > self.chunk_size:
                    out.extend(self._recursive_split(part, rest))
                    buf = ""
                else:
                    buf = part
        if buf:
            out.append(buf)
        return out
