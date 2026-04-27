"""Fixed-size chunker with overlap."""

from __future__ import annotations
from typing import Dict, List

from .base import Chunk, Chunker
from ..loaders.base import RawDocument


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, config: Dict):
        self.chunk_size = int(config.get("chunk_size", 800))
        self.overlap = int(config.get("overlap", 100))
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def split(self, doc: RawDocument) -> List[Chunk]:
        text = doc.plaintext
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        idx = 0
        cursor = 0
        while cursor < len(text):
            end = min(cursor + self.chunk_size, len(text))
            piece = text[cursor:end].strip()
            if piece:
                chunks.append(Chunk(
                    text=piece,
                    section=doc.section_for_char(cursor),
                    page=doc.page_for_char(cursor),
                    char_count=len(piece),
                    strategy=self.name,
                    chunk_index=idx,
                    source_filename=doc.source_filename,
                ))
                idx += 1
            if end >= len(text):
                break
            cursor += step
        return chunks
