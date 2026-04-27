"""Shared Chunk dataclass and abstract Chunker."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..loaders.base import RawDocument


@dataclass
class Chunk:
    text: str
    section: str
    page: int
    char_count: int
    strategy: str
    chunk_index: int
    source_filename: str


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def split(self, doc: RawDocument) -> List[Chunk]:
        ...
