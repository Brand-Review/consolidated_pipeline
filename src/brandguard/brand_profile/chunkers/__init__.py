"""Chunker factory."""

from __future__ import annotations
from typing import Dict

from .base import Chunk, Chunker
from .fixed_chunker import FixedChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker


_REGISTRY = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(strategy: str, chunking_config: Dict) -> Chunker:
    if strategy not in _REGISTRY:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                         f"Valid: {list(_REGISTRY)}")
    cls = _REGISTRY[strategy]
    per_strategy = chunking_config.get(strategy, {}) if isinstance(chunking_config, dict) else {}
    return cls(per_strategy)


__all__ = ["Chunk", "Chunker", "get_chunker"]
