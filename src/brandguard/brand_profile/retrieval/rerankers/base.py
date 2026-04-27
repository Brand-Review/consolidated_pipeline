"""Reranker abstract base. Concrete implementations live alongside."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):
    @abstractmethod
    def score_pairs(self, query: str, candidates: List[str]) -> List[float]:
        """Return a parallel list of relevance scores for each candidate."""
        raise NotImplementedError
