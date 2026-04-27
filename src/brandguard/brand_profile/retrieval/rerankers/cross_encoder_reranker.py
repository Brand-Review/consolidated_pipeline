"""Cross-encoder reranker wrapping sentence-transformers CrossEncoder.

Lazy-loads the model on first use; subsequent calls reuse the in-memory instance.
"""

from __future__ import annotations
import logging
from typing import List, Optional

from .base import Reranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 20):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[object] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker: {self.model_name}")
        self._model = CrossEncoder(self.model_name)

    def score_pairs(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        self._ensure_model()
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]
