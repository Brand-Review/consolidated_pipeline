"""
EmbeddingService — single entry point for dense (E5) and sparse (Qdrant BM25)
encoders. Lazy-loads both models.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class EmbeddingService:
    def __init__(
        self,
        dense_model: str = "intfloat/multilingual-e5-large",
        sparse_model: str = "Qdrant/bm25",
        sparse_options: Optional[Dict[str, Any]] = None,
    ):
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.sparse_options = sparse_options or {}
        self._dense = None
        self._sparse = None

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    def _get_dense(self):
        if self._dense is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading dense model: {self.dense_model_name}")
            self._dense = SentenceTransformer(self.dense_model_name)
        return self._dense

    def embed_dense_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        return self._get_dense().encode(prefixed, normalize_embeddings=True).tolist()

    def embed_dense_query(self, text: str) -> List[float]:
        return self._get_dense().encode(
            _QUERY_PREFIX + text, normalize_embeddings=True
        ).tolist()

    # ------------------------------------------------------------------
    # Sparse (Qdrant BM25)
    # ------------------------------------------------------------------

    def _get_sparse(self):
        if self._sparse is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as e:
                raise ImportError(
                    "fastembed is required for sparse BM25 embeddings: pip install fastembed"
                ) from e
            logger.info(f"Loading sparse model: {self.sparse_model_name}")
            self._sparse = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse

    def embed_sparse_passages(self, texts: List[str]):
        if not texts:
            return []
        model = self._get_sparse()
        return list(model.embed(texts))

    def embed_sparse_query(self, text: str):
        model = self._get_sparse()
        gen = model.query_embed(text) if hasattr(model, "query_embed") else model.embed([text])
        return next(iter(gen))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sparse_to_qdrant(self, sparse_embedding) -> "SparseVector":
        from qdrant_client.models import SparseVector
        indices = list(getattr(sparse_embedding, "indices", []))
        values = list(getattr(sparse_embedding, "values", []))
        return SparseVector(indices=indices, values=values)

    @property
    def dense_dim(self) -> int:
        model = self._get_dense()
        return int(model.get_sentence_embedding_dimension())
