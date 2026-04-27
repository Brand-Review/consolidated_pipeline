"""
HybridRetriever — dense + sparse → server-side RRF/DBSF fusion → Reranked list.

Single Qdrant round-trip via `query_points(prefetch=[dense, sparse], query=FusionQuery(...))`.
Query vectors for dense and sparse are produced in parallel. Sparse-encoding
failures are caught and retrieval falls back to dense-only per `fallback.on_sparse_failure`.
The reranker is injected; if None (or disabled) the fused order is returned as-is.
"""

from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

from ..embeddings import EmbeddingService
from .config import RetrievalConfig
from .rerankers.base import Reranker
from .types import FusedCandidate, Reranked, RetrievalResult

logger = logging.getLogger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "bm25"


class HybridRetriever:
    def __init__(
        self,
        qdrant_client: Any,
        embeddings: EmbeddingService,
        config: RetrievalConfig,
        reranker: Optional[Reranker] = None,
    ):
        self.client = qdrant_client
        self.embeddings = embeddings
        self.config = config
        self.reranker = reranker

    def retrieve(
        self,
        brand_id: str,
        query: str,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
    ) -> RetrievalResult:
        started = time.monotonic()
        top_k = top_k or self.config.final_top_k
        pool = candidate_pool or self.config.candidate_pool_size

        collection = f"brand_{brand_id}_guidelines"
        if not self._collection_exists(collection):
            logger.warning(f"No guideline collection for brand {brand_id}")
            return RetrievalResult(
                query=query,
                latency_ms=(time.monotonic() - started) * 1000.0,
                fallback_reason="collection_missing",
            )

        dense_vec, sparse_vec, sparse_err = self._encode_parallel(query)

        used: List[str] = ["dense"]
        fallback_reason: Optional[str] = None
        candidates: List[FusedCandidate] = []

        if sparse_vec is not None:
            used.append("sparse")
            try:
                candidates = self._query_fused(collection, dense_vec, sparse_vec, pool)
            except Exception as e:
                logger.warning(f"Fused query failed ({e}); falling back to dense-only.")
                fallback_reason = "fused_query_failed"
                candidates = self._query_dense_only(collection, dense_vec, pool)
                used = ["dense"]
        else:
            # Sparse encoding failed; honor fallback policy.
            logger.warning(f"Sparse encoder failed ({sparse_err}); dense-only fallback.")
            if self.config.fallback.on_sparse_failure == "fail":
                raise RuntimeError(f"Sparse encoding failed and fallback=fail: {sparse_err}")
            fallback_reason = "sparse_unavailable"
            candidates = self._query_dense_only(collection, dense_vec, pool)

        reranked = self._rerank_or_passthrough(query, candidates)
        final = reranked[:top_k]

        return RetrievalResult(
            query=query,
            chunks=final,
            dense_top_score=None,
            sparse_top_score=None,
            fusion_weights=dict(self.config.fusion.weights),
            latency_ms=(time.monotonic() - started) * 1000.0,
            used_strategies=used + (["rerank"] if self._will_rerank() and not self._rerank_skipped else []),
            fallback_reason=fallback_reason or self._last_rerank_fallback,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collection_exists(self, name: str) -> bool:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            return name in existing
        except Exception as e:
            logger.warning(f"get_collections failed: {e}")
            return False

    def _encode_parallel(self, query: str) -> Tuple[List[float], Optional[Any], Optional[str]]:
        """Encode dense + sparse in parallel. Returns (dense, sparse_or_None, sparse_err_msg)."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            d_fut = ex.submit(self.embeddings.embed_dense_query, query)
            s_fut = ex.submit(self.embeddings.embed_sparse_query, query)
            dense_vec = d_fut.result()
            try:
                sparse_vec = s_fut.result()
                return dense_vec, sparse_vec, None
            except Exception as e:
                return dense_vec, None, str(e)

    def _query_fused(
        self,
        collection: str,
        dense_vec: List[float],
        sparse_embedding: Any,
        limit: int,
    ) -> List[FusedCandidate]:
        from qdrant_client.models import Prefetch, FusionQuery, Fusion

        sparse_qdrant = self.embeddings.sparse_to_qdrant(sparse_embedding)
        algo = self.config.fusion.algorithm
        fusion_enum = Fusion.DBSF if algo == "dbsf" else Fusion.RRF

        prefetch = [
            Prefetch(
                query=dense_vec,
                using=_DENSE_VECTOR_NAME,
                limit=self.config.per_source_top_k.dense,
            ),
            Prefetch(
                query=sparse_qdrant,
                using=_SPARSE_VECTOR_NAME,
                limit=self.config.per_source_top_k.sparse,
            ),
        ]
        resp = self.client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=fusion_enum),
            limit=limit,
            with_payload=True,
        )
        return [self._to_candidate(p) for p in resp.points]

    def _query_dense_only(
        self,
        collection: str,
        dense_vec: List[float],
        limit: int,
    ) -> List[FusedCandidate]:
        resp = self.client.query_points(
            collection_name=collection,
            query=dense_vec,
            using=_DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        return [self._to_candidate(p) for p in resp.points]

    @staticmethod
    def _to_candidate(p) -> FusedCandidate:
        payload = p.payload or {}
        return FusedCandidate(
            point_id=str(p.id),
            text=payload.get("text", ""),
            payload=payload,
            fused_score=float(p.score) if p.score is not None else 0.0,
        )

    # ------------------------------------------------------------------
    # Rerank plumbing
    # ------------------------------------------------------------------

    _rerank_skipped: bool = False
    _last_rerank_fallback: Optional[str] = None

    def _will_rerank(self) -> bool:
        return (
            self.reranker is not None
            and self.config.reranker.enabled
            and self.config.reranker.type != "none"
        )

    def _rerank_or_passthrough(
        self,
        query: str,
        candidates: List[FusedCandidate],
    ) -> List[Reranked]:
        self._rerank_skipped = False
        self._last_rerank_fallback = None
        if not candidates:
            return []
        if not self._will_rerank():
            return [Reranked(candidate=c, rerank_score=c.fused_score) for c in candidates]
        try:
            scores = self.reranker.score_pairs(query, [c.text for c in candidates])
            pairs = list(zip(candidates, scores))
            pairs.sort(key=lambda t: t[1], reverse=True)
            return [Reranked(candidate=c, rerank_score=float(s)) for c, s in pairs]
        except Exception as e:
            logger.warning(f"Reranker failed ({e}); keeping fused order.")
            if self.config.fallback.on_reranker_failure == "fail":
                raise
            self._rerank_skipped = True
            self._last_rerank_fallback = "reranker_failed"
            return [Reranked(candidate=c, rerank_score=c.fused_score) for c in candidates]
