"""
Text RAG — hybrid dense (E5) + sparse (BM25) Qdrant index, per-brand collection.

Collection naming: brand_{brand_id}_guidelines
Dense vector: named "dense", size=1024 (multilingual-e5-large), cosine.
Sparse vector: named "bm25" (Qdrant/bm25 via fastembed).

Indexing path orchestrates: dedup → dense+sparse embed → atomic upsert.
Retrieval path: `retrieve_hybrid` runs dense + sparse → server-side RRF fusion
→ optional rerank. Legacy `retrieve` / `retrieve_by_query` now wrap the hybrid
path and return concatenated text. Set RAG_DISABLE_HYBRID=1 to force the legacy
dense-only path (emergency rollback).
"""

from __future__ import annotations
import logging
import os
import uuid
from typing import List, Dict, Any, Optional

from .embeddings import EmbeddingService
from .deduper import Deduper
from .retrieval import HybridRetriever, RetrievalResult, load_retrieval_config
from .retrieval.rerankers import get_reranker

logger = logging.getLogger(__name__)

# Pre-defined retrieval queries per compliance check type
CHECK_TYPE_QUERIES: Dict[str, str] = {
    "color":       "color palette brand colors hex codes gradient forbidden colors",
    "typography":  "font typography typeface Bangla Bengali English approved fonts",
    "logo":        "logo usage placement position size corner height pixel background",
    "copywriting": "tone of voice copywriting brand voice formality warmth energy",
}

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "bm25"


def _hybrid_disabled() -> bool:
    return os.environ.get("RAG_DISABLE_HYBRID", "").strip().lower() in {"1", "true", "yes"}


class TextRAG:
    """
    Per-brand hybrid text vector store. Dense (E5) and sparse (BM25) live
    together in one Qdrant collection so upserts keep both indexes in sync.
    """

    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embeddings: Optional[EmbeddingService] = None,
        dedup_threshold: float = 0.95,
        dedup_enabled: bool = True,
    ):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY", None)
        self.embeddings = embeddings or EmbeddingService()
        self.dedup_threshold = dedup_threshold
        self.dedup_enabled = dedup_enabled
        self._qdrant: Optional[Any] = None
        self._hybrid: Optional[HybridRetriever] = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_qdrant(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            kwargs = {"url": self.qdrant_url, "check_compatibility": False}
            if self.qdrant_api_key:
                kwargs["api_key"] = self.qdrant_api_key
            self._qdrant = QdrantClient(**kwargs)
        return self._qdrant

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def collection_name(brand_id: str) -> str:
        return f"brand_{brand_id}_guidelines"

    def _ensure_collection(self, brand_id: str, dense_dim: int):
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, SparseIndexParams,
        )
        client = self._get_qdrant()
        name = self.collection_name(brand_id)
        existing = [c.name for c in client.get_collections().collections]

        if name in existing:
            if self._collection_is_hybrid(name):
                return
            logger.warning(
                f"Collection {name} exists with legacy (non-hybrid) schema — "
                f"recreating with dense+bm25 named vectors. Any previously indexed "
                f"chunks will be lost and must be re-ingested."
            )
            client.delete_collection(collection_name=name)

        client.create_collection(
            collection_name=name,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        logger.info(f"Created hybrid Qdrant collection: {name}")

    def _collection_is_hybrid(self, name: str) -> bool:
        try:
            info = self._get_qdrant().get_collection(collection_name=name)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors or {}
            has_dense = (
                isinstance(vectors, dict) and _DENSE_VECTOR_NAME in vectors
            )
            has_sparse = _SPARSE_VECTOR_NAME in sparse
            return bool(has_dense and has_sparse)
        except Exception as e:
            logger.warning(f"Could not inspect collection {name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(self, brand_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Embed and store text chunks for a brand.

        chunks: list of dicts with keys:
            text (required), section, page, chunk_index, char_count,
            strategy, source_filename.

        Returns:
            {
              "indexed": int,
              "skipped_dedup": int,
              "dropped": [ {chunk_index, matched_existing_id, similarity, source} ],
            }
        """
        if not chunks:
            return {"indexed": 0, "skipped_dedup": 0, "dropped": []}

        texts = [c["text"] for c in chunks]
        dense_vecs = self.embeddings.embed_dense_passages(texts)
        dense_dim = len(dense_vecs[0])

        self._ensure_collection(brand_id, dense_dim)
        collection = self.collection_name(brand_id)

        # Dedup (within batch + against existing collection)
        if self.dedup_enabled:
            deduper = Deduper(self._get_qdrant(), threshold=self.dedup_threshold)
            kept_chunks, kept_dense, dropped = deduper.filter_new_chunks(
                collection, chunks, dense_vecs
            )
        else:
            kept_chunks, kept_dense, dropped = chunks, dense_vecs, []

        if not kept_chunks:
            logger.info(
                f"Indexed 0 / {len(chunks)} chunks for brand {brand_id} (all duplicates)"
            )
            return {
                "indexed": 0,
                "skipped_dedup": len(dropped),
                "dropped": [self._drop_to_dict(d) for d in dropped],
            }

        # Sparse embed only the kept survivors
        kept_texts = [c["text"] for c in kept_chunks]
        sparse_vecs = self.embeddings.embed_sparse_passages(kept_texts)

        from qdrant_client.models import PointStruct
        points = []
        for chunk, dvec, svec in zip(kept_chunks, kept_dense, sparse_vecs):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    _DENSE_VECTOR_NAME: dvec,
                    _SPARSE_VECTOR_NAME: self.embeddings.sparse_to_qdrant(svec),
                },
                payload={
                    "text": chunk["text"],
                    "section": chunk.get("section", ""),
                    "page": int(chunk.get("page", 0)),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "char_count": int(chunk.get("char_count", len(chunk["text"]))),
                    "strategy": chunk.get("strategy", ""),
                    "source_filename": chunk.get("source_filename", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "brand_id": brand_id,
                },
            ))

        self._get_qdrant().upsert(collection_name=collection, points=points)
        logger.info(
            f"Indexed {len(points)} / {len(chunks)} chunks for brand {brand_id} "
            f"(skipped {len(dropped)} duplicates)"
        )
        return {
            "indexed": len(points),
            "skipped_dedup": len(dropped),
            "dropped": [self._drop_to_dict(d) for d in dropped],
        }

    def retrieve(self, brand_id: str, check_type: str, top_k: int = 3) -> str:
        query_text = CHECK_TYPE_QUERIES.get(check_type, check_type)
        return self.retrieve_by_query(brand_id, query_text, top_k)

    def retrieve_by_query(self, brand_id: str, query: str, top_k: int = 3) -> str:
        if _hybrid_disabled():
            return self._retrieve_dense_only(brand_id, query, top_k)
        try:
            result = self.retrieve_hybrid(brand_id, query, top_k=top_k)
            return result.as_concatenated_text()
        except Exception as e:
            logger.error(f"Hybrid retrieval failed for brand {brand_id} ({e}); falling back to dense-only.")
            return self._retrieve_dense_only(brand_id, query, top_k)

    def retrieve_hybrid(
        self,
        brand_id: str,
        query: str,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
        brand_override: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        retriever = self._get_hybrid_retriever(brand_override)
        return retriever.retrieve(
            brand_id=brand_id,
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

    def retrieve_hybrid_for_check(
        self,
        brand_id: str,
        check_type: str,
        top_k: Optional[int] = None,
        brand_override: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        query_text = CHECK_TYPE_QUERIES.get(check_type, check_type)
        return self.retrieve_hybrid(brand_id, query_text, top_k=top_k, brand_override=brand_override)

    def _get_hybrid_retriever(self, brand_override: Optional[Dict[str, Any]] = None) -> HybridRetriever:
        # If a per-brand override is supplied we rebuild; otherwise cache the shared retriever.
        if brand_override is not None:
            config = load_retrieval_config(brand_override=brand_override)
            reranker = get_reranker(config.reranker)
            return HybridRetriever(
                qdrant_client=self._get_qdrant(),
                embeddings=self.embeddings,
                config=config,
                reranker=reranker,
            )
        if self._hybrid is None:
            config = load_retrieval_config()
            reranker = get_reranker(config.reranker)
            self._hybrid = HybridRetriever(
                qdrant_client=self._get_qdrant(),
                embeddings=self.embeddings,
                config=config,
                reranker=reranker,
            )
        return self._hybrid

    def _retrieve_dense_only(self, brand_id: str, query: str, top_k: int) -> str:
        try:
            client = self._get_qdrant()
            name = self.collection_name(brand_id)

            existing = [c.name for c in client.get_collections().collections]
            if name not in existing:
                logger.warning(f"No guideline collection found for brand {brand_id}")
                return ""

            query_vector = self.embeddings.embed_dense_query(query)

            response = client.query_points(
                collection_name=name,
                query=query_vector,
                using=_DENSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
            )

            texts = [r.payload.get("text", "") for r in response.points if r.payload]
            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Text RAG dense-only retrieval failed for brand {brand_id}: {e}")
            return ""

    def delete_brand_collection(self, brand_id: str):
        try:
            client = self._get_qdrant()
            client.delete_collection(self.collection_name(brand_id))
        except Exception as e:
            logger.warning(f"Failed to delete guideline collection for brand {brand_id}: {e}")

    # ------------------------------------------------------------------

    @staticmethod
    def _drop_to_dict(d) -> Dict[str, Any]:
        return {
            "chunk_index": d.chunk_index,
            "matched_existing_id": d.matched_existing_id,
            "similarity": round(float(d.similarity), 4),
            "source": d.source,
        }
