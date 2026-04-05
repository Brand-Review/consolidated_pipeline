"""
Text RAG
Embeds brand guideline text chunks with multilingual-E5 and stores/retrieves
them from Qdrant (one collection per brand).

Collection naming: brand_{brand_id}_guidelines
Vector dim: 1024 (multilingual-e5-large) or 768 (multilingual-e5-base)
"""

from __future__ import annotations
import logging
import os
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Query prefix required by E5 instruction-following models
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

# Pre-defined retrieval queries per compliance check type
CHECK_TYPE_QUERIES: Dict[str, str] = {
    "color":       "color palette brand colors hex codes gradient forbidden colors",
    "typography":  "font typography typeface Bangla Bengali English approved fonts",
    "logo":        "logo usage placement position size corner height pixel background",
    "copywriting": "tone of voice copywriting brand voice formality warmth energy",
}


class TextRAG:
    """
    Manages per-brand text vector collections in Qdrant using multilingual-E5 embeddings.
    """

    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        model_name: str = "intfloat/multilingual-e5-large",
    ):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY", None)
        self.model_name = model_name
        self._model = None
        self._qdrant: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

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

    def _ensure_collection(self, brand_id: str, vector_size: int):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_qdrant()
        name = self.collection_name(brand_id)
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(self, brand_id: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and store text chunks for a brand.
        chunks: [{"text": str, "section": str, "page": int}]
        Returns number of points indexed.
        """
        if not chunks:
            return 0

        model = self._get_model()
        client = self._get_qdrant()

        # Embed with passage prefix (E5 requirement)
        texts = [_PASSAGE_PREFIX + c["text"] for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()

        self._ensure_collection(brand_id, len(vectors[0]))

        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk["text"],
                    "section": chunk.get("section", ""),
                    "page": chunk.get("page", 0),
                    "brand_id": brand_id,
                },
            )
            for vec, chunk in zip(vectors, chunks)
        ]

        client.upsert(collection_name=self.collection_name(brand_id), points=points)
        logger.info(f"Indexed {len(points)} chunks for brand {brand_id}")
        return len(points)

    def retrieve(self, brand_id: str, check_type: str, top_k: int = 3) -> str:
        """
        Retrieve top-k relevant guideline chunks for a given compliance check type.
        Returns concatenated text ready for prompt injection.

        check_type: one of "color", "typography", "logo", "copywriting"
        """
        query_text = CHECK_TYPE_QUERIES.get(check_type, check_type)
        return self.retrieve_by_query(brand_id, query_text, top_k)

    def retrieve_by_query(self, brand_id: str, query: str, top_k: int = 3) -> str:
        """Retrieve chunks by free-form query string."""
        try:
            model = self._get_model()
            client = self._get_qdrant()
            name = self.collection_name(brand_id)

            # Check collection exists
            existing = [c.name for c in client.get_collections().collections]
            if name not in existing:
                logger.warning(f"No guideline collection found for brand {brand_id}")
                return ""

            query_vector = model.encode(
                _QUERY_PREFIX + query, normalize_embeddings=True
            ).tolist()

            response = client.query_points(
                collection_name=name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

            texts = [r.payload.get("text", "") for r in response.points if r.payload]
            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Text RAG retrieval failed for brand {brand_id}: {e}")
            return ""

    def delete_brand_collection(self, brand_id: str):
        """Remove all indexed chunks for a brand."""
        try:
            client = self._get_qdrant()
            client.delete_collection(self.collection_name(brand_id))
        except Exception as e:
            logger.warning(f"Failed to delete guideline collection for brand {brand_id}: {e}")
