"""
Asset RAG (Multi-Modal)
Embeds approved/rejected brand images with CLIP and stores/retrieves them
from Qdrant (one collection per brand).

Collection naming: brand_{brand_id}_assets
Vector dim: 512 (CLIP ViT-B/32)
"""

from __future__ import annotations
import base64
import io
import logging
import os
import uuid
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CLIP_DIM = 512  # ViT-B/32


class AssetRAG:
    """
    Manages per-brand image vector collections in Qdrant using CLIP embeddings.
    Stores approved and rejected example images as searchable reference assets.
    """

    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY", None)
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self._clip_model = None
        self._clip_preprocess = None
        self._qdrant: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_clip(self):
        if self._clip_model is None:
            import open_clip
            import torch
            logger.info(f"Loading CLIP model: {self.clip_model_name} ({self.clip_pretrained})")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.clip_pretrained
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
        return self._clip_model, self._clip_preprocess

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
        return f"brand_{brand_id}_assets"

    def _ensure_collection(self, brand_id: str):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_qdrant()
        name = self.collection_name(brand_id)
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=_CLIP_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {name}")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_image(self, image_bytes: bytes) -> List[float]:
        """Embed raw image bytes using CLIP, return normalized vector."""
        import torch
        from PIL import Image

        model, preprocess = self._get_clip()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()

    def _embed_numpy(self, image_array: np.ndarray) -> List[float]:
        """Embed a numpy BGR/RGB image array using CLIP."""
        import cv2
        from PIL import Image

        # Convert BGR → RGB if needed, then to PIL
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_array
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return self._embed_image(buf.getvalue())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_assets(
        self,
        brand_id: str,
        approved_images: List[bytes],
        rejected_images: List[Dict[str, Any]],
    ) -> int:
        """
        Index approved and rejected images for a brand.

        approved_images: list of raw image bytes
        rejected_images: list of {"image_bytes": bytes, "rejection_reasons": [str]}

        Returns total number of points indexed.
        """
        self._ensure_collection(brand_id)
        client = self._get_qdrant()
        from qdrant_client.models import PointStruct

        points = []

        for img_bytes in approved_images:
            try:
                vec = self._embed_image(img_bytes)
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "label": "approved",
                        "rejection_reasons": [],
                        "image_b64": base64.b64encode(img_bytes).decode(),
                        "brand_id": brand_id,
                    },
                ))
            except Exception as e:
                logger.warning(f"Failed to embed approved image: {e}")

        for entry in rejected_images:
            try:
                img_bytes = entry["image_bytes"]
                reasons = entry.get("rejection_reasons", [])
                vec = self._embed_image(img_bytes)
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "label": "rejected",
                        "rejection_reasons": reasons,
                        "image_b64": base64.b64encode(img_bytes).decode(),
                        "brand_id": brand_id,
                    },
                ))
            except Exception as e:
                logger.warning(f"Failed to embed rejected image: {e}")

        if points:
            client.upsert(collection_name=self.collection_name(brand_id), points=points)
            logger.info(f"Indexed {len(points)} assets for brand {brand_id}")

        return len(points)

    def retrieve_similar(
        self,
        brand_id: str,
        query_image: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar approved/rejected reference assets for a query image.

        Returns list of:
          {"label": "approved"|"rejected", "rejection_reasons": [...],
           "image_b64": str, "score": float}
        """
        try:
            client = self._get_qdrant()
            name = self.collection_name(brand_id)
            existing = [c.name for c in client.get_collections().collections]
            if name not in existing:
                logger.warning(f"No asset collection found for brand {brand_id}")
                return []

            query_vec = self._embed_numpy(query_image)
            response = client.query_points(
                collection_name=name,
                query=query_vec,
                limit=top_k,
                with_payload=True,
            )

            out = []
            for r in response.points:
                p = r.payload or {}
                out.append({
                    "label": p.get("label", "unknown"),
                    "rejection_reasons": p.get("rejection_reasons", []),
                    "image_b64": p.get("image_b64", ""),
                    "score": float(r.score),
                })
            return out

        except Exception as e:
            logger.error(f"Asset RAG retrieval failed for brand {brand_id}: {e}")
            return []

    def delete_brand_collection(self, brand_id: str):
        """Remove all indexed assets for a brand."""
        try:
            client = self._get_qdrant()
            client.delete_collection(self.collection_name(brand_id))
        except Exception as e:
            logger.warning(f"Failed to delete asset collection for brand {brand_id}: {e}")
