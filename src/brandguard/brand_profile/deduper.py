"""
Deduper — skips near-duplicate chunks within a brand's existing Qdrant collection
plus within the current ingest batch.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DropReason:
    chunk_index: int
    matched_existing_id: Optional[str]
    similarity: float
    source: str  # "within_batch" | "existing_collection"


class Deduper:
    def __init__(self, qdrant_client, threshold: float = 0.95):
        self.client = qdrant_client
        self.threshold = threshold

    def filter_new_chunks(
        self,
        collection_name: str,
        candidate_chunks: List[Any],
        candidate_dense_vecs: List[List[float]],
    ) -> Tuple[List[Any], List[List[float]], List[DropReason]]:
        if not candidate_chunks:
            return [], [], []

        kept_chunks: List[Any] = []
        kept_vecs: List[List[float]] = []
        dropped: List[DropReason] = []

        existing_present = self._collection_exists(collection_name)

        for idx, (chunk, vec) in enumerate(zip(candidate_chunks, candidate_dense_vecs)):
            # Within-batch dedup: compare against already-kept vectors
            batch_hit = self._best_cosine(vec, kept_vecs)
            if batch_hit is not None and batch_hit[1] >= self.threshold:
                dropped.append(DropReason(
                    chunk_index=idx,
                    matched_existing_id=None,
                    similarity=batch_hit[1],
                    source="within_batch",
                ))
                continue

            # Cross-collection dedup
            if existing_present:
                match = self._query_top1(collection_name, vec)
                if match is not None and match[1] >= self.threshold:
                    dropped.append(DropReason(
                        chunk_index=idx,
                        matched_existing_id=str(match[0]),
                        similarity=match[1],
                        source="existing_collection",
                    ))
                    continue

            kept_chunks.append(chunk)
            kept_vecs.append(vec)

        return kept_chunks, kept_vecs, dropped

    # ------------------------------------------------------------------

    def _collection_exists(self, name: str) -> bool:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            return name in existing
        except Exception as e:
            logger.warning(f"Deduper: get_collections failed: {e}")
            return False

    def _query_top1(self, collection: str, vec: List[float]):
        try:
            from qdrant_client.models import NamedVector
            try:
                resp = self.client.query_points(
                    collection_name=collection,
                    query=vec,
                    using="dense",
                    limit=1,
                    with_payload=False,
                )
            except Exception:
                resp = self.client.query_points(
                    collection_name=collection,
                    query=vec,
                    limit=1,
                    with_payload=False,
                )
            points = resp.points if hasattr(resp, "points") else resp
            if not points:
                return None
            p = points[0]
            return (p.id, float(p.score))
        except Exception as e:
            logger.warning(f"Deduper: query failed against {collection}: {e}")
            return None

    @staticmethod
    def _best_cosine(vec: List[float], pool: List[List[float]]):
        if not pool:
            return None
        best = (-1, -1.0)
        for i, other in enumerate(pool):
            s = _cosine(vec, other)
            if s > best[1]:
                best = (i, s)
        return best


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
