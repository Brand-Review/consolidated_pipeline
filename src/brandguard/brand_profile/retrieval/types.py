"""Typed retrieval structures shared across the hybrid retrieval layer."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FusedCandidate:
    """A chunk that survived Qdrant server-side fusion (RRF/DBSF)."""
    point_id: str
    text: str
    payload: Dict[str, Any]
    fused_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


@dataclass
class Reranked:
    candidate: FusedCandidate
    rerank_score: float


@dataclass
class RetrievalResult:
    """Structured retrieval payload. Phase 3's generator consumes this directly."""
    query: str
    chunks: List[Reranked] = field(default_factory=list)
    dense_top_score: Optional[float] = None
    sparse_top_score: Optional[float] = None
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    used_strategies: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None

    def as_concatenated_text(self, separator: str = "\n\n") -> str:
        """Legacy-style string join used by the old `text_rag.retrieve` contract."""
        return separator.join(r.candidate.text for r in self.chunks if r.candidate.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [
                {
                    "point_id": r.candidate.point_id,
                    "text": r.candidate.text,
                    "payload": r.candidate.payload,
                    "fused_score": r.candidate.fused_score,
                    "dense_rank": r.candidate.dense_rank,
                    "sparse_rank": r.candidate.sparse_rank,
                    "rerank_score": r.rerank_score,
                }
                for r in self.chunks
            ],
            "dense_top_score": self.dense_top_score,
            "sparse_top_score": self.sparse_top_score,
            "fusion_weights": self.fusion_weights,
            "latency_ms": self.latency_ms,
            "used_strategies": self.used_strategies,
            "fallback_reason": self.fallback_reason,
        }
