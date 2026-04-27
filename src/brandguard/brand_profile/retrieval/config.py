"""Typed retrieval config. Loaded from rag.yaml via rag_config.load_rag_config()."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FusionConfig:
    algorithm: str = "rrf"           # rrf | dbsf
    rrf_k: int = 60
    weights: Dict[str, float] = field(default_factory=lambda: {"dense": 0.7, "sparse": 0.3})


@dataclass
class PerSourceTopK:
    dense: int = 10
    sparse: int = 10


@dataclass
class RerankerConfig:
    enabled: bool = True
    type: str = "cross_encoder"      # cross_encoder | llm_judge | none
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 20
    timeout_seconds: float = 5.0


@dataclass
class FallbackConfig:
    on_sparse_failure: str = "dense_only"    # dense_only | fail
    on_reranker_failure: str = "skip_rerank" # skip_rerank | fail


@dataclass
class RetrievalConfig:
    candidate_pool_size: int = 20
    final_top_k: int = 5
    fusion: FusionConfig = field(default_factory=FusionConfig)
    per_source_top_k: PerSourceTopK = field(default_factory=PerSourceTopK)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RetrievalConfig":
        raw = raw or {}
        fusion_raw = raw.get("fusion") or {}
        per_source_raw = raw.get("per_source_top_k") or {}
        rerank_raw = raw.get("reranker") or {}
        fallback_raw = raw.get("fallback") or {}
        return cls(
            candidate_pool_size=int(raw.get("candidate_pool_size", 20)),
            final_top_k=int(raw.get("final_top_k", 5)),
            fusion=FusionConfig(
                algorithm=str(fusion_raw.get("algorithm", "rrf")).lower(),
                rrf_k=int(fusion_raw.get("rrf_k", 60)),
                weights={
                    "dense": float((fusion_raw.get("weights") or {}).get("dense", 0.7)),
                    "sparse": float((fusion_raw.get("weights") or {}).get("sparse", 0.3)),
                },
            ),
            per_source_top_k=PerSourceTopK(
                dense=int(per_source_raw.get("dense", 10)),
                sparse=int(per_source_raw.get("sparse", 10)),
            ),
            reranker=RerankerConfig(
                enabled=bool(rerank_raw.get("enabled", True)),
                type=str(rerank_raw.get("type", "cross_encoder")).lower(),
                model=str(rerank_raw.get("model", "BAAI/bge-reranker-base")),
                batch_size=int(rerank_raw.get("batch_size", 20)),
                timeout_seconds=float(rerank_raw.get("timeout_seconds", 5)),
            ),
            fallback=FallbackConfig(
                on_sparse_failure=str(fallback_raw.get("on_sparse_failure", "dense_only")).lower(),
                on_reranker_failure=str(fallback_raw.get("on_reranker_failure", "skip_rerank")).lower(),
            ),
        )


def load_retrieval_config(brand_override: Optional[Dict[str, Any]] = None) -> RetrievalConfig:
    """Convenience: read global rag.yaml, apply per-brand override, return typed config."""
    from ..rag_config import load_rag_config, apply_overrides
    base = load_rag_config()
    merged = apply_overrides(base, brand_override) if brand_override else base
    return RetrievalConfig.from_dict(merged.get("retrieval"))
