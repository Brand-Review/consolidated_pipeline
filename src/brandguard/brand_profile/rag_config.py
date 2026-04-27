"""RAG config loader — reads configs/rag.yaml once, supports per-brand overrides."""

from __future__ import annotations
import copy
import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "chunking": {
        "default_strategy": "recursive",
        "fixed": {"chunk_size": 800, "overlap": 100},
        "recursive": {
            "separators": ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            "chunk_size": 800,
            "min_chunk_size": 100,
        },
        "semantic": {
            "embedding_model": "intfloat/multilingual-e5-large",
            "breakpoint_percentile": 90,
            "min_chunk_chars": 200,
            "max_chunk_chars": 1500,
        },
    },
    "embeddings": {
        "dense_model": "intfloat/multilingual-e5-large",
        "dense_dim": 1024,
        "sparse_model": "Qdrant/bm25",
        "sparse_tokenizer": "word",
        "sparse_lowercase": True,
    },
    "dedup": {"enabled": True, "cosine_threshold": 0.95, "scope": "within_brand"},
    "s3": {"prefix": "brand-onboarding"},
    "retrieval": {
        "candidate_pool_size": 20,
        "final_top_k": 5,
        "fusion": {
            "algorithm": "rrf",
            "rrf_k": 60,
            "weights": {"dense": 0.7, "sparse": 0.3},
        },
        "per_source_top_k": {"dense": 10, "sparse": 10},
        "reranker": {
            "enabled": True,
            "type": "cross_encoder",
            "model": "BAAI/bge-reranker-base",
            "batch_size": 20,
            "timeout_seconds": 5,
        },
        "fallback": {
            "on_sparse_failure": "dense_only",
            "on_reranker_failure": "skip_rerank",
        },
    },
    "generation": {
        "temperature": 0.0,
        "max_tokens": 800,
        "max_context_chunks": 5,
        "sanitize": {"max_question_chars": 2000},
    },
    "verification": {
        "enabled": True,
        "max_concurrent_verifications": 5,
        "per_claim_timeout_seconds": 10,
        "on_timeout_status": "unverified",
    },
    "confidence": {
        "weights": {"retrieval": 0.4, "citation_coverage": 0.4, "completeness": 0.2},
        "thresholds": {"idk_composite": 0.4, "idk_retrieval": 0.3},
    },
}


def _find_config_path() -> Optional[str]:
    override = os.environ.get("RAG_CONFIG_PATH")
    if override and os.path.exists(override):
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "..", "configs", "rag.yaml"),
        os.path.join(os.getcwd(), "configs", "rag.yaml"),
        os.path.join(os.getcwd(), "consolidated_pipeline", "configs", "rag.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


_cached_config: Optional[Dict[str, Any]] = None


def load_rag_config() -> Dict[str, Any]:
    global _cached_config
    if _cached_config is not None:
        return copy.deepcopy(_cached_config)

    path = _find_config_path()
    config = copy.deepcopy(_DEFAULT_CONFIG)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            config = _deep_merge(config, loaded)
            logger.info(f"Loaded RAG config from {path}")
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}. Falling back to defaults.")
    else:
        logger.info("No rag.yaml found — using defaults")

    _cached_config = config
    return copy.deepcopy(config)


def apply_overrides(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    return _deep_merge(copy.deepcopy(base), override)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
