"""Hybrid retrieval package (Phase 2).

Dense + sparse fusion via Qdrant server-side RRF/DBSF, optional cross-encoder
or LLM-as-judge rerank, graceful fallback. `text_rag.retrieve_hybrid` delegates
here; `text_rag.retrieve` keeps its legacy string contract by wrapping the
structured result.
"""

from .types import FusedCandidate, Reranked, RetrievalResult
from .config import RetrievalConfig, load_retrieval_config
from .hybrid_retriever import HybridRetriever

__all__ = [
    "FusedCandidate",
    "Reranked",
    "RetrievalResult",
    "RetrievalConfig",
    "load_retrieval_config",
    "HybridRetriever",
]
