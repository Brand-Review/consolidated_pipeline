"""Composite confidence scorer for grounded answers.

Pure math — takes the retrieval result, the verified answer, and a
completeness score, and returns a `ConfidenceBreakdown`:

  retrieval          = mean of the top-K rerank scores, clipped to [0, 1]
  citation_coverage  = supported citations / total citations (0 if none)
  completeness       = judge score in [0, 1] (passed in from CompletenessJudge)
  composite          = Σ w_i · component_i  using config weights

The composite score and its components drive the IDK gate in the pipeline.
"""

from __future__ import annotations

from typing import Optional

from ..retrieval.types import RetrievalResult
from .config import ConfidenceConfig
from .types import ConfidenceBreakdown, VerifiedAnswer


class ConfidenceScorer:
    """Stateless — one instance can score every request."""

    def __init__(self, config: ConfidenceConfig, top_k: int = 3):
        self._config = config
        self._top_k = max(1, top_k)

    def score(
        self,
        retrieval: RetrievalResult,
        verified: VerifiedAnswer,
        completeness: Optional[float],
    ) -> ConfidenceBreakdown:
        retrieval_conf = self._retrieval_confidence(retrieval)
        coverage = self._citation_coverage(verified)
        comp = self._clip01(completeness if completeness is not None else 0.0)

        w = self._config.weights
        composite = (
            w.retrieval * retrieval_conf
            + w.citation_coverage * coverage
            + w.completeness * comp
        )

        return ConfidenceBreakdown(
            composite=round(self._clip01(composite), 4),
            retrieval=round(retrieval_conf, 4),
            citation_coverage=round(coverage, 4),
            completeness=round(comp, 4),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _retrieval_confidence(self, retrieval: RetrievalResult) -> float:
        if not retrieval or not retrieval.chunks:
            return 0.0
        top = retrieval.chunks[: self._top_k]
        scores = [self._clip01(r.rerank_score) for r in top]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _citation_coverage(self, verified: VerifiedAnswer) -> float:
        total = verified.total_citations or 0
        if total <= 0:
            return 0.0
        supported = verified.supported_count or 0
        return self._clip01(supported / total)

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
