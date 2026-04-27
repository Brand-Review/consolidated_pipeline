"""Grounded RAG pipeline — the single orchestrator behind `POST /api/brand/{brand_id}/ask`.

Wires Phase 2 retrieval to the Phase 3 generator, citation verifier, completeness
judge, confidence scorer, and IDK responder. Short-circuits to a structured IDK
payload when retrieval comes up empty, when the generator itself refuses, or when
the composite / retrieval confidence falls below configured thresholds.

One instance is built at app startup and reused across requests — all components
are stateless or internally thread-safe.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...core.llm_client import LLMClient
from ..retrieval.types import RetrievalResult
from ..text_rag import TextRAG
from .citation_verifier import CitationVerifier
from .completeness_judge import CompletenessJudge
from .confidence_scorer import ConfidenceScorer
from .config import GroundedConfig, load_grounded_config
from .grounded_generator import GroundedGenerator
from .idk_responder import IDKResponder
from .types import (
    Citation,
    ConfidenceBreakdown,
    GroundedResponse,
    RawAnswer,
    VerifiedAnswer,
)

logger = logging.getLogger(__name__)


class GroundedRAGPipeline:
    """Orchestrates retrieval → generation → verification → scoring → IDK gate."""

    def __init__(
        self,
        text_rag: TextRAG,
        llm: LLMClient,
        config: Optional[GroundedConfig] = None,
        generator: Optional[GroundedGenerator] = None,
        verifier: Optional[CitationVerifier] = None,
        completeness_judge: Optional[CompletenessJudge] = None,
        scorer: Optional[ConfidenceScorer] = None,
        idk_responder: Optional[IDKResponder] = None,
    ):
        self._text_rag = text_rag
        self._llm = llm
        self._config = config or load_grounded_config()
        self._generator = generator or GroundedGenerator(llm, self._config.generation)
        self._verifier = verifier or CitationVerifier(llm, self._config.verification)
        self._completeness = completeness_judge or CompletenessJudge(llm)
        self._scorer = scorer or ConfidenceScorer(self._config.confidence)
        self._idk = idk_responder or IDKResponder()

    def answer(
        self,
        brand_id: str,
        question: str,
        brand_override: Optional[Dict[str, Any]] = None,
    ) -> GroundedResponse:
        question = (question or "").strip()
        if not question:
            return self._empty_question_response()

        retrieval = self._retrieve(brand_id, question, brand_override)
        retrieval_debug = self._retrieval_debug(retrieval)

        if not retrieval.chunks:
            reason = retrieval.fallback_reason or "no_retrieved_chunks"
            return self._idk_response(
                question=question,
                retrieval=retrieval,
                raw_answer=None,
                reason=reason,
                confidence=self._scorer.score(retrieval, VerifiedAnswer(text=""), 0.0),
                retrieval_debug=retrieval_debug,
            )

        raw = self._generator.generate(question, retrieval.chunks)

        # Short-circuit: the generator itself said "I don't know".
        if raw.said_idk:
            confidence = self._scorer.score(retrieval, VerifiedAnswer(text=""), 0.0)
            return self._idk_response(
                question=question,
                retrieval=retrieval,
                raw_answer=raw,
                reason="generator_said_idk",
                confidence=confidence,
                retrieval_debug=retrieval_debug,
            )

        verified = self._verifier.verify(raw.claims, retrieval.chunks)
        verified.text = raw.text

        completeness = self._completeness.judge(question, raw.text)
        confidence = self._scorer.score(retrieval, verified, completeness)

        if self._should_idk(confidence):
            reason = self._idk_reason(confidence)
            return self._idk_response(
                question=question,
                retrieval=retrieval,
                raw_answer=raw,
                reason=reason,
                confidence=confidence,
                retrieval_debug=retrieval_debug,
            )

        return GroundedResponse(
            answer=raw.text,
            citations=verified.citations,
            confidence=confidence,
            unsupported_claims=self._unsupported_claims(verified),
            found=[],
            missing=None,
            suggested_documents=[],
            is_idk=False,
            idk_reason=None,
            retrieval_debug=retrieval_debug,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _retrieve(
        self,
        brand_id: str,
        question: str,
        brand_override: Optional[Dict[str, Any]],
    ) -> RetrievalResult:
        try:
            return self._text_rag.retrieve_hybrid(
                brand_id=brand_id,
                query=question,
                brand_override=brand_override,
            )
        except Exception as exc:
            logger.warning("[GroundedRAGPipeline] retrieval failed brand=%s: %s", brand_id, exc)
            return RetrievalResult(query=question, fallback_reason=f"retrieval_error: {exc}")

    def _should_idk(self, confidence: ConfidenceBreakdown) -> bool:
        thresholds = self._config.confidence.thresholds
        return (
            confidence.composite < thresholds.idk_composite
            or confidence.retrieval < thresholds.idk_retrieval
        )

    def _idk_reason(self, confidence: ConfidenceBreakdown) -> str:
        thresholds = self._config.confidence.thresholds
        if confidence.retrieval < thresholds.idk_retrieval:
            return "low_retrieval_confidence"
        return "low_composite_confidence"

    def _idk_response(
        self,
        question: str,
        retrieval: RetrievalResult,
        raw_answer: Optional[RawAnswer],
        reason: str,
        confidence: ConfidenceBreakdown,
        retrieval_debug: Optional[Dict[str, Any]],
    ) -> GroundedResponse:
        payload = self._idk.synthesize(
            question=question,
            retrieval=retrieval,
            raw_answer=raw_answer,
            reason=reason,
        )
        return GroundedResponse(
            answer=None,
            citations=[],
            confidence=confidence,
            unsupported_claims=[],
            found=payload.get("found", []),
            missing=payload.get("missing"),
            suggested_documents=payload.get("suggested_documents", []),
            is_idk=True,
            idk_reason=payload.get("idk_reason", reason),
            retrieval_debug=retrieval_debug,
        )

    def _empty_question_response(self) -> GroundedResponse:
        return GroundedResponse(
            answer=None,
            citations=[],
            confidence=ConfidenceBreakdown(composite=0.0, retrieval=0.0, citation_coverage=0.0, completeness=0.0),
            unsupported_claims=[],
            found=[],
            missing="No question was provided.",
            suggested_documents=[],
            is_idk=True,
            idk_reason="empty_question",
            retrieval_debug=None,
        )

    @staticmethod
    def _unsupported_claims(verified: VerifiedAnswer) -> List:
        unsupported_idxs = {
            c.idx for c in verified.citations if c.status == "unsupported"
        }
        return [
            claim
            for claim in verified.claims
            if any(idx in unsupported_idxs for idx in claim.cited_indices)
        ]

    @staticmethod
    def _retrieval_debug(retrieval: RetrievalResult) -> Dict[str, Any]:
        return {
            "latency_ms": round(retrieval.latency_ms, 2),
            "used_strategies": list(retrieval.used_strategies),
            "fallback_reason": retrieval.fallback_reason,
            "chunk_count": len(retrieval.chunks),
        }
