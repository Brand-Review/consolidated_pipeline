"""Grounded generation + citation verification layer (Phase 3).

Wraps the Phase 2 retriever with: grounded generation, citation parsing and
verification, confidence scoring, and a structured "I don't know" path.
"""

from .types import (
    Claim,
    Citation,
    RawAnswer,
    VerifiedAnswer,
    ConfidenceBreakdown,
    GroundedResponse,
)
from .config import (
    GroundedConfig,
    GenerationConfig,
    VerificationConfig,
    ConfidenceConfig,
    load_grounded_config,
)
from .citation_parser import CitationParser
from .confidence_scorer import ConfidenceScorer
from .grounded_generator import GroundedGenerator
from .citation_verifier import CitationVerifier
from .completeness_judge import CompletenessJudge
from .idk_responder import IDKResponder
from .grounded_pipeline import GroundedRAGPipeline

__all__ = [
    "Claim",
    "Citation",
    "RawAnswer",
    "VerifiedAnswer",
    "ConfidenceBreakdown",
    "GroundedResponse",
    "GroundedConfig",
    "GenerationConfig",
    "VerificationConfig",
    "ConfidenceConfig",
    "load_grounded_config",
    "CitationParser",
    "ConfidenceScorer",
    "GroundedGenerator",
    "CitationVerifier",
    "CompletenessJudge",
    "IDKResponder",
    "GroundedRAGPipeline",
]
