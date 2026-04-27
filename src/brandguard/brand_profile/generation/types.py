"""Typed structures for the grounded generation / citation layer (Phase 3)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Claim:
    """A sentence from the LLM answer and the chunk indices it cites."""
    text: str
    cited_indices: List[int] = field(default_factory=list)
    char_span: Tuple[int, int] = (0, 0)


@dataclass
class Citation:
    """Post-verification record for a single [N] citation."""
    idx: int                                  # 1-indexed chunk position from retrieval
    status: str                               # supported | partial | unsupported | unverified
    source_filename: str = ""
    section: str = ""
    page: int = 0
    chunk_text_preview: str = ""
    reasoning: str = ""


@dataclass
class RawAnswer:
    """Pre-verification output of the generator."""
    text: str
    claims: List[Claim] = field(default_factory=list)
    said_idk: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiedAnswer:
    text: str
    claims: List[Claim] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    unsupported_count: int = 0
    supported_count: int = 0
    total_citations: int = 0


@dataclass
class ConfidenceBreakdown:
    composite: float
    retrieval: float
    citation_coverage: float
    completeness: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "composite": self.composite,
            "retrieval": self.retrieval,
            "citation_coverage": self.citation_coverage,
            "completeness": self.completeness,
        }


@dataclass
class GroundedResponse:
    """Final payload returned from POST /api/brand/{brand_id}/ask."""
    answer: Optional[str]
    citations: List[Citation] = field(default_factory=list)
    confidence: Optional[ConfidenceBreakdown] = None
    unsupported_claims: List[Claim] = field(default_factory=list)
    found: List[Dict[str, Any]] = field(default_factory=list)
    missing: Optional[str] = None
    suggested_documents: List[Dict[str, Any]] = field(default_factory=list)
    is_idk: bool = False
    idk_reason: Optional[str] = None
    retrieval_debug: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "unsupported_claims": [c.__dict__ for c in self.unsupported_claims],
            "found": self.found,
            "missing": self.missing,
            "suggested_documents": self.suggested_documents,
            "is_idk": self.is_idk,
            "idk_reason": self.idk_reason,
            "retrieval_debug": self.retrieval_debug,
        }
