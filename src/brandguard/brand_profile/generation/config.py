"""Typed generation / verification / confidence config."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SanitizeConfig:
    max_question_chars: int = 2000


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 800
    max_context_chunks: int = 5
    llm_timeout_seconds: int = 60
    sanitize: SanitizeConfig = field(default_factory=SanitizeConfig)


@dataclass
class VerificationConfig:
    enabled: bool = True
    max_concurrent_verifications: int = 5
    per_claim_timeout_seconds: float = 10.0
    on_timeout_status: str = "unverified"  # unverified | unsupported


@dataclass
class ConfidenceWeights:
    retrieval: float = 0.4
    citation_coverage: float = 0.4
    completeness: float = 0.2


@dataclass
class ConfidenceThresholds:
    idk_composite: float = 0.4
    idk_retrieval: float = 0.3


@dataclass
class ConfidenceConfig:
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)


@dataclass
class GroundedConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "GroundedConfig":
        raw = raw or {}
        gen_raw = raw.get("generation") or {}
        san_raw = gen_raw.get("sanitize") or {}
        ver_raw = raw.get("verification") or {}
        conf_raw = raw.get("confidence") or {}
        weights_raw = conf_raw.get("weights") or {}
        thresh_raw = conf_raw.get("thresholds") or {}
        return cls(
            generation=GenerationConfig(
                temperature=float(gen_raw.get("temperature", 0.0)),
                max_tokens=int(gen_raw.get("max_tokens", 800)),
                max_context_chunks=int(gen_raw.get("max_context_chunks", 5)),
                llm_timeout_seconds=int(gen_raw.get("llm_timeout_seconds", 60)),
                sanitize=SanitizeConfig(
                    max_question_chars=int(san_raw.get("max_question_chars", 2000)),
                ),
            ),
            verification=VerificationConfig(
                enabled=bool(ver_raw.get("enabled", True)),
                max_concurrent_verifications=int(ver_raw.get("max_concurrent_verifications", 5)),
                per_claim_timeout_seconds=float(ver_raw.get("per_claim_timeout_seconds", 10)),
                on_timeout_status=str(ver_raw.get("on_timeout_status", "unverified")).lower(),
            ),
            confidence=ConfidenceConfig(
                weights=ConfidenceWeights(
                    retrieval=float(weights_raw.get("retrieval", 0.4)),
                    citation_coverage=float(weights_raw.get("citation_coverage", 0.4)),
                    completeness=float(weights_raw.get("completeness", 0.2)),
                ),
                thresholds=ConfidenceThresholds(
                    idk_composite=float(thresh_raw.get("idk_composite", 0.4)),
                    idk_retrieval=float(thresh_raw.get("idk_retrieval", 0.3)),
                ),
            ),
        )


def load_grounded_config(brand_override: Optional[Dict[str, Any]] = None) -> GroundedConfig:
    from ..rag_config import load_rag_config, apply_overrides
    base = load_rag_config()
    merged = apply_overrides(base, brand_override) if brand_override else base
    return GroundedConfig.from_dict(merged)
