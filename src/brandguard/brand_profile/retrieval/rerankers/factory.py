"""Reranker factory. Dispatches on RerankerConfig.type."""

from __future__ import annotations
import logging
from typing import Optional

from ..config import RerankerConfig
from .base import Reranker

logger = logging.getLogger(__name__)


def get_reranker(config: RerankerConfig) -> Optional[Reranker]:
    if not config.enabled or config.type == "none":
        return None
    if config.type == "cross_encoder":
        from .cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    if config.type == "llm_judge":
        from .llm_judge_reranker import LLMJudgeReranker
        return LLMJudgeReranker(timeout_seconds=config.timeout_seconds)
    logger.warning(f"Unknown reranker type {config.type!r}; disabling rerank.")
    return None
