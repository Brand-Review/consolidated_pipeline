"""Completeness judge.

Thin LLM wrapper around `completeness_judge_v1` prompt. Scores how completely
an answer addresses the question on a 0.0–1.0 scale. Does NOT judge factual
correctness — that is the citation verifier's job.

Always returns a clipped float in [0, 1]; on any error (timeout, parse fail,
missing field) returns 0.0 with a logged warning, so the overall pipeline
can still compute a composite confidence.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...core.llm_client import LLMClient, LLMResponseError
from ...core.prompt_registry import PromptRegistry, registry as default_registry

logger = logging.getLogger(__name__)


class CompletenessJudge:
    def __init__(
        self,
        llm: LLMClient,
        prompt_registry: Optional[PromptRegistry] = None,
        timeout_seconds: int = 15,
    ):
        self._llm = llm
        self._registry = prompt_registry or default_registry
        self._timeout = max(1, int(timeout_seconds))

    def judge(self, question: str, answer: str) -> float:
        if not (question or "").strip() or not (answer or "").strip():
            return 0.0

        tmpl = self._registry.get("completeness_judge")
        user_msg = tmpl.user_template.format(question=question, answer=answer)
        messages = [
            {"role": "system", "content": tmpl.system},
            {"role": "user", "content": user_msg},
        ]

        try:
            parsed, _usage = self._llm.chat(
                messages,
                response_format={"type": "json_object"},
                timeout=self._timeout,
            )
        except LLMResponseError as exc:
            logger.warning("[CompletenessJudge] LLM call failed: %s", exc)
            return 0.0

        if not isinstance(parsed, dict):
            return 0.0

        raw_score = parsed.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning("[CompletenessJudge] non-numeric score: %r", raw_score)
            return 0.0

        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
