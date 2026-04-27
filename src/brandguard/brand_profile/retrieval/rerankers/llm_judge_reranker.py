"""LLM-as-judge reranker. Single batched call returns a score per candidate."""

from __future__ import annotations
import logging
from typing import List, Optional

from .base import Reranker

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a relevance judge for a retrieval-augmented system. "
    "Score each candidate chunk on how well it answers the given query, "
    "from 0.0 (irrelevant) to 1.0 (highly relevant). "
    "Return ONLY valid JSON of the form {\"scores\": [float, float, ...]} "
    "with one score per candidate, in the same order as they were given."
)


class LLMJudgeReranker(Reranker):
    def __init__(self, llm_client=None, timeout_seconds: float = 5.0):
        self._llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    def _client(self):
        if self._llm_client is None:
            from ....core.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def score_pairs(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        user = (
            f"Query: {query}\n\n"
            f"Candidates (numbered):\n{numbered}\n\n"
            f"Return {{\"scores\": [...]}} with {len(candidates)} floats in [0.0, 1.0]."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        result, _usage = self._client().chat(
            messages=messages,
            response_format={"type": "json_object"},
            timeout=int(self.timeout_seconds) if self.timeout_seconds else 120,
        )
        raw_scores = result.get("scores") if isinstance(result, dict) else None
        if not isinstance(raw_scores, list) or len(raw_scores) != len(candidates):
            raise ValueError(
                f"LLM judge returned malformed scores: "
                f"expected list of {len(candidates)}, got {raw_scores!r}"
            )
        return [float(s) for s in raw_scores]
