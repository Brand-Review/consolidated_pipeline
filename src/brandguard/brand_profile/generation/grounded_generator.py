"""Grounded answer generator.

Formats the top-K retrieved chunks as numbered context blocks, asks the LLM
for an answer that cites claims with `[N]`, and returns a `RawAnswer` with
parsed claims ready for citation verification.

The LLM is forced to return `{"answer": "..."}` so it routes through the
shared `LLMClient.chat()` (which always parses JSON). Plain-text answers
don't fit that contract — the JSON envelope is the minimum overhead to
reuse the shared client.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ...core.llm_client import LLMClient, LLMResponseError
from ...core.prompt_registry import PromptRegistry, registry as default_registry
from ..retrieval.types import Reranked
from .citation_parser import CitationParser
from .config import GenerationConfig
from .types import RawAnswer

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """Turns `(question, chunks)` into a `RawAnswer` with citation-tagged claims."""

    def __init__(
        self,
        llm: LLMClient,
        config: GenerationConfig,
        parser: Optional[CitationParser] = None,
        prompt_registry: Optional[PromptRegistry] = None,
    ):
        self._llm = llm
        self._config = config
        self._parser = parser or CitationParser()
        self._registry = prompt_registry or default_registry

    def generate(self, question: str, chunks: List[Reranked]) -> RawAnswer:
        """Call the LLM, parse claims, and return a RawAnswer."""
        clean_question = self._sanitize_question(question)
        top_chunks = chunks[: max(1, self._config.max_context_chunks)]

        if not top_chunks:
            # Nothing to ground on — emit the canonical IDK immediately.
            text = (
                "I don't know based on the provided context. "
                "No relevant content was retrieved for this question."
            )
            return RawAnswer(text=text, claims=[], said_idk=True, usage={})

        numbered_context = self._format_context(top_chunks)
        tmpl = self._registry.get("grounded_answer")
        user_msg = tmpl.user_template.format(
            question=clean_question,
            numbered_context=numbered_context,
        )
        messages = [
            {"role": "system", "content": tmpl.system},
            {"role": "user", "content": user_msg},
        ]

        try:
            parsed, usage = self._llm.chat(
                messages,
                response_format={"type": "json_object"},
                timeout=self._config.llm_timeout_seconds,
            )
        except LLMResponseError as exc:
            logger.warning("[GroundedGenerator] LLM call failed: %s", exc)
            text = (
                "I don't know based on the provided context. "
                "The answer service was unable to produce a grounded response."
            )
            return RawAnswer(text=text, claims=[], said_idk=True, usage={})

        answer_text = self._extract_answer(parsed)
        said_idk = self._parser.said_idk(answer_text)
        claims = [] if said_idk else self._parser.parse(answer_text)

        return RawAnswer(
            text=answer_text,
            claims=claims,
            said_idk=said_idk,
            usage=usage or {},
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _sanitize_question(self, question: str) -> str:
        q = (question or "").strip()
        limit = self._config.sanitize.max_question_chars
        if limit and len(q) > limit:
            q = q[:limit].rstrip() + "…"
        return q

    def _format_context(self, chunks: List[Reranked]) -> str:
        blocks: List[str] = []
        for i, reranked in enumerate(chunks, start=1):
            text = (reranked.candidate.text or "").strip()
            blocks.append(f"[{i}] {text}")
        return "\n\n".join(blocks)

    def _extract_answer(self, parsed: dict) -> str:
        if isinstance(parsed, dict):
            # Primary contract: {"answer": "..."}
            raw = parsed.get("answer")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            # Fallback: some models stuff everything into a "text" or "response" field.
            for key in ("text", "response", "content"):
                v = parsed.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        logger.warning(
            "[GroundedGenerator] LLM returned unexpected payload shape: keys=%s",
            list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
        return (
            "I don't know based on the provided context. "
            "The answer service returned an unrecognized payload."
        )
