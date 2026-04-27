"""Structured IDK (I-don't-know) responder.

When the confidence gate fires, we don't just echo "I don't know based on the
provided context." — we return a payload that tells the caller:

  - `found`: the top retrieved chunks (source, section, page, preview) so the
    user can see that the system *did* look and what it found
  - `missing`: a short human-readable description of what was not covered
  - `suggested_documents`: unique source filenames from the retrieved chunks
    that may be worth checking directly

No LLM call — this is deterministic summarization of the retrieval result and
(optionally) the tail of the generator's own answer text after the IDK
sentinel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..retrieval.types import RetrievalResult
from .types import RawAnswer


_PREVIEW_CHARS = 240
_FOUND_TOP_K = 5


class IDKResponder:
    """Stateless — reuse a single instance."""

    def synthesize(
        self,
        question: str,
        retrieval: Optional[RetrievalResult],
        raw_answer: Optional[RawAnswer] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        found = self._summarize_found(retrieval)
        missing = self._extract_missing(raw_answer, reason, retrieval)
        suggested = self._suggest_documents(retrieval)
        return {
            "idk_reason": reason or "confidence_below_threshold",
            "found": found,
            "missing": missing,
            "suggested_documents": suggested,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _summarize_found(self, retrieval: Optional[RetrievalResult]) -> List[Dict[str, Any]]:
        if not retrieval or not retrieval.chunks:
            return []
        out: List[Dict[str, Any]] = []
        for i, reranked in enumerate(retrieval.chunks[:_FOUND_TOP_K], start=1):
            payload = reranked.candidate.payload or {}
            preview = (reranked.candidate.text or "").strip().replace("\n", " ")
            if len(preview) > _PREVIEW_CHARS:
                preview = preview[:_PREVIEW_CHARS] + "…"
            out.append({
                "idx": i,
                "source_filename": str(payload.get("source_filename", "")),
                "section": str(payload.get("section", "")),
                "page": int(payload.get("page", 0) or 0),
                "preview": preview,
                "rerank_score": round(float(reranked.rerank_score or 0.0), 4),
            })
        return out

    def _extract_missing(
        self,
        raw_answer: Optional[RawAnswer],
        reason: Optional[str],
        retrieval: Optional[RetrievalResult],
    ) -> str:
        # Prefer whatever the LLM said after the IDK sentinel — it often names
        # exactly what was and wasn't covered.
        if raw_answer and raw_answer.said_idk and raw_answer.text:
            lower = raw_answer.text.lower()
            sentinel = "i don't know based on the provided context"
            idx = lower.find(sentinel)
            if idx != -1:
                tail = raw_answer.text[idx + len(sentinel):].lstrip(" .\n")
                if tail.strip():
                    return tail.strip()

        # Fall back to a reason-based template.
        if reason == "low_retrieval_confidence":
            return "The retrieval layer did not find chunks with strong relevance to the question."
        if reason == "low_composite_confidence":
            return "The retrieved context did not adequately support a cited answer to the question."
        if retrieval is None or not retrieval.chunks:
            return "No indexed content matched the question."
        return "The indexed documents do not contain enough information to answer this question confidently."

    def _suggest_documents(self, retrieval: Optional[RetrievalResult]) -> List[Dict[str, Any]]:
        if not retrieval or not retrieval.chunks:
            return []
        seen: Dict[str, Dict[str, Any]] = {}
        for reranked in retrieval.chunks:
            payload = reranked.candidate.payload or {}
            name = str(payload.get("source_filename", "")).strip()
            if not name or name in seen:
                continue
            seen[name] = {
                "source_filename": name,
                "best_section": str(payload.get("section", "")),
                "best_rerank_score": round(float(reranked.rerank_score or 0.0), 4),
            }
        return list(seen.values())
