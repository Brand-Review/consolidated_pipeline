"""Post-hoc citation verifier.

For each `[N]` citation in every claim, runs a small LLM judge call that
decides whether chunk N supports the claim sentence. Verifications run in
parallel via a bounded thread pool; each call has a per-claim timeout.

Returns a `VerifiedAnswer` with one `Citation` record per `[N]` reference
(including out-of-range indices and timeouts), plus tallies that the
confidence scorer consumes.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import List, Optional

from ...core.llm_client import LLMClient, LLMResponseError
from ...core.prompt_registry import PromptRegistry, registry as default_registry
from ..retrieval.types import Reranked
from .config import VerificationConfig
from .types import Citation, Claim, VerifiedAnswer

logger = logging.getLogger(__name__)


_VALID_STATUSES = {"supported", "partial", "unsupported", "unverified"}
_CHUNK_PREVIEW_CHARS = 240


@dataclass
class _VerificationTask:
    claim: Claim
    idx: int                  # 1-indexed citation number
    chunk: Optional[Reranked] # None if idx is out of range


class CitationVerifier:
    """Verifies every `[N]` citation produced by the generator."""

    def __init__(
        self,
        llm: LLMClient,
        config: VerificationConfig,
        prompt_registry: Optional[PromptRegistry] = None,
    ):
        self._llm = llm
        self._config = config
        self._registry = prompt_registry or default_registry

    def verify(self, claims: List[Claim], chunks: List[Reranked]) -> VerifiedAnswer:
        if not self._config.enabled:
            citations = self._build_unverified_citations(claims, chunks)
            return self._assemble(claims, citations)

        tasks = self._enumerate_tasks(claims, chunks)
        if not tasks:
            return self._assemble(claims, [])

        prompt = self._registry.get("citation_judge")

        citations: List[Citation] = []
        max_workers = max(1, self._config.max_concurrent_verifications)
        timeout = self._config.per_claim_timeout_seconds

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._judge_one, task, prompt): task for task in tasks
            }
            for fut in concurrent.futures.as_completed(futures, timeout=None):
                task = futures[fut]
                try:
                    citations.append(fut.result(timeout=timeout))
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "[CitationVerifier] timeout idx=%d claim=%r",
                        task.idx, task.claim.text[:80],
                    )
                    citations.append(self._timeout_citation(task))
                except Exception as exc:  # defensive: never fail the whole request
                    logger.warning(
                        "[CitationVerifier] unexpected error idx=%d: %s", task.idx, exc,
                    )
                    citations.append(self._timeout_citation(task, reasoning=str(exc)))

        citations.sort(key=lambda c: c.idx)

        return self._assemble(claims, citations)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _enumerate_tasks(
        self, claims: List[Claim], chunks: List[Reranked]
    ) -> List[_VerificationTask]:
        tasks: List[_VerificationTask] = []
        n = len(chunks)
        for claim in claims:
            for idx in claim.cited_indices:
                if 1 <= idx <= n:
                    tasks.append(_VerificationTask(claim=claim, idx=idx, chunk=chunks[idx - 1]))
                else:
                    tasks.append(_VerificationTask(claim=claim, idx=idx, chunk=None))
        return tasks

    def _judge_one(self, task: _VerificationTask, prompt) -> Citation:
        if task.chunk is None:
            return Citation(
                idx=task.idx,
                status="unsupported",
                reasoning=f"Citation [{task.idx}] is out of range for the retrieved context.",
            )

        user_msg = prompt.user_template.format(
            claim=task.claim.text,
            chunk=task.chunk.candidate.text,
        )
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user_msg},
        ]

        try:
            parsed, _usage = self._llm.chat(
                messages,
                response_format={"type": "json_object"},
                timeout=int(max(1, self._config.per_claim_timeout_seconds)),
            )
        except LLMResponseError as exc:
            logger.warning("[CitationVerifier] LLM call failed idx=%d: %s", task.idx, exc)
            return self._timeout_citation(task, reasoning=f"LLM error: {exc}")

        status = str(parsed.get("status", "")).strip().lower() if isinstance(parsed, dict) else ""
        if status not in _VALID_STATUSES:
            status = "unverified"
        reasoning = ""
        if isinstance(parsed, dict):
            reasoning = str(parsed.get("reasoning", "")).strip()

        return self._make_citation(task, status=status, reasoning=reasoning)

    def _timeout_citation(
        self, task: _VerificationTask, reasoning: Optional[str] = None
    ) -> Citation:
        status = self._config.on_timeout_status
        if status not in _VALID_STATUSES:
            status = "unverified"
        msg = reasoning or (
            "Verification timed out; citation could not be confirmed."
            if status == "unverified"
            else "Verification timed out; treated as unsupported."
        )
        return self._make_citation(task, status=status, reasoning=msg)

    def _make_citation(
        self, task: _VerificationTask, status: str, reasoning: str
    ) -> Citation:
        chunk = task.chunk
        payload = (chunk.candidate.payload if chunk else {}) or {}
        preview = ""
        if chunk and chunk.candidate.text:
            preview = chunk.candidate.text.strip().replace("\n", " ")
            if len(preview) > _CHUNK_PREVIEW_CHARS:
                preview = preview[:_CHUNK_PREVIEW_CHARS] + "…"
        return Citation(
            idx=task.idx,
            status=status,
            source_filename=str(payload.get("source_filename", "")),
            section=str(payload.get("section", "")),
            page=int(payload.get("page", 0) or 0),
            chunk_text_preview=preview,
            reasoning=reasoning,
        )

    def _build_unverified_citations(
        self, claims: List[Claim], chunks: List[Reranked]
    ) -> List[Citation]:
        citations: List[Citation] = []
        for task in self._enumerate_tasks(claims, chunks):
            citations.append(
                self._make_citation(
                    task,
                    status="unverified",
                    reasoning="Verification disabled by configuration.",
                )
            )
        return citations

    def _assemble(self, claims: List[Claim], citations: List[Citation]) -> VerifiedAnswer:
        supported = sum(1 for c in citations if c.status == "supported")
        unsupported = sum(1 for c in citations if c.status == "unsupported")
        return VerifiedAnswer(
            text="",  # populated by the pipeline from the RawAnswer
            claims=claims,
            citations=citations,
            supported_count=supported,
            unsupported_count=unsupported,
            total_citations=len(citations),
        )
