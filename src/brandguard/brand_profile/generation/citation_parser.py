"""Parse LLM answer text into Claims with citation indices.

Pure Python — no LLM calls. Given the generator's raw text, split on sentences,
extract `[N]` or `[N, M]` markers, and emit a `Claim` per sentence with:
  - `text`:            the sentence with leading/trailing whitespace stripped
  - `cited_indices`:   1-indexed chunk numbers referenced in the sentence
  - `char_span`:       (start, end) offsets into the original answer text

Also detects the IDK sentinel phrase:
  "I don't know based on the provided context."
so the pipeline can short-circuit verification and route to the IDK responder.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .types import Claim


# Matches [1], [1, 2], [1,2,3] — captures the inner digit list
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Exact sentinel phrase the grounded_answer prompt is told to emit on refusal.
_IDK_SENTINEL = "i don't know based on the provided context"


class CitationParser:
    """Stateless — instantiate once and reuse."""

    def said_idk(self, answer_text: str) -> bool:
        """True if the answer contains the exact IDK refusal phrase."""
        return _IDK_SENTINEL in (answer_text or "").lower()

    def parse(self, answer_text: str) -> List[Claim]:
        """Split `answer_text` into sentences and attach citation indices."""
        if not answer_text or not answer_text.strip():
            return []

        claims: List[Claim] = []
        for sentence_text, start, end in self._sentence_spans(answer_text):
            stripped = sentence_text.strip()
            if not stripped:
                continue
            indices = self._extract_indices(stripped)
            # Trim leading whitespace from the span so char_span aligns with stripped text's
            # position in the original answer.
            leading = len(sentence_text) - len(sentence_text.lstrip())
            trailing = len(sentence_text) - len(sentence_text.rstrip())
            claims.append(
                Claim(
                    text=stripped,
                    cited_indices=indices,
                    char_span=(start + leading, end - trailing),
                )
            )
        return claims

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _extract_indices(self, sentence: str) -> List[int]:
        """Return a de-duplicated, order-preserving list of cited chunk numbers."""
        seen: set = set()
        ordered: List[int] = []
        for match in _CITATION_PATTERN.finditer(sentence):
            for raw in match.group(1).split(","):
                raw = raw.strip()
                if not raw.isdigit():
                    continue
                n = int(raw)
                if n not in seen:
                    seen.add(n)
                    ordered.append(n)
        return ordered

    def _sentence_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Return `(sentence_text, start_offset, end_offset)` triples.

        Prefer NLTK's Punkt tokenizer (already installed for the semantic
        chunker). Fall back to a regex splitter on punctuation + whitespace so
        this module still works in test environments without the NLTK data.
        """
        sentences = self._tokenize(text)
        spans: List[Tuple[str, int, int]] = []
        cursor = 0
        for sent in sentences:
            if not sent:
                continue
            # Locate the sentence in the original text starting from the cursor.
            idx = text.find(sent, cursor)
            if idx == -1:
                # Tokenizer may have altered whitespace; fall back to cursor.
                idx = cursor
            end = idx + len(sent)
            spans.append((sent, idx, end))
            cursor = end
        return spans

    def _tokenize(self, text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            try:
                return [s for s in sent_tokenize(text) if s.strip()]
            except LookupError:
                import nltk
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                return [s for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
