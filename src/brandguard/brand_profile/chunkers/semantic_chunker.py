"""Semantic chunker — splits at low-similarity sentence boundaries."""

from __future__ import annotations
import logging
from typing import Dict, List

from .base import Chunk, Chunker
from ..loaders.base import RawDocument

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, config: Dict):
        self.model_name = config.get("embedding_model", "intfloat/multilingual-e5-large")
        self.breakpoint_percentile = float(config.get("breakpoint_percentile", 90))
        self.min_chunk_chars = int(config.get("min_chunk_chars", 200))
        self.max_chunk_chars = int(config.get("max_chunk_chars", 1500))
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading semantic-chunk embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def split(self, doc: RawDocument) -> List[Chunk]:
        text = doc.plaintext
        if not text.strip():
            return []

        sentences = self._sentences(text)
        if len(sentences) <= 1:
            return self._single_chunk(doc, text)

        # Locate each sentence's absolute offset
        offsets: List[int] = []
        cursor = 0
        for sent in sentences:
            idx = text.find(sent, cursor)
            if idx < 0:
                idx = cursor
            offsets.append(idx)
            cursor = idx + len(sent)

        vectors = self._get_model().encode(
            ["passage: " + s for s in sentences], normalize_embeddings=True
        )

        # Cosine similarity of consecutive pairs
        import numpy as np
        sims = []
        for i in range(len(sentences) - 1):
            sims.append(float(np.dot(vectors[i], vectors[i + 1])))

        # Breakpoints: similarity below the (100 - percentile) threshold → split
        distances = [1.0 - s for s in sims]
        if not distances:
            return self._single_chunk(doc, text)
        threshold = float(np.percentile(distances, self.breakpoint_percentile))

        breakpoints = {i for i, d in enumerate(distances) if d >= threshold}

        chunks: List[Chunk] = []
        idx = 0
        group: List[int] = []
        for i in range(len(sentences)):
            group.append(i)
            current_text = " ".join(sentences[g] for g in group).strip()
            is_last = i == len(sentences) - 1
            should_split = (i in breakpoints) or is_last or len(current_text) >= self.max_chunk_chars
            if should_split and len(current_text) >= self.min_chunk_chars or is_last:
                start_off = offsets[group[0]]
                chunks.append(Chunk(
                    text=current_text,
                    section=doc.section_for_char(start_off),
                    page=doc.page_for_char(start_off),
                    char_count=len(current_text),
                    strategy=self.name,
                    chunk_index=idx,
                    source_filename=doc.source_filename,
                ))
                idx += 1
                group = []
        return chunks

    def _single_chunk(self, doc: RawDocument, text: str) -> List[Chunk]:
        stripped = text.strip()
        if not stripped:
            return []
        return [Chunk(
            text=stripped,
            section=doc.section_for_char(0),
            page=doc.page_for_char(0),
            char_count=len(stripped),
            strategy=self.name,
            chunk_index=0,
            source_filename=doc.source_filename,
        )]

    def _sentences(self, text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except LookupError:
                import nltk
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            import re
            return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
