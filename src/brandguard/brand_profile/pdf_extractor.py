"""
PDF Rule Extractor
Extracts structured brand rules (colors, typography, logo, brand voice) from a
guideline PDF via OpenRouter. Plaintext extraction and chunking are now handled
by `loaders/pdf_loader.py` and `chunkers/`.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any, List, Tuple

from ..core.llm_client import LLMClient, LLMResponseError
from .loaders.pdf_loader import load_pdf
from .loaders.base import RawDocument
from .chunkers import get_chunker
from .rag_config import load_rag_config, apply_overrides

logger = logging.getLogger(__name__)

_OPENROUTER_RULE_EXTRACTION_PROMPT = """You are a brand compliance AI. Given the following brand guideline text, extract a JSON object with EXACTLY these keys:

{
  "color_rules": {
    "palette": ["#hex1", "#hex2"],
    "gradient": "description or null",
    "forbidden": [],
    "raw_description": "exact relevant text"
  },
  "typography_rules": {
    "bangla_font": "font name or null",
    "english_font": "font name or null",
    "approved_fonts": [],
    "forbidden_fonts": [],
    "raw_description": "exact relevant text"
  },
  "logo_rules": {
    "position": "top-right or null",
    "min_height_px": 85,
    "dark_bg_use_white": true,
    "colorful_on_white_only": true,
    "allowed_zones": ["top-right"],
    "raw_description": "exact relevant text"
  },
  "brand_voice_rules": {
    "tone": "description or null",
    "language": "description or null",
    "raw_description": "exact relevant text"
  }
}

Return ONLY the JSON object, no explanation. Brand guideline text:

"""


class PDFRuleExtractor:
    """
    Extracts structured brand rules from a PDF. Plaintext and chunks come
    from the shared loader + chunker pipeline so non-PDF formats share the
    same ingest path.
    """

    def __init__(self, openrouter_api_key: str = None, model: str = None):
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.llm = LLMClient(api_key=self.api_key, model=self.model)

    def extract(
        self,
        pdf_path: str,
        chunking_strategy: str = None,
        chunking_overrides: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], RawDocument, List[Dict[str, Any]]]:
        """
        Parse a PDF and return (structured_rules, raw_document, chunks).

        chunks are dicts of {text, section, page, char_count, strategy,
        chunk_index, source_filename}.
        """
        raw = load_pdf(pdf_path, filename=os.path.basename(pdf_path))
        chunks = self._chunk(raw, chunking_strategy, chunking_overrides)
        structured_rules = self._extract_rules_via_llm(raw.plaintext)
        return structured_rules, raw, chunks

    # ------------------------------------------------------------------

    def _chunk(
        self,
        raw: RawDocument,
        strategy: str = None,
        overrides: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        cfg = load_rag_config()
        if overrides:
            cfg = apply_overrides(cfg, {"chunking": overrides})
        chunking_cfg = cfg.get("chunking", {})
        strategy = strategy or chunking_cfg.get("default_strategy", "recursive")
        chunker = get_chunker(strategy, chunking_cfg)
        chunks = chunker.split(raw)
        return [c.__dict__ for c in chunks]

    def _extract_rules_via_llm(self, full_text: str) -> Dict[str, Any]:
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY set — returning empty rules")
            return self._empty_rules()

        try:
            truncated = full_text[:8000]
            messages = [
                {
                    "role": "user",
                    "content": _OPENROUTER_RULE_EXTRACTION_PROMPT + truncated,
                }
            ]
            result, _ = self.llm.chat(
                messages,
                response_format={"type": "json_object"},
                timeout=60,
            )
            return result
        except LLMResponseError as e:
            logger.error(f"LLM rule extraction failed: {e}")
            return self._empty_rules()

    @staticmethod
    def _empty_rules() -> Dict[str, Any]:
        return {
            "color_rules": {"palette": [], "gradient": None, "forbidden": [], "raw_description": ""},
            "typography_rules": {
                "bangla_font": None, "english_font": None,
                "approved_fonts": [], "forbidden_fonts": [], "raw_description": ""
            },
            "logo_rules": {
                "position": None, "min_height_px": None,
                "dark_bg_use_white": False, "colorful_on_white_only": False,
                "allowed_zones": [], "raw_description": ""
            },
            "brand_voice_rules": {"tone": None, "language": None, "raw_description": ""},
        }
