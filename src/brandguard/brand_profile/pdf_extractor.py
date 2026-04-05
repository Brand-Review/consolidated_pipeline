"""
PDF Rule Extractor
Parses brand guideline PDFs into:
  1. Section-chunked text segments (for RAG indexing)
  2. Structured rules JSON (via OpenRouter LLM)
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Section headings to split on (case-insensitive)
_SECTION_PATTERNS = re.compile(
    r"(?:^|\n)\s*(logo|color|colour|typography|font|brand voice|tone|copywriting|graphic|background|palette)\b",
    re.IGNORECASE,
)

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
    Extracts text chunks and structured rules from a brand guideline PDF.
    Uses PyMuPDF for text extraction and OpenRouter for structured rule extraction.
    """

    def __init__(self, openrouter_api_key: str = None, model: str = "openai/gpt-4o-mini"):
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model

    def extract(self, pdf_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse a PDF and return (structured_rules_dict, text_chunks_list).

        structured_rules_dict keys: color_rules, typography_rules, logo_rules, brand_voice_rules
        text_chunks_list items: {"text": str, "section": str, "page": int}
        """
        raw_text_by_page = self._extract_text_pages(pdf_path)
        full_text = "\n".join(raw_text_by_page.values())
        chunks = self._chunk_by_section(raw_text_by_page)
        structured_rules = self._extract_rules_via_llm(full_text)
        return structured_rules, chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text_pages(self, pdf_path: str) -> Dict[int, str]:
        """Return {page_number: text} using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf")

        pages: Dict[int, str] = {}
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                pages[i + 1] = page.get_text("text")
        return pages

    def _chunk_by_section(self, pages: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks by section heading.
        Returns list of {"text": str, "section": str, "page": int}.
        """
        chunks: List[Dict[str, Any]] = []
        current_section = "General"
        current_lines: List[str] = []
        current_page = 1

        for page_num, page_text in pages.items():
            for line in page_text.splitlines():
                m = _SECTION_PATTERNS.match(line.strip())
                if m:
                    # Flush current buffer as a chunk
                    if current_lines:
                        chunks.append({
                            "text": " ".join(current_lines).strip(),
                            "section": current_section,
                            "page": current_page,
                        })
                        current_lines = []
                    current_section = line.strip()
                    current_page = page_num
                else:
                    cleaned = line.strip()
                    if cleaned:
                        current_lines.append(cleaned)

        # Flush remainder
        if current_lines:
            chunks.append({
                "text": " ".join(current_lines).strip(),
                "section": current_section,
                "page": current_page,
            })

        # Remove empty chunks
        return [c for c in chunks if len(c["text"]) > 20]

    def _extract_rules_via_llm(self, full_text: str) -> Dict[str, Any]:
        """
        Call OpenRouter to extract structured brand rules from the guideline text.
        Falls back to empty rules dict on failure.
        """
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY set — returning empty rules")
            return self._empty_rules()

        try:
            import requests

            # Truncate to avoid token limits (~8k chars)
            truncated = full_text[:8000]
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": _OPENROUTER_RULE_EXTRACTION_PROMPT + truncated,
                    }
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
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
