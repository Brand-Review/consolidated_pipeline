"""
OCR Postprocessor
Filters OCR noise before semantic/LLM analysis.
"""

import re
from typing import Dict, Any, List, Optional

try:
    from ..utils.spell_checker import UI_WHITELIST
except Exception:
    UI_WHITELIST = set()

CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹", "₽"}
CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR"}


def _is_date(text: str) -> bool:
    date_patterns = [
        r"^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$",
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$"
    ]
    return any(re.match(pattern, text) for pattern in date_patterns)


def _is_percentage(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)?%$", text))


def filter_ocr_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filter OCR tokens before LLM.

    Drop tokens if:
    - length ≤ 2 AND numeric
    - ALL CAPS ≤ 4 chars (CRM, USD)
    - currency symbols or currency codes
    - dates, percentages
    - UI keywords
    """
    if not tokens:
        return {
            "cleanText": "",
            "removedTokens": [],
            "ocrReliability": 0.0
        }

    removed_tokens = []
    kept_tokens = []
    confidences = []

    for token in tokens:
        text = (token.get("text") or token.get("word") or "").strip()
        token_type = token.get("tokenType", "word")
        confidence = token.get("confidence", 0.0)

        if confidence:
            confidences.append(float(confidence))

        if not text:
            removed_tokens.append({**token, "reason": "empty"})
            continue

        # numeric short tokens
        if text.isdigit() and len(text) <= 2:
            removed_tokens.append({**token, "reason": "short numeric"})
            continue

        # all caps <= 4
        if text.isupper() and len(text) <= 4:
            removed_tokens.append({**token, "reason": "all_caps_short"})
            continue

        # currency symbols or codes
        if text in CURRENCY_SYMBOLS or text.upper() in CURRENCY_CODES or token_type == "currency":
            removed_tokens.append({**token, "reason": "currency"})
            continue

        # dates and percentages
        if _is_date(text) or _is_percentage(text):
            removed_tokens.append({**token, "reason": "date_or_percentage"})
            continue

        # UI keywords
        if text.lower() in UI_WHITELIST or token_type == "ui":
            removed_tokens.append({**token, "reason": "ui_keyword"})
            continue

        kept_tokens.append(text)

    clean_text = " ".join(kept_tokens).strip()
    ocr_reliability = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "cleanText": clean_text,
        "removedTokens": removed_tokens,
        "ocrReliability": round(max(0.0, min(1.0, ocr_reliability)), 2)
    }

