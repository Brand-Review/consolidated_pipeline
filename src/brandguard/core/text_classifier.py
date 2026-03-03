"""
Text Classifier
Classifies OCR text blocks into marketing_copy | ui_text | data_table | decorative
"""

from typing import Dict, Any, List
import re

try:
    from ..utils.spell_checker import UI_WHITELIST
except Exception:
    UI_WHITELIST = set()

CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹", "₽"}
CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR"}


def _is_date(text: str) -> bool:
    patterns = [
        r"^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$",
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$"
    ]
    return any(re.match(p, text) for p in patterns)


def _is_table_like(text: str) -> bool:
    return bool(re.search(r"\d+\s*[\|\-]\s*\d+", text))


def classify_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classified = []
    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            label = "decorative"
        elif text.upper() in CURRENCY_CODES or text in CURRENCY_SYMBOLS:
            label = "data_table"
        elif any(ch.isdigit() for ch in text) or _is_date(text) or _is_table_like(text):
            label = "data_table"
        elif text.isupper() and len(text) <= 4:
            label = "ui_text"
        elif text.lower() in UI_WHITELIST:
            label = "ui_text"
        elif len(text) <= 2:
            label = "decorative"
        else:
            label = "marketing_copy"

        classified.append({
            "text": text,
            "bbox": block.get("bbox", []),
            "label": label
        })

    return classified

