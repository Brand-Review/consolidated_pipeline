"""
OCR Extractor
Outputs raw text and bounding boxes only.
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .google_ocr_engine import extract_text_google_vision

logger = logging.getLogger(__name__)


def extract_ocr(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Output only per block:
    - text
    - bbox
    - source = "ocr"
    """
    try:
        result = extract_text_google_vision(image)
        words = result.get("words", []) or []
        blocks: List[Dict[str, Any]] = []
        for word in words:
            text = word.get("text", "")
            bbox = word.get("bbox", [])
            if text and bbox:
                blocks.append({"text": text, "bbox": bbox, "source": "ocr"})

        return blocks
    except Exception as e:
        logger.warning(f"[OCRExtractor] OCR failed: {e}")
        return []

