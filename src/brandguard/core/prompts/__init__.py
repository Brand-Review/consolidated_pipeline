"""
Prompt modules for sensor-only analysis
"""

from .text_extraction_v1 import get_text_extraction_prompt, PROMPT_VERSION as TEXT_EXTRACTION_VERSION
from .tone_classification_v1 import get_tone_classification_prompt, PROMPT_VERSION as TONE_CLASSIFICATION_VERSION
from .image_text_ocr_v1 import get_image_ocr_prompt, PROMPT_VERSION as IMAGE_OCR_VERSION
from .typography_detection_v1 import get_typography_detection_prompt, PROMPT_VERSION as TYPOGRAPHY_VERSION
from .brand_evidence_extraction_v1 import get_brand_evidence_prompt, PROMPT_VERSION as BRAND_EVIDENCE_VERSION
from .visual_signal_extractor_v1 import (
    get_visual_signal_extractor_system_prompt,
    get_visual_signal_extractor_prompt,
    get_visual_signal_extractor_complete,
    VISUAL_SIGNAL_EXTRACTOR_VERSION
)

__all__ = [
    'get_text_extraction_prompt',
    'get_tone_classification_prompt',
    'get_image_ocr_prompt',
    'get_typography_detection_prompt',
    'get_brand_evidence_prompt',
    'get_visual_signal_extractor_system_prompt',
    'get_visual_signal_extractor_prompt',
    'get_visual_signal_extractor_complete',
    'TEXT_EXTRACTION_VERSION',
    'TONE_CLASSIFICATION_VERSION',
    'IMAGE_OCR_VERSION',
    'TYPOGRAPHY_VERSION',
    'BRAND_EVIDENCE_VERSION',
    'VISUAL_SIGNAL_EXTRACTOR_VERSION',
]

