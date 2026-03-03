"""
OCR Preprocessing Utilities
Provides standardized image preprocessing for OCR operations.

CRITICAL RULES:
- Always convert BGR → RGB before OCR
- Upscale image 2× using INTER_CUBIC for better text recognition
- Never OCR empty or masked images
- Validate image before processing
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def preprocess_image_for_ocr(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Preprocess image for OCR operations.
    
    Steps:
    1. Validate image (not None, not empty, not masked)
    2. Convert BGR → RGB
    3. Upscale 2× using INTER_CUBIC
    
    Args:
        image: Input image as numpy array (BGR format expected)
        
    Returns:
        Tuple of (preprocessed_image, error_message)
        - preprocessed_image: Preprocessed RGB image (2× upscaled) or None if validation fails
        - error_message: Error description or None if successful
    """
    try:
        # Step 1: Validate input
        if image is None:
            return None, "Image is None"
        
        if not isinstance(image, np.ndarray):
            return None, f"Image must be numpy array, got {type(image)}"
        
        # Check if image is empty
        if image.size == 0:
            return None, "Image is empty (size = 0)"
        
        # Check image dimensions
        if len(image.shape) < 2:
            return None, f"Image has invalid shape: {image.shape}"
        
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None, f"Image has zero dimensions: {h}x{w}"
        
        # Check if image is completely masked (all zeros or all same value)
        if len(image.shape) == 3:
            # Color image - check if all pixels are the same (likely masked)
            unique_values = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
            if unique_values <= 2:  # Only background and maybe one other value
                logger.warning(f"[OCR Preprocess] Image appears masked (only {unique_values} unique colors)")
                # Don't fail, but log warning
        else:
            # Grayscale - check if all pixels are the same
            unique_values = len(np.unique(image))
            if unique_values <= 2:
                logger.warning(f"[OCR Preprocess] Image appears masked (only {unique_values} unique values)")
        
        # Step 2: Convert BGR → RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format (OpenCV default)
            ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # BGRA format - convert to RGB
            ocr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 2:
            # Grayscale - convert to RGB (3 channels)
            ocr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Already RGB or unknown format - use as is
            ocr_image = image.copy()
            logger.warning(f"[OCR Preprocess] Unknown image format, using as-is: shape={image.shape}")
        
        # Step 3: Upscale 2× using INTER_CUBIC
        # INTER_CUBIC provides better quality for text recognition than INTER_LINEAR
        new_width = w * 2
        new_height = h * 2
        ocr_image = cv2.resize(ocr_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        logger.debug(f"[OCR Preprocess] Preprocessed image: {w}x{h} → {new_width}x{new_height} (RGB)")
        
        return ocr_image, None
        
    except Exception as e:
        logger.error(f"[OCR Preprocess] Preprocessing failed: {e}", exc_info=True)
        return None, f"Preprocessing error: {str(e)}"


def validate_ocr_result(text: Optional[str], confidence: float, min_confidence_threshold: float = 0.0) -> Tuple[bool, str]:
    """
    Validate OCR result.
    
    CRITICAL RULES:
    - Low-confidence OCR is NOT a failure if text exists
    - Empty text is only a failure if we're certain no text exists
    - Always preserve partial text even with low confidence
    
    Args:
        text: Extracted text (can be None or empty)
        confidence: OCR confidence score (0.0-1.0)
        min_confidence_threshold: Minimum confidence to consider valid (default 0.0 = accept any)
        
    Returns:
        Tuple of (is_valid, reason)
        - is_valid: True if result should be accepted (even with low confidence)
        - reason: Explanation of validation result
    """
    # If text exists, it's valid (even with low confidence)
    if text and text.strip():
        if confidence < min_confidence_threshold:
            return True, f"Text extracted with low confidence ({confidence:.2f}) - preserving partial text"
        return True, "Text extracted successfully"
    
    # No text - only invalid if confidence suggests text should exist
    if confidence > 0.5:
        return False, f"No text extracted despite high confidence ({confidence:.2f})"
    
    return True, "No text detected (expected for images without text)"

