"""
OCR Engine
Dedicated OCR extraction module - signals only, no scoring, no compliance judgment.

Contract:
- Extract raw text from images
- Detect if text exists (without analyzing content)
- Report OCR failures with clear reasons
- NEVER compute spelling, grammar, or compliance
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import tempfile
import os
import traceback
from ..utils.ocr_utils import preprocess_image_for_ocr

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR Engine - Dedicated text extraction from images.
    
    Responsibilities:
    - Extract raw text from image
    - Detect if text exists (without analyzing content)
    - Report OCR failures with clear reasons
    
    FORBIDDEN:
    - No spelling checking
    - No grammar analysis
    - No compliance checking
    - No score computation
    """
    
    def __init__(self, vllm_analyzer: Optional[Any] = None):
        """
        Initialize OCR Engine.
        
        Args:
            vllm_analyzer: Optional VLLM analyzer for OCR fallback
        """
        self.vllm_analyzer = vllm_analyzer
        self._paddle_ocr = None
        
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text content from image using OCR.
        
        Contract (from ANALYZER_CONTRACT.md):
        {
            "hasText": bool,
            "text": str | null,
            "confidence": float,
            "failure": null | FailureDict
        }
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            Dictionary with OCR results following the contract:
            {
                "hasText": bool,  # True if text was successfully extracted
                "text": str | null,  # Extracted text or null if failed/no text
                "confidence": float,  # Confidence in extraction (0.0-1.0)
                "failure": null | {  # Failure object if OCR failed when text is visible
                    "reason": str,
                    "failure_type": "ocr_failure|no_text|invalid_input",
                    "recommendations": [str]
                },
                "methods_tried": [str],  # List of OCR methods attempted
                "timestamp": str  # ISO 8601 timestamp
            }
        """
        timestamp = datetime.now().isoformat()
        ocr_methods_tried = []
        
        try:
            # STEP 1: Preprocess image for OCR (BGR→RGB, 2× upscale, validation)
            ocr_image, preprocess_error = preprocess_image_for_ocr(image)
            if ocr_image is None:
                return {
                    "hasText": False,
                    "text": None,
                    "confidence": 0.0,
                    "failure": {
                        "reason": f"Image preprocessing failed: {preprocess_error}",
                        "failure_type": "invalid_input",
                        "recommendations": [
                            "Verify image format is supported",
                            "Check that image is properly loaded",
                            "Ensure image is not empty or masked"
                        ]
                    },
                    "methods_tried": [],
                    "timestamp": timestamp
                }
            
            # Try pytesseract first (most common OCR library)
            try:
                import pytesseract
                ocr_methods_tried.append('pytesseract')
                
                # Use preprocessed RGB image (already converted and upscaled)
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(ocr_image)
                
                # Perform OCR
                extracted_text = pytesseract.image_to_string(pil_image, lang='eng')
                # CRITICAL: Accept text even if confidence is low - preserve partial text
                if extracted_text and extracted_text.strip():
                    word_count = len(extracted_text.split())
                    logger.info(f"✅ OCR extracted {word_count} words using pytesseract")
                    return {
                        "hasText": True,
                        "text": extracted_text.strip(),  # Preserve text even if low confidence
                        "confidence": 0.90,  # High confidence for successful pytesseract extraction
                        "failure": None,
                        "methods_tried": ocr_methods_tried,
                        "timestamp": timestamp
                    }
                else:
                    logger.warning("pytesseract returned empty text, trying fallback methods")
            except ImportError:
                logger.warning("pytesseract not installed, trying fallback methods")
            except Exception as e:
                error_str = str(e).lower()
                if 'tesseract is not installed' in error_str or 'tesseract-ocr' in error_str:
                    logger.error("❌ Tesseract OCR binary not installed. Install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)")
                else:
                    logger.warning(f"pytesseract OCR failed: {e}")
            
            # Try PaddleOCR as fallback
            try:
                from paddleocr import PaddleOCR
                ocr_methods_tried.append('PaddleOCR')
                
                # Initialize PaddleOCR (lazy initialization)
                if self._paddle_ocr is None:
                    logger.info("Initializing PaddleOCR...")
                    self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                
                # Perform OCR on preprocessed RGB image
                ocr_result = self._paddle_ocr.ocr(ocr_image, cls=True)
                if ocr_result and ocr_result[0]:
                    # Extract text from all detected regions
                    text_lines = [line[1][0] for line in ocr_result[0] if line and len(line) > 1]
                    extracted_text = '\n'.join(text_lines)
                    # CRITICAL: Accept text even if confidence is low - preserve partial text
                    if extracted_text and extracted_text.strip():
                        word_count = len(extracted_text.split())
                        logger.info(f"✅ OCR extracted {word_count} words using PaddleOCR")
                        return {
                            "hasText": True,
                            "text": extracted_text.strip(),  # Preserve text even if low confidence
                            "confidence": 0.85,  # Good confidence for PaddleOCR
                            "failure": None,
                            "methods_tried": ocr_methods_tried,
                            "timestamp": timestamp
                        }
                    else:
                        logger.warning("PaddleOCR returned no text regions")
                else:
                    logger.warning("PaddleOCR returned empty result")
            except ImportError:
                logger.warning("PaddleOCR not installed, trying fallback methods")
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")
            
            # FIX: NO LLM USAGE FOR OCR - removed VLLM fallback
            # OCR must use only deterministic methods: PaddleOCR, EasyOCR, Tesseract
            
            # Try one more time with additional image enhancement (on already preprocessed image)
            try:
                logger.info("Attempting OCR with additional image enhancement...")
                # ocr_image is already RGB and upscaled, now enhance further
                # Convert to grayscale for thresholding
                gray = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply adaptive thresholding for better contrast
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                
                # Try pytesseract one more time with enhanced image
                try:
                    import pytesseract
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(thresh)
                    extracted_text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')
                    # CRITICAL: Accept text even with low confidence - preserve partial text
                    if extracted_text and extracted_text.strip() and len(extracted_text.strip()) > 2:
                        word_count = len(extracted_text.split())
                        logger.info(f"✅ OCR succeeded with additional enhancement: {word_count} words")
                        ocr_methods_tried.append('pytesseract_enhanced')
                        return {
                            "hasText": True,
                            "text": extracted_text.strip(),  # Preserve text even if low confidence
                            "confidence": 0.75,  # Lower confidence as additional preprocessing was needed
                            "failure": None,
                            "methods_tried": ocr_methods_tried,
                            "timestamp": timestamp
                        }
                except Exception as e:
                    logger.debug(f"Enhanced OCR attempt failed: {e}")
            except Exception as e:
                logger.debug(f"Image enhancement failed: {e}")
            
            # If all OCR methods failed, return failure response
            # NOTE: This is a "no_text" failure, not "ocr_failure"
            # "ocr_failure" is only returned when text IS visible but OCR failed to extract it
            # That determination is made by the orchestrator (comparing with vision analyzer)
            logger.warning(f"⚠️ All OCR methods failed. Methods tried: {ocr_methods_tried}")
            
            diagnostic_msg = self._get_ocr_diagnostics(image, ocr_methods_tried)
            
            return {
                "hasText": False,
                "text": None,
                "confidence": 0.0,
                "failure": {
                    "reason": f"OCR failed to extract text from image. Methods tried: {', '.join(ocr_methods_tried)}",
                    "failure_type": "no_text",  # Could be "ocr_failure" if orchestrator determines text is visible
                    "recommendations": [
                        "Install tesseract-ocr binary: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)",
                        "Install pytesseract: pip install pytesseract",
                        "Alternatively, install PaddleOCR: pip install paddlepaddle paddleocr",
                        "Check image quality and ensure text is clearly visible",
                        diagnostic_msg
                    ]
                },
                "methods_tried": ocr_methods_tried,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"OCR Engine: Text extraction failed: {e}")
            logger.error(f"OCR traceback: {traceback.format_exc()}")
            
            return {
                "hasText": False,
                "text": None,
                "confidence": 0.0,
                "failure": {
                    "reason": f"OCR Engine encountered an unexpected error: {str(e)}",
                    "failure_type": "invalid_input",
                    "recommendations": [
                        "Verify image format is supported (numpy array, BGR format)",
                        "Check image integrity",
                        "Review logs for detailed error information"
                    ]
                },
                "methods_tried": ocr_methods_tried if 'ocr_methods_tried' in locals() else [],
                "timestamp": timestamp
            }
    
    def _get_ocr_diagnostics(self, image: np.ndarray, methods_tried: List[str]) -> str:
        """
        Get diagnostic information about OCR attempt.
        
        Args:
            image: Image that was processed
            methods_tried: List of OCR methods that were attempted
            
        Returns:
            Diagnostic message string
        """
        try:
            diagnostics = []
            diagnostics.append(f"Image shape: {image.shape if isinstance(image, np.ndarray) else 'unknown'}")
            diagnostics.append(f"Image dtype: {image.dtype if isinstance(image, np.ndarray) else 'unknown'}")
            
            if isinstance(image, np.ndarray):
                diagnostics.append(f"Image min/max values: {image.min()}/{image.max()}")
                diagnostics.append(f"Image mean: {image.mean():.2f}")
            
            # Check which OCR libraries are available
            available_libs = []
            try:
                import pytesseract
                available_libs.append("pytesseract: available")
            except ImportError:
                available_libs.append("pytesseract: NOT INSTALLED")
            
            try:
                from paddleocr import PaddleOCR
                available_libs.append("PaddleOCR: available")
            except ImportError:
                available_libs.append("PaddleOCR: NOT INSTALLED")
            
            if self.vllm_analyzer:
                available_libs.append("VLLM: available")
            else:
                available_libs.append("VLLM: not available")
            
            return f"Diagnostics: {' | '.join(diagnostics)} | Available: {' | '.join(available_libs)}"
        except Exception as e:
            return f"Diagnostics collection failed: {str(e)}"

