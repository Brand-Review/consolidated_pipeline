"""
Visual Signal Extraction Engine
Extracts verifiable signals from images - NO scoring, NO compliance judgment.
Pure signal extraction only.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)


class VisualSignalExtractor:
    """
    Visual Signal Extraction Engine for Brand Compliance.
    
    Role: Extract verifiable signals only - DO NOT score or judge compliance.
    """
    
    def __init__(self, settings=None, imported_models=None):
        self.settings = settings
        self.imported_models = imported_models or {}
        self.vllm_analyzer = None
        
        # Initialize VLLM analyzer if available for signal extraction
        self._initialize_vllm()
    
    def _initialize_vllm(self):
        """Initialize VLLM analyzer for visual signal extraction"""
        try:
            # Use imported models from model_imports (self-contained)
            if 'HybridToneAnalyzer' in self.imported_models and self.imported_models['HybridToneAnalyzer']:
                HybridToneAnalyzer = self.imported_models['HybridToneAnalyzer']
                self.vllm_analyzer = HybridToneAnalyzer()
                logger.info("✅ VisualSignalExtractor: VLLM analyzer initialized")
            elif 'VLLMToneAnalyzer' in self.imported_models and self.imported_models['VLLMToneAnalyzer']:
                VLLMToneAnalyzer = self.imported_models['VLLMToneAnalyzer']
                self.vllm_analyzer = VLLMToneAnalyzer()
                logger.info("✅ VisualSignalExtractor: VLLM analyzer initialized")
            else:
                logger.info("ℹ️ VisualSignalExtractor: VLLM analyzers not available, using fallback")
                self.vllm_analyzer = None
        except Exception as e:
            logger.warning(f"⚠️ VisualSignalExtractor: VLLM not available: {e}")
            self.vllm_analyzer = None
    
    def extract_signals(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract visual signals from image.
        
        RULES:
        - DO NOT score
        - DO NOT judge compliance
        - ONLY extract verifiable signals
        
        Returns:
            Dictionary with extracted signals:
            {
                "logoSignals": {...},
                "textSignals": {...},
                "colorSignals": {...},
                "signalsExtracted": bool,
                "confidence": float
            }
        """
        try:
            logger.info("[VisualSignalExtractor] Starting signal extraction...")
            
            signals = {
                "logoSignals": self._extract_logo_signals(image),
                "textSignals": self._extract_text_signals(image),
                "colorSignals": self._extract_color_signals(image),
                "signalsExtracted": False,
                "confidence": 0.0
            }
            
            # Determine if signals were successfully extracted
            has_logo_signal = signals["logoSignals"].get("detected", False) or signals["logoSignals"].get("failureReason") is not None
            has_text_signal = signals["textSignals"].get("visibleTextDetected", False) or signals["textSignals"].get("failureReason") is not None
            has_color_signal = len(signals["colorSignals"].get("dominantColors", [])) > 0 or signals["colorSignals"].get("failureReason") is not None
            
            signals["signalsExtracted"] = has_logo_signal or has_text_signal or has_color_signal
            
            # Calculate overall confidence (average of signal confidences)
            confidences = []
            if signals["logoSignals"].get("confidence"):
                confidences.append(signals["logoSignals"]["confidence"])
            if signals["textSignals"].get("confidence"):
                confidences.append(signals["textSignals"]["confidence"])
            if signals["colorSignals"].get("dominantColors"):
                color_confs = [c.get("confidence", 0.0) for c in signals["colorSignals"]["dominantColors"]]
                if color_confs:
                    confidences.append(sum(color_confs) / len(color_confs))
            
            signals["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(f"[VisualSignalExtractor] Signal extraction completed. Signals extracted: {signals['signalsExtracted']}, Confidence: {signals['confidence']:.2f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"[VisualSignalExtractor] Signal extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "logoSignals": {"detected": False, "logos": [], "failureReason": f"Signal extraction failed: {str(e)}"},
                "textSignals": {"visibleTextDetected": False, "confidence": 0.0, "extractedText": "", "spellingErrors": [], "phraseErrors": [], "failureReason": f"Signal extraction failed: {str(e)}"},
                "colorSignals": {"background": None, "primaryAccent": None, "textColor": None, "dominantColors": [], "failureReason": f"Signal extraction failed: {str(e)}"},
                "signalsExtracted": False,
                "confidence": 0.0
            }
    
    def _extract_logo_signals(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract logo signals - detect visible logos only (no placement validation).
        
        Returns:
            {
                "detected": bool,
                "logos": [{"type": str, "position": str, "confidence": float, "bbox": [x1, y1, x2, y2]}],
                "failureReason": null | str
            }
        """
        try:
            # Use logo detector if available
            logo_detections = []
            
            if self.imported_models.get('LogoDetector'):
                try:
                    LogoDetector = self.imported_models['LogoDetector']
                    logo_config = {
                        'type': 'yolos',
                        'confidence_threshold': 0.5,
                        'path': 'ellabettison/Logo-Detection-finetune'
                    }
                    detector = LogoDetector(logo_config)
                    if detector.load_model():
                        raw_detections = detector.detect_logos(image)
                        
                        # Format detections
                        for det in (raw_detections or []):
                            bbox = det.get('bbox', [])
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                img_h, img_w = image.shape[:2]
                                center_x = (x1 + x2) / 2 / img_w if img_w > 0 else 0.5
                                center_y = (y1 + y2) / 2 / img_h if img_h > 0 else 0.5
                                
                                # Determine position
                                if center_x < 0.33:
                                    position = 'top-left' if center_y < 0.33 else ('bottom-left' if center_y > 0.67 else 'left')
                                elif center_x > 0.67:
                                    position = 'top-right' if center_y < 0.33 else ('bottom-right' if center_y > 0.67 else 'right')
                                else:
                                    position = 'top-center' if center_y < 0.33 else ('bottom-center' if center_y > 0.67 else 'center')
                                
                                # Determine type (symbol, wordmark, combined) - simplified
                                logo_type = 'combined'  # Default, could be enhanced with classification
                                
                                logo_detections.append({
                                    "type": logo_type,
                                    "position": position,
                                    "confidence": float(det.get('confidence', 0.0)),
                                    "bbox": bbox
                                })
                except Exception as e:
                    logger.warning(f"[VisualSignalExtractor] Logo detection failed: {e}")
                    return {
                        "detected": False,
                        "logos": [],
                        "failureReason": f"Logo detection system error: {str(e)}"
                    }
            
            # If no detections but logo detector is available, it means no logos found (legitimate)
            if not logo_detections:
                return {
                    "detected": False,
                    "logos": [],
                    "failureReason": None  # No failure - just no logos detected
                }
            
            return {
                "detected": True,
                "logos": logo_detections,
                "failureReason": None
            }
            
        except Exception as e:
            logger.error(f"[VisualSignalExtractor] Logo signal extraction failed: {e}")
            return {
                "detected": False,
                "logos": [],
                "failureReason": f"Logo signal extraction failed: {str(e)}"
            }
    
    def _extract_text_signals(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text signals - check for visible text and spelling errors.
        
        Returns:
            {
                "visibleTextDetected": bool,
                "confidence": float,
                "extractedText": str,
                "spellingErrors": [...],
                "phraseErrors": [...],
                "failureReason": null | str
            }
        """
        try:
            from ..utils.image_analyzer import detect_visible_text_regions
            from ..utils.spell_checker import check_spelling
            
            # STEP 1: Check if visible text exists (NOT OCR, just visual detection)
            text_indicators = detect_visible_text_regions(image)
            visible_text_detected = text_indicators.get('has_text', False)
            text_confidence = text_indicators.get('confidence', 0.0)
            
            # STEP 2: Extract text using OCR (only if visible text detected)
            extracted_text = ""
            spelling_errors = []
            phrase_errors = []
            
            if visible_text_detected:
                # Extract text using OCR with preprocessing
                from ..utils.ocr_utils import preprocess_image_for_ocr
                
                ocr_image, preprocess_error = preprocess_image_for_ocr(image)
                if ocr_image is None:
                    logger.warning(f"[VisualSignals] OCR preprocessing failed: {preprocess_error}")
                    extracted_text = ""  # Continue with empty text, don't fail
                else:
                try:
                        # Try multiple OCR methods with preprocessed image
                    import pytesseract
                        from PIL import Image as PILImage
                        pil_image = PILImage.fromarray(ocr_image)  # Already RGB and upscaled
                        extracted_text = pytesseract.image_to_string(pil_image, lang='eng').strip()
                except:
                    try:
                        from paddleocr import PaddleOCR
                        if not hasattr(self, '_paddle_ocr'):
                            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                            ocr_result = self._paddle_ocr.ocr(ocr_image, cls=True)  # Use preprocessed RGB image
                        if ocr_result and ocr_result[0]:
                            text_lines = [line[1][0] for line in ocr_result[0] if line and len(line) > 1]
                            extracted_text = '\n'.join(text_lines).strip()
                    except:
                        pass
                
                # STEP 3: Check spelling if text was extracted and is clearly readable
                if extracted_text and len(extracted_text.split()) >= 3:
                    # Only check spelling if text is clearly readable (confidence > 0.7)
                    if text_confidence >= 0.7:
                        spelling_errors_raw, suggestions, phrase_violations = check_spelling(extracted_text)
                        
                        # Format spelling errors - only include those with confidence >= 0.85
                        for error in spelling_errors_raw:
                            if isinstance(error, dict):
                                error_confidence = error.get('confidence', 0.0)
                                if error_confidence >= 0.85:  # RULE: Only report high-confidence errors
                                    # Determine location (headline, body, cta)
                                    location = 'body'  # Default, could be enhanced with text structure analysis
                                    
                                    spelling_errors.append({
                                        "word": error.get('word', ''),
                                        "suggestion": error.get('correction'),
                                        "confidence": error_confidence,
                                        "location": location
                                    })
                        
                        # Format phrase errors
                        for violation in phrase_violations:
                            if isinstance(violation, dict):
                                phrase_errors.append({
                                    "phrase": violation.get('phrase', ''),
                                    "suggestion": violation.get('suggestion', ''),
                                    "severity": "low",  # Default, could be enhanced
                                    "confidence": violation.get('confidence', 0.85)
                                })
                
                # STEP 4: Check for OCR failure
                if visible_text_detected and not extracted_text and text_confidence >= 0.3:
                    return {
                        "visibleTextDetected": True,
                        "confidence": text_confidence,
                        "extractedText": "",
                        "spellingErrors": [],
                        "phraseErrors": [],
                        "failureReason": f"OCR failed to extract text despite {len(text_indicators.get('text_regions', []))} visible text regions detected (confidence: {text_confidence:.2f})"
                    }
            
            return {
                "visibleTextDetected": visible_text_detected,
                "confidence": text_confidence,
                "extractedText": extracted_text,
                "spellingErrors": spelling_errors,  # Only high-confidence errors (>= 0.85)
                "phraseErrors": phrase_errors,
                "failureReason": None
            }
            
        except Exception as e:
            logger.error(f"[VisualSignalExtractor] Text signal extraction failed: {e}")
            return {
                "visibleTextDetected": False,
                "confidence": 0.0,
                "extractedText": "",
                "spellingErrors": [],
                "phraseErrors": [],
                "failureReason": f"Text signal extraction failed: {str(e)}"
            }
    
    def _extract_color_signals(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract color signals - dominant colors only (no brand validation).
        
        Returns:
            {
                "background": {"hex": str, "name": str, "confidence": float},
                "primaryAccent": {"hex": str, "name": str, "confidence": float},
                "textColor": {"hex": str, "name": str, "confidence": float},
                "dominantColors": [...],
                "failureReason": null | str
            }
        """
        try:
            from ..core.color_analyzer import ColorAnalyzer
            
            # Extract dominant colors using color analyzer (signal extraction only)
            color_analyzer = ColorAnalyzer(self.settings, self.imported_models)
            color_result = color_analyzer.analyze_colors(image, {'n_colors': 8})
            
            dominant_colors = color_result.get('dominant_colors', [])
            
            # Format dominant colors
            formatted_colors = []
            for color_hex in dominant_colors[:8]:  # Top 8 colors
                if color_hex:
                    formatted_colors.append({
                        "hex": color_hex.upper() if color_hex.startswith('#') else f"#{color_hex.upper()}",
                        "name": None,  # Could be enhanced with color name lookup
                        "confidence": 0.8,  # Default confidence for extracted colors
                        "percentage": 12.5  # Approximate (1/8 of total)
                    })
            
            # Determine background, primary accent, and text color (simplified)
            background = formatted_colors[0] if formatted_colors else None
            primary_accent = formatted_colors[1] if len(formatted_colors) > 1 else None
            text_color = formatted_colors[2] if len(formatted_colors) > 2 else None
            
            return {
                "background": background or {"hex": None, "name": None, "confidence": 0.0},
                "primaryAccent": primary_accent or {"hex": None, "name": None, "confidence": 0.0},
                "textColor": text_color or {"hex": None, "name": None, "confidence": 0.0},
                "dominantColors": formatted_colors,
                "failureReason": None
            }
            
        except Exception as e:
            logger.error(f"[VisualSignalExtractor] Color signal extraction failed: {e}")
            return {
                "background": {"hex": None, "name": None, "confidence": 0.0},
                "primaryAccent": {"hex": None, "name": None, "confidence": 0.0},
                "textColor": {"hex": None, "name": None, "confidence": 0.0},
                "dominantColors": [],
                "failureReason": f"Color signal extraction failed: {str(e)}"
            }

