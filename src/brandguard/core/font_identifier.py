"""
Multilingual Font Identifier Module
Combines PaddleOCR (multilingual text detection) with font identification.
Supports: English, Spanish, French, German (Latin) + observational for other scripts.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Supported languages for font identification (Latin scripts only)
FONT_SUPPORTED_LANGS = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'latin']

# PaddleOCR supported languages
PADDLEOCR_LANGS = {
    'en': 'English',
    'ch': 'Chinese',
    'zh': 'Chinese',
    'fr': 'French',
    'german': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'ja': 'Japanese',
    'korean': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ru': 'Russian',
    'fa': 'Persian',
    'ur': 'Urdu',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'pa': 'Punjabi',
    'gu': 'Gujarati',
    'or': 'Odia',
    'as': 'Assamese',
    'sa': 'Sanskrit',
    'ug': 'Uyghur',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'ku': 'Kurdish',
    'ckb': 'Sorani',
    'el': 'Greek',
    'he': 'Hebrew',
    'uk': 'Ukrainian',
    'bg': 'Bulgarian',
    'sr': 'Serbian',
    'mk': 'Macedonian',
    'mn': 'Mongolian',
    'hy': 'Armenian',
    'ka': 'Georgian',
    'be': 'Belarusian',
    'uk': 'Ukrainian',
    'pl': 'Polish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'nl': 'Dutch',
    'da': 'Danish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'tr': 'Turkish',
    'hr': 'Croatian',
    'sl': 'Slovenian',
    'ca': 'Catalan',
    'gl': 'Galician',
    'eu': 'Basque',
}


class MultilingualFontIdentifier:
    """
    Multilingual font identification using PaddleOCR + script detection + font ID.
    
    - Uses PaddleOCR for text detection in 80+ languages
    - Detects script type (Latin, Devanagari, Bengali, Arabic, CJK, etc.)
    - Font identification only for Latin scripts
    - Observational mode for non-Latin scripts
    """
    
    def __init__(self, lang: str = 'en', confidence_threshold: float = 0.3, use_gpu: bool = False):
        """
        Initialize MultilingualFontIdentifier.
        
        Args:
            lang: Primary language code for OCR
            confidence_threshold: Minimum confidence for font predictions
            use_gpu: Whether to use GPU
        """
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        self.paddle_ocr = None
        self.font_identifier = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize OCR and font identification components"""
        # Initialize EasyOCR (supports 80+ languages including Bengali, Hindi, Spanish)
        try:
            import easyocr
            
            # Map language codes to EasyOCR format
            lang_map = {
                'en': 'en', 'bn': 'bn', 'hi': 'hi', 'ar': 'ar',
                'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                'pt': 'pt', 'ru': 'ru', 'zh': 'ch_sim', 'ja': 'ja',
                'ko': 'ko', 'th': 'th', 'vi': 'vi', 'id': 'id'
            }
            ocr_lang = lang_map.get(self.lang, 'en')
            
            # Initialize EasyOCR reader
            self.easyocr_reader = easyocr.Reader(
                [ocr_lang],
                gpu=self.use_gpu,
                verbose=False
            )
            logger.info(f"✅ EasyOCR initialized with language: {ocr_lang}")
        except ImportError:
            logger.warning("⚠️ EasyOCR not installed. Install: pip install easyocr")
            self.easyocr_reader = None
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
        
        # Also try PaddleOCR as fallback
        self.paddle_ocr = None
        try:
            from paddleocr import PaddleOCR
            ocr_lang = self.lang if self.lang in PADDLEOCR_LANGS else 'en'
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True, 
                lang=ocr_lang
            )
            logger.info(f"✅ PaddleOCR initialized as fallback")
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR not available: {e}")
            self.paddle_ocr = None
        
        # Try Google Cloud Vision OCR as another option
        self.google_ocr_available = False
        try:
            from brandguard.ocr.google_ocr_engine import extract_text_google_vision
            # Test if credentials are available
            try:
                test_result = extract_text_google_vision(np.zeros((10, 10, 3), dtype=np.uint8))
                if test_result.get('error') is None or 'credentials' not in str(test_result.get('error', {})):
                    self.google_ocr_available = True
                    self.google_ocr_func = extract_text_google_vision
                    logger.info("✅ Google Cloud Vision OCR available")
            except Exception as e:
                if 'credentials' not in str(e).lower():
                    self.google_ocr_available = True
                    self.google_ocr_func = extract_text_google_vision
                    logger.info("✅ Google Cloud Vision OCR available")
                else:
                    logger.warning(f"⚠️ Google Cloud Vision credentials not configured")
        except ImportError:
            logger.warning("⚠️ Google Cloud Vision OCR not available")
        except Exception as e:
            logger.warning(f"⚠️ Google Cloud Vision OCR check failed: {e}")
        
        # Initialize font identifier for Latin scripts
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            # Use ariadnak font identifier for Latin fonts (1300+ fonts)
            MODEL_NAME = "ariadnak/font-identifier"
            
            self.font_identifier = {
                'processor': AutoImageProcessor.from_pretrained(MODEL_NAME),
                'model': AutoModelForImageClassification.from_pretrained(MODEL_NAME),
                'device': 'cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu',
                'model_name': MODEL_NAME
            }
            self.font_identifier['model'].to(self.font_identifier['device'])
            self.font_identifier['model'].eval()
            logger.info(f"✅ FontIdentifier loaded: {MODEL_NAME}")
        except Exception as e:
            logger.warning(f"⚠️ FontIdentifier (Latin) failed to load: {e}")
            self.font_identifier = None
    
    def _detect_script(self, text: str) -> str:
        """
        Detect script type from text.
        
        Returns: 'latin', 'devanagari', 'bengali', 'arabic', 'cjk', 'unknown'
        """
        if not text:
            return 'unknown'
        
        # Check for Devanagari (Hindi, Marathi, etc.)
        devanagari_range = range(0x0900, 0x097F)
        if any(ord(c) in devanagari_range for c in text):
            return 'devanagari'
        
        # Check for Bengali
        bengali_range = range(0x0980, 0x09FF)
        if any(ord(c) in bengali_range for c in text):
            return 'bengali'
        
        # Check for Arabic
        arabic_range = range(0x0600, 0x06FF)
        if any(ord(c) in arabic_range for c in text):
            return 'arabic'
        
        # Check for CJK
        cjk_ranges = [
            range(0x4E00, 0x9FFF),   # Chinese
            range(0x3040, 0x309F),   # Japanese Hiragana
            range(0x30A0, 0x30FF),   # Japanese Katakana
            range(0xAC00, 0xD7AF),  # Korean Hangul
        ]
        if any(any(ord(c) in r for c in text) for r in cjk_ranges):
            return 'cjk'
        
        # Check for Latin
        latin_range = range(0x0000, 0x024F)
        if any(ord(c) in latin_range for c in text):
            return 'latin'
        
        return 'unknown'
    
    def _identify_font(self, image: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Identify font in image using HuggingFace model"""
        if not self.font_identifier:
            return {"fonts_detected": [], "error": "FontIdentifier not loaded"}
        
        try:
            import torch
            
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            
            # Process
            processor = self.font_identifier['processor']
            model = self.font_identifier['model']
            device = self.font_identifier['device']
            
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k)
            
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            id2label = model.config.id2label
            
            fonts = []
            for prob, idx in zip(top_probs, top_indices):
                if prob >= self.confidence_threshold:
                    fonts.append({
                        "font_name": id2label.get(idx, f"unknown_{idx}"),
                        "confidence": float(prob),
                        "font_id": int(idx)
                    })
            
            return {
                "fonts_detected": fonts,
                "primary_font": fonts[0] if fonts else None,
                "total_fonts_matched": len(fonts)
            }
        except Exception as e:
            logger.error(f"Font identification error: {e}")
            return {"fonts_detected": [], "error": str(e)}
    
    def analyze(self, image: np.ndarray, detect_fonts: bool = True) -> Dict[str, Any]:
        """
        Analyze image for multilingual text and font identification.
        
        Args:
            image: Input image as numpy array
            detect_fonts: Whether to attempt font identification
            
        Returns:
            Dictionary with detected text, scripts, and fonts
        """
        results = {
            "status": "success",
            "text_regions": [],
            "script_analysis": {},
            "font_detection": {
                "status": "unsupported",
                "reason": "Font identification requires Latin script",
                "detected_fonts": []
            },
            "supported_scripts": list(PADDLEOCR_LANGS.keys())
        }
        
        # Try EasyOCR first, then PaddleOCR fallback
        ocr_result = None
        ocr_method = None
        
        if self.easyocr_reader is not None:
            try:
                # Run EasyOCR
                ocr_result = self.easyocr_reader.readtext(image)
                ocr_method = "EasyOCR"
                logger.info(f"✅ EasyOCR detected {len(ocr_result)} text regions")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR failed: {e}")
        
        # Fallback to PaddleOCR
        if ocr_result is None and self.paddle_ocr is not None:
            try:
                ocr_result = self.paddle_ocr.ocr(image, cls=True)
                ocr_method = "PaddleOCR"
                if ocr_result and ocr_result[0]:
                    logger.info(f"✅ PaddleOCR detected {len(ocr_result[0])} text regions")
            except Exception as e:
                logger.warning(f"⚠️ PaddleOCR failed: {e}")
        
        # Fallback to Google Cloud Vision OCR
        if ocr_result is None and self.google_ocr_available:
            try:
                height, width = image.shape[:2]
                google_result = self.google_ocr_func(image, image_width=width, image_height=height)
                if google_result.get('hasText') and not google_result.get('error'):
                    ocr_method = "GoogleVisionOCR"
                    # Convert Google result to internal format
                    ocr_result = []
                    for word in google_result.get('words', []):
                        # Denormalize bbox
                        x1 = int(word['bbox'][0] * width)
                        y1 = int(word['bbox'][1] * height)
                        x2 = int(word['bbox'][2] * width)
                        y2 = int(word['bbox'][3] * height)
                        ocr_result.append(([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], word['text'], word['confidence']))
                    logger.info(f"✅ Google Vision OCR detected {len(ocr_result)} text regions")
            except Exception as e:
                logger.warning(f"⚠️ Google Vision OCR failed: {e}")
        
        if ocr_result is None:
            results["status"] = "error"
            results["error"] = "No OCR available. Install EasyOCR or configure Google Cloud Vision credentials"
            return results
        
        try:
            text_regions = []
            scripts_detected = {}
            latin_regions = []
            
            # Process EasyOCR results
            if ocr_method == "EasyOCR":
                for item in ocr_result:
                    # EasyOCR format: (bbox, text, confidence)
                    bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = item[1]
                    conf = item[2]
                    
                    # Convert bbox to standard format
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    bbox_std = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    
                    script = self._detect_script(text)
                    scripts_detected[script] = scripts_detected.get(script, 0) + 1
                    
                    region_data = {
                        "text": text,
                        "confidence": float(conf),
                        "bbox": bbox_std,
                        "script": script,
                        "bbox_coords": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                    }
                    
                    if detect_fonts and script == 'latin':
                        if x2 > x1 and y2 > y1 and (x2-x1) > 20 and (y2-y1) > 20:
                            region_img = image[y1:y2, x1:x2]
                            if region_img.size > 0:
                                font_result = self._identify_font(region_img, top_k=3)
                                region_data["font_analysis"] = font_result
                                latin_regions.append(region_data)
                    
                    text_regions.append(region_data)
            
            # Process PaddleOCR results
            elif ocr_method == "PaddleOCR" and ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    bbox = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    conf = text_info[1]
                    script = self._detect_script(text)
                    
                    scripts_detected[script] = scripts_detected.get(script, 0) + 1
                    
                    region_data = {
                        "text": text,
                        "confidence": float(conf),
                        "bbox": bbox,
                        "script": script,
                        "bbox_coords": {
                            "x1": int(bbox[0][0]),
                            "y1": int(bbox[0][1]),
                            "x2": int(bbox[2][0]),
                            "y2": int(bbox[2][1])
                        }
                    }
                    
                    if detect_fonts and script == 'latin':
                        x1 = max(0, int(bbox[0][0]))
                        y1 = max(0, int(bbox[0][1]))
                        x2 = min(image.shape[1], int(bbox[2][0]))
                        y2 = min(image.shape[0], int(bbox[2][1]))
                        
                        if x2 > x1 and y2 > y1:
                            region_img = image[y1:y2, x1:x2]
                            if region_img.size > 0 and region_img.shape[0] > 20 and region_img.shape[1] > 20:
                                font_result = self._identify_font(region_img, top_k=3)
                                region_data["font_analysis"] = font_result
                                latin_regions.append(region_data)
                    
                    text_regions.append(region_data)
            
            # Process Google Vision OCR results
            elif ocr_method == "GoogleVisionOCR" and ocr_result:
                for item in ocr_result:
                    # Our converted format: (bbox, text, confidence)
                    bbox = item[0]  # [[x1,y1], [x2,y2], [x2,y2], [x1,y2]]
                    text = item[1]
                    conf = item[2]
                    
                    x1, y1 = bbox[0][0], bbox[0][1]
                    x2, y2 = bbox[2][0], bbox[2][1]
                    
                    script = self._detect_script(text)
                    scripts_detected[script] = scripts_detected.get(script, 0) + 1
                    
                    region_data = {
                        "text": text,
                        "confidence": float(conf),
                        "bbox": bbox,
                        "script": script,
                        "bbox_coords": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                    }
                    
                    if detect_fonts and script == 'latin':
                        if x2 > x1 and y2 > y1 and (x2-x1) > 20 and (y2-y1) > 20:
                            region_img = image[y1:y2, x1:x2]
                            if region_img.size > 0:
                                font_result = self._identify_font(region_img, top_k=3)
                                region_data["font_analysis"] = font_result
                                latin_regions.append(region_data)
                    
                    text_regions.append(region_data)
            
            results["text_regions"] = text_regions
            results["script_analysis"] = {
                "scripts_detected": scripts_detected,
                "primary_script": max(scripts_detected, key=scripts_detected.get) if scripts_detected else "unknown",
                "total_regions": len(text_regions)
            }
            
            # Font detection summary
            if latin_regions:
                all_fonts = []
                for region in latin_regions:
                    fonts = region.get("font_analysis", {}).get("fonts_detected", [])
                    if fonts:
                        all_fonts.extend(fonts)
                
                # Deduplicate by font name
                unique_fonts = {}
                for font in all_fonts:
                    name = font.get("font_name")
                    if name and (name not in unique_fonts or font.get("confidence", 0) > unique_fonts[name].get("confidence", 0)):
                        unique_fonts[name] = font
                
                results["font_detection"] = {
                    "status": "completed",
                    "script_supported": True,
                    "detected_fonts": list(unique_fonts.values())[:10],
                    "total_unique_fonts": len(unique_fonts),
                    "regions_analyzed": len(latin_regions)
                }
            else:
                primary = results["script_analysis"]["primary_script"]
                results["font_detection"] = {
                    "status": "observational_only",
                    "script_supported": False,
                    "reason": f"Primary script '{primary}' does not support font identification",
                    "suggestion": "Font identification available for Latin scripts only (en, es, fr, de, it, pt, etc.)"
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Multilingual font analysis error: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages for OCR"""
        return PADDLEOCR_LANGS
    
    def is_font_identification_supported(self, script: str) -> bool:
        """Check if font identification is supported for given script"""
        return script == 'latin'
    
    def cleanup(self):
        """Clean up resources"""
        if self.font_identifier:
            del self.font_identifier['model']
            del self.font_identifier['processor']
            self.font_identifier = None
        logger.info("MultilingualFontIdentifier cleanup completed")


def create_font_identifier(lang: str = 'en', use_gpu: bool = False) -> MultilingualFontIdentifier:
    """
    Factory function to create MultilingualFontIdentifier.
    
    Args:
        lang: Primary language code
        use_gpu: Whether to use GPU
        
    Returns:
        MultilingualFontIdentifier instance
    """
    return MultilingualFontIdentifier(lang=lang, use_gpu=use_gpu)
