"""
Typography Analysis Module
Handles typography classification and validation using observable metrics only.
DOES NOT attempt to identify exact font names from raster images.
"""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from .typography_classifier import TypographyClassifier
from .typography_style_clusterer import TypographyStyleClusterer

logger = logging.getLogger(__name__)

class TypographyAnalyzer:
    """Handles typography analysis using FontComplianceChecker from FontTypographyChecker"""
    
    def __init__(self, settings, imported_models: Dict[str, Any], lang: str = 'en'):
        """Initialize the typography analyzer"""
        self.settings = settings
        self.imported_models = imported_models
        self.lang = lang
        self.font_compliance_checker = None
        self.font_identifier = None  # Multilingual font identifier
        self.typography_classifier = TypographyClassifier()  # Typography classifier
        self.style_clusterer = TypographyStyleClusterer(self.typography_classifier)  # NEW: Style clusterer
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize typography analysis components"""
        try:
            # Try to get FontComplianceChecker from imported models
            if 'FontComplianceChecker' in self.imported_models and self.imported_models['FontComplianceChecker']:
                try:
                    FontComplianceChecker = self.imported_models['FontComplianceChecker']
                    self.font_compliance_checker = FontComplianceChecker(
                        use_gpu=False,
                        lang=self.lang
                    )
                    logger.info(f"✅ FontComplianceChecker initialized with language: {self.lang}")
                except Exception as e:
                    logger.warning(f"⚠️ FontComplianceChecker initialization failed: {e}, using fallback")
                    self.font_compliance_checker = None
            else:
                logger.warning("⚠️ FontComplianceChecker not available, using fallback")
                self.font_compliance_checker = None
            
            # Initialize MultilingualFontIdentifier (PaddleOCR + HuggingFace)
            if 'FontIdentifier' in self.imported_models and self.imported_models['FontIdentifier']:
                try:
                    FontIdentifierFactory = self.imported_models['FontIdentifier']
                    self.font_identifier = FontIdentifierFactory(lang=self.lang)
                    logger.info(f"✅ MultilingualFontIdentifier initialized with language: {self.lang}")
                except Exception as e:
                    logger.warning(f"⚠️ MultilingualFontIdentifier initialization failed: {e}")
                    self.font_identifier = None
            else:
                logger.warning("⚠️ MultilingualFontIdentifier not available")
                self.font_identifier = None
                
        except Exception as e:
            logger.error(f"Typography analysis initialization failed: {e}")
            import traceback
            logger.error(f"Typography initialization traceback: {traceback.format_exc()}")
            self.font_compliance_checker = None
            self.font_identifier = None
    
    def analyze_typography(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze typography from raster image.
        
        CRITICAL RULES:
        - CANNOT reliably identify font families from raster images
        - DO NOT guess font names
        - Only evaluate: text hierarchy, contrast, readability
        - If brand font list is missing → status MUST be "unknown"
        - No compliance score if font is unknown
        
        Args:
            image: Input image as numpy array
            text_regions: Optional list of text regions to analyze
            options: Analysis options including expected_fonts
            
        Returns:
            Dictionary containing typography observations, limitations, and status
        """
        try:
            logger.info("🔍 Starting typography analysis (raster image - no font identification)...")
            
            # CRITICAL: Check if brand fonts are provided
            expected_fonts = None
            if options:
                expected_fonts = options.get('expected_fonts', '') or options.get('expected_fonts', '')
            
            # Parse expected fonts if provided
            expected_font_list = []
            if expected_fonts and expected_fonts.strip():
                expected_font_list = [f.strip() for f in expected_fonts.split(',') if f.strip()]
            
            # CRITICAL: If brand font list is missing → status MUST be "observed"
            if not expected_font_list:
                logger.warning("⚠️ Brand font guidelines not provided - typography status will be 'observed'")
                return self._analyze_without_font_guidelines(image, text_regions)
            
            # If brand fonts are provided, we can still only evaluate hierarchy, contrast, readability
            # We CANNOT validate font families from raster images
            return self._analyze_typography_observations(image, text_regions, expected_font_list)
            
        except Exception as e:
            logger.error(f"Typography analysis failed: {e}")
            import traceback
            logger.error(f"Typography analysis traceback: {traceback.format_exc()}")
            return {
                'status': 'failed',
                'reason': f'Analysis error: {str(e)}',
                'observations': {},
                'limitations': ['Typography analysis failed due to technical error'],
                'typography_score': 0.0,
                'errors': [str(e)]
            }
    
    def _analyze_with_font_compliance_checker(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze typography using FontComplianceChecker for text region extraction only.
        
        CRITICAL: We DO NOT trust font family names from raster images.
        We only use FontComplianceChecker to extract text regions for hierarchy/contrast/readability analysis.
        """
        try:
            # Save image to temporary file for FontComplianceChecker
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Convert numpy array to PIL Image and save
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB conversion
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
                
                pil_image.save(tmp_file.name, 'JPEG')
                tmp_file_path = tmp_file.name
            
            # Analyze with FontComplianceChecker (for text region extraction)
            analysis_results = self.font_compliance_checker.analyze_image(
                tmp_file_path,
                merge_nearby_regions=True,
                distance_threshold=20
            )
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            # CRITICAL: Convert results but DO NOT trust font family names
            return self._convert_font_compliance_results(analysis_results, image)
            
        except Exception as e:
            logger.error(f"FontComplianceChecker analysis failed: {e}")
            # Fallback to observable-only analysis
            return self._analyze_typography_observations(image, None, [])
    
    def _convert_font_compliance_results(self, analysis_results: Dict[str, Any], image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Convert FontComplianceChecker results to our format.
        
        CRITICAL: We DO NOT trust font family names from raster image analysis.
        Font identification is unreliable - we mark all fonts as 'unknown' and only
        use the results for hierarchy, contrast, readability analysis.
        """
        try:
            # Extract text regions (for hierarchy/contrast/readability)
            text_regions = analysis_results.get('text_regions', [])
            
            # CRITICAL: DO NOT use font family names - they are unreliable from raster images
            # Convert text regions to format suitable for hierarchy/contrast/readability analysis
            processed_regions = []
            for region in text_regions:
                # Extract only reliable data: size, position, text
                font_metrics = region.get('font_metrics', {})
                bbox = region.get('bbox', [0, 0, 0, 0])
                
                # ✅ FIX: Remove font_family completely - do not expose it at all
                processed_regions.append({
                    'text': region.get('text', ''),
                    'bbox': bbox,
                    'font_size': font_metrics.get('font_size', bbox[3] - bbox[1] if len(bbox) >= 4 else 12)
                    # ✅ NO font_family, NO font_identification_reliable - font identity not exposed
                })
            
            # Analyze only observable metrics
            observations = {
                'text_hierarchy': self._analyze_text_hierarchy(processed_regions),
                'contrast': self._analyze_text_contrast(image, processed_regions) if image is not None else {'contrast_score': 0.0, 'observations': ['Image not available for contrast analysis']},
                'readability': self._analyze_readability(processed_regions)
            }
            
            # Calculate score based only on observable metrics
            typography_score = self._calculate_observable_score(observations)
            
            limitations = [
                'Font family identification not reliable from raster images - all fonts marked as "unknown"',
                'Font compliance cannot be validated without reliable font identification',
                'Only text hierarchy, contrast, and readability were evaluated'
            ]
            
            # Cluster into typography styles (no per-word output)
            image_h, image_w = image.shape[:2] if image is not None else (1000, 1000)
            prepared_regions = []
            for region in processed_regions:
                bbox = region.get('bbox', [])
                if isinstance(bbox, list) and len(bbox) == 4:
                    bbox_h = bbox[3] - bbox[1]
                    if bbox_h <= 1.0:
                        font_size_ratio = max(0.0, float(bbox_h))
                    else:
                        font_size_ratio = float(bbox_h) / float(image_h or 1)
                    approx_size_px = font_size_ratio * float(image_h or 1)
                    prepared_regions.append({
                        'text': region.get('text', ''),
                        'bbox': bbox,
                        'approximateSizePx': round(approx_size_px, 1),
                        'fontSizeRatio': round(font_size_ratio, 4)
                    })

            typography_styles = self.style_clusterer.cluster_text_regions(
                image,
                prepared_regions,
                image_h,
                image_w,
                hierarchy_detected=observations.get('text_hierarchy', {}).get('hierarchy_detected', False)
            )

            return {
                'status': 'passed',  # Passed for observable metrics
                'observations': observations,
                'limitations': limitations,
                'typography_score': typography_score,
                'compliance_score': None,  # No font compliance score
                'font_compliance': {
                    'status': 'unknown',
                    'reason': 'Font family identification not reliable from raster images',
                    'validated': False
                },
                'typographyStyles': typography_styles,
                'recommendations': self._generate_observable_recommendations(observations),
                'errors': []
            }
            
        except Exception as e:
            logger.error(f"Error converting FontComplianceChecker results: {e}")
            return {
                'status': 'failed',
                'reason': f'Error processing typography results: {str(e)}',
                'observations': {},
                'limitations': ['Error processing typography results'],
                'typography_score': None,
                'errors': [str(e)]
            }
    
    def _analyze_without_font_guidelines(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze typography when brand font guidelines are missing.
        
        CRITICAL: Status MUST be "observed" - no compliance score.
        Only returns observations (hierarchy, contrast, readability) + typography classification.
        """
        try:
            logger.info("Analyzing typography without font guidelines - status will be 'observed'")
            
            # Extract text regions if not provided
            if text_regions is None:
                text_regions = self._extract_text_regions(image, None)
            
            # Analyze what we CAN evaluate: hierarchy, contrast, readability
            observations = {
                'text_hierarchy': self._analyze_text_hierarchy(text_regions),
                'contrast': self._analyze_text_contrast(image, text_regions),
                'readability': self._analyze_readability(text_regions)
            }
            
            # Extract scores from observations
            contrast_score = observations.get('contrast', {}).get('score', 0.0) if isinstance(observations.get('contrast'), dict) else 0.0
            readability_score = observations.get('readability', {}).get('score', 0.0) if isinstance(observations.get('readability'), dict) else 0.0
            text_hierarchy_detected = observations.get('text_hierarchy', {}).get('detected', False) if isinstance(observations.get('text_hierarchy'), dict) else len(text_regions) > 1
            
            # ✅ ALWAYS compute observable score
            observable_score = self._calculate_observable_score(observations)
            
            # 🔥 CRITICAL: Cluster text regions into typography styles (2-5 max)
            # DO NOT add typography metadata to individual OCR words
            image_h, image_w = image.shape[:2] if image is not None else (1000, 1000)
            
            # Prepare text regions for clustering (ensure approximateSizePx exists)
            prepared_regions = []
            for region in text_regions:
                if isinstance(region, dict) and 'bbox' in region:
                    bbox = region['bbox']
                    if len(bbox) == 4:
                        # Calculate font size ratio (normalized to image height)
                        bbox_h = bbox[3] - bbox[1]
                        if bbox_h <= 1.0:
                            font_size_ratio = max(0.0, float(bbox_h))
                        else:
                            font_size_ratio = float(bbox_h) / float(image_h or 1)

                        approx_size_px = font_size_ratio * float(image_h or 1)

                        prepared_region = {
                            'text': region.get('text', ''),
                            'bbox': bbox,
                            'approximateSizePx': round(approx_size_px, 1),
                            'fontSizeRatio': round(font_size_ratio, 4)
                            # ✅ NO fontClassification, NO confidence - OCR tokens excluded from typography output
                        }
                        prepared_regions.append(prepared_region)
            
            # Cluster into typography styles (2-5 max)
            typography_styles = self.style_clusterer.cluster_text_regions(
                image,
                prepared_regions,
                image_h,
                image_w,
                hierarchy_detected=text_hierarchy_detected
            )

            # Ensure styles are not empty when analyzer ran
            if not typography_styles:
                typography_styles = [{
                    'role': 'ui',
                    'confidenceLevel': 'low'
                }]
            
            # Collect limitations and similar fonts from styles
            all_limitations = set()
            all_similar_fonts = []
            for style in typography_styles:
                # Similar fonts are already aggregated in styles
                pass
            
            # Build limitations list
            limitations = [
                'Brand font guidelines not provided - cannot validate font compliance',
                'Font family names cannot be reliably detected from raster images',
                'Only text hierarchy, contrast, readability, and typography style clustering were evaluated',
                'Typography styles are clustered from text regions - not per-word analysis'
            ]
            limitations.extend(sorted(all_limitations))
            
            return {
                'status': 'observed',  # ✅ Changed from 'unknown' to 'observed'
                'visibilityState': 'observed',
                'reason': 'Brand font guidelines not provided - font compliance cannot be determined',
                'message': 'Typography observed using OCR regions (no font rules provided)',
                'textRegionsCount': len(text_regions),
                'typographyStyles': typography_styles,  # ✅ NEW: Clustered styles (2-5 max)
                'observations': {
                    'textHierarchyDetected': text_hierarchy_detected,
                    'readabilityScore': readability_score,
                    'contrastScore': contrast_score,
                    'textRegionsCount': len(text_regions)
                },
                'observableScore': observable_score,  # ✅ ALWAYS computed from hierarchy + contrast + readability
                'limitations': limitations,
                'typography_score': None,  # No score when observed (no brand rules)
                'compliance_score': None,  # No compliance score
                'font_compliance': None,  # No font compliance data
                'recommendations': [
                    'Provide brand font guidelines to enable font compliance validation',
                    'Review text hierarchy, contrast, readability, and typography styles'
                ],
                'errors': []
                # ✅ NO confidence field - OCR confidence stays in OCR layer only
            }
            
        except Exception as e:
            logger.error(f"Typography analysis without guidelines failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Analysis error: {str(e)}',
                'observations': {},
                'limitations': ['Typography analysis failed'],
                'typography_score': None,
                'errors': [str(e)]
            }
    
    def _analyze_typography_observations(self, image: np.ndarray, text_regions: Optional[List[Dict]], expected_fonts: List[str]) -> Dict[str, Any]:
        """
        Analyze typography observations (hierarchy, contrast, readability).
        
        CRITICAL: We CANNOT reliably identify font families from raster images.
        We DO NOT guess font names.
        We only evaluate what we can observe: hierarchy, contrast, readability.
        """
        try:
            logger.info("Analyzing typography observations (hierarchy, contrast, readability)")
            
            # Extract text regions if not provided
            if text_regions is None:
                text_regions = self._extract_text_regions(image, None)
            
            # Analyze what we CAN evaluate
            observations = {
                'text_hierarchy': self._analyze_text_hierarchy(text_regions),
                'contrast': self._analyze_text_contrast(image, text_regions),
                'readability': self._analyze_readability(text_regions)
            }
            
            # 🔥 CRITICAL: Cluster text regions into typography styles (2-5 max)
            # DO NOT add typography metadata to individual OCR words
            image_h, image_w = image.shape[:2] if image is not None else (1000, 1000)
            
            # Prepare text regions for clustering (compute fontSizeRatio)
            prepared_regions = []
            for region in text_regions:
                if isinstance(region, dict) and 'bbox' in region:
                    bbox = region['bbox']
                    if len(bbox) == 4:
                        # Calculate font size ratio (normalized to image height)
                        bbox_h = bbox[3] - bbox[1]
                        if bbox_h <= 1.0:
                            font_size_ratio = max(0.0, float(bbox_h))
                        else:
                            font_size_ratio = float(bbox_h) / float(image_h or 1)

                        approx_size_px = font_size_ratio * float(image_h or 1)
                        
                        prepared_region = {
                            'text': region.get('text', ''),
                            'bbox': bbox,
                            'approximateSizePx': round(approx_size_px, 1),
                            'fontSizeRatio': round(font_size_ratio, 4)
                            # ✅ NO fontClassification, NO confidence - OCR tokens excluded from typography output
                        }
                        prepared_regions.append(prepared_region)
            
            # Cluster into typography styles (2-5 max)
            text_hierarchy_detected = observations.get('text_hierarchy', {}).get('detected', False) if isinstance(observations.get('text_hierarchy'), dict) else len(text_regions) > 1
            typography_styles = self.style_clusterer.cluster_text_regions(
                image,
                prepared_regions,
                image_h,
                image_w,
                hierarchy_detected=text_hierarchy_detected
            )

            # Ensure styles are not empty when analyzer ran
            if not typography_styles:
                typography_styles = [{
                    'role': 'ui',
                    'confidenceLevel': 'low'
                }]
            
            # Build limitations list
            limitations = [
                'Font family names cannot be reliably detected from raster images',
                'Font compliance cannot be validated without reliable font identification',
                'Only text hierarchy, contrast, readability, and typography style clustering were evaluated',
                'Typography styles are clustered from text regions - not per-word analysis'
            ]
            
            # Calculate score based only on observable metrics (not font compliance)
            typography_score = self._calculate_observable_score(observations)
            
            return {
                'status': 'observed',  # ✅ Changed to 'observed' (compliance only when brand rules exist)
                'observations': observations,
                'typographyStyles': typography_styles,  # ✅ Clustered styles (2-5 max)
                'limitations': limitations,
                'typography_score': typography_score,
                'observableScore': typography_score,  # ✅ ALWAYS computed from hierarchy + contrast + readability
                'compliance_score': None,  # No font compliance score (fonts cannot be validated)
                'font_compliance': {
                    'status': 'not_applicable',  # Changed from 'unknown' to 'not_applicable'
                    'reason': 'Font family identification not reliable from raster images',
                    'expected_fonts': expected_fonts,
                    'validated': False
                },
                'recommendations': self._generate_observable_recommendations(observations),
                'errors': []
                # ✅ NO confidence field - OCR confidence stays in OCR layer only
            }
            
        except Exception as e:
            logger.error(f"Typography observations analysis failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Analysis error: {str(e)}',
                'observations': {},
                'limitations': ['Typography analysis failed'],
                'typography_score': None,
                'errors': [str(e)]
            }
    
    def _generate_basic_recommendations(self, font_compliance: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations from font compliance data"""
        recommendations = []
        
        non_compliant_fonts = font_compliance.get('non_compliant_fonts', [])
        approved_fonts = font_compliance.get('approved_fonts', [])
        
        if non_compliant_fonts:
            recommendations.append(f"Replace {len(non_compliant_fonts)} non-compliant fonts with brand-approved alternatives")
        
        if not approved_fonts:
            recommendations.append("Use brand-approved fonts for better consistency")
        
        if font_compliance.get('compliance_score', 0.0) < 0.8:
            recommendations.append("Improve typography compliance to meet brand standards")
        
        return recommendations
    
    def update_language(self, lang: str):
        """Update the OCR language for text extraction"""
        try:
            self.lang = lang
            if self.font_compliance_checker:
                self.font_compliance_checker.update_ocr_language(lang)
                logger.info(f"Updated OCR language to: {lang}")
        except Exception as e:
            logger.error(f"Failed to update OCR language: {e}")
    
    def _validate_typography(self, fonts_detected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate typography against brand rules (fallback method)"""
        try:
            compliance_results = {
                'approved_fonts': [],
                'non_compliant_fonts': [],
                'compliance_score': 0.0
            }
            
            # Basic validation - mark all as non-compliant for fallback
            compliance_results['non_compliant_fonts'] = fonts_detected
            compliance_results['compliance_score'] = 0.5  # Neutral score for fallback
            
            return compliance_results
            
        except Exception as e:
            logger.error(f"Typography validation failed: {e}")
            return {
                'approved_fonts': [],
                'non_compliant_fonts': fonts_detected,
                'compliance_score': 0.0
            }
    
    def _calculate_typography_score(self, compliance_results: Dict[str, Any]) -> float:
        """Calculate overall typography score"""
        try:
            base_score = compliance_results.get('compliance_score', 0.0)
            
            # Apply additional scoring factors
            approved_count = len(compliance_results.get('approved_fonts', []))
            non_compliant_count = len(compliance_results.get('non_compliant_fonts', []))
            
            # Bonus for having approved fonts
            if approved_count > 0:
                base_score += 0.1
            
            # Penalty for non-compliant fonts
            if non_compliant_count > 0:
                base_score -= 0.1 * non_compliant_count
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Typography score calculation failed: {e}")
            return 0.0
    
    def _generate_typography_recommendations(self, compliance_results: Dict[str, Any]) -> List[str]:
        """Generate typography recommendations"""
        try:
            recommendations = []
            
            non_compliant_fonts = compliance_results.get('non_compliant_fonts', [])
            approved_fonts = compliance_results.get('approved_fonts', [])
            
            if non_compliant_fonts:
                recommendations.append(f"Replace {len(non_compliant_fonts)} non-compliant fonts with brand-approved alternatives")
            
            if not approved_fonts:
                recommendations.append("Use brand-approved fonts for better consistency")
            
            if compliance_results.get('compliance_score', 0.0) < 0.8:
                recommendations.append("Improve typography compliance to meet brand standards")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Typography recommendations generation failed: {e}")
            return ["Review typography for brand compliance"]
    
    # NOTE: Old fallback methods removed - now using _analyze_without_font_guidelines
    # and _analyze_typography_observations for all typography analysis
    
    def _extract_text_regions(self, image: np.ndarray, ocr_regions: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Extract text regions from image using OCR.
        Does NOT identify font families - only extracts text and bounding boxes.
        
        🔥 DECOUPLED FROM TESSERACT: Prefers OCR regions if provided, falls back to Tesseract only if needed.
        """
        try:
            # 🔥 Use provided OCR regions first (from Google Vision or other OCR)
            if ocr_regions and len(ocr_regions) > 0:
                logger.info(f"[Typography] Using {len(ocr_regions)} provided OCR text regions")
                return ocr_regions
            
            text_regions = []
            
            # Fallback: Try using Tesseract to extract text regions
            try:
                import pytesseract
                from PIL import Image as PILImage
                
                # Convert to PIL Image
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = PILImage.fromarray(image_rgb)
                else:
                    pil_image = PILImage.fromarray(image)
                
                # Get OCR data with bounding boxes
                ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                
                # Group text by regions (lines/paragraphs)
                current_line = []
                current_y = None
                
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    if text:
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        conf = ocr_data['conf'][i]
                        
                        # Group by vertical position (same line)
                        if current_y is None or abs(y - current_y) < h * 0.5:
                            current_line.append({
                                'text': text,
                                'bbox': [x, y, x + w, y + h],
                                'confidence': conf / 100.0
                            })
                            current_y = y
                        else:
                            # New line - save previous line
                            if current_line:
                                texts = [t['text'] for t in current_line]
                                bboxes = [t['bbox'] for t in current_line]
                                x_min = min(b[0] for b in bboxes)
                                y_min = min(b[1] for b in bboxes)
                                x_max = max(b[2] for b in bboxes)
                                y_max = max(b[3] for b in bboxes)
                                
                                text_regions.append({
                                    'text': ' '.join(texts),
                                    'bbox': [x_min, y_min, x_max, y_max],
                                    'font_size': h,  # Approximate font size from height
                                    'confidence': sum(t['confidence'] for t in current_line) / len(current_line)
                                })
                            
                            current_line = [{
                                'text': text,
                                'bbox': [x, y, x + w, y + h],
                                'confidence': conf / 100.0
                            }]
                            current_y = y
                
                # Add last line
                if current_line:
                    texts = [t['text'] for t in current_line]
                    bboxes = [t['bbox'] for t in current_line]
                    x_min = min(b[0] for b in bboxes)
                    y_min = min(b[1] for b in bboxes)
                    x_max = max(b[2] for b in bboxes)
                    y_max = max(b[3] for b in bboxes)
                    
                    text_regions.append({
                        'text': ' '.join(texts),
                        'bbox': [x_min, y_min, x_max, y_max],
                        'font_size': bboxes[0][3] - bboxes[0][1] if bboxes else 12,
                        'confidence': sum(t['confidence'] for t in current_line) / len(current_line)
                    })
                
            except ImportError:
                logger.warning("pytesseract not available for text region extraction")
            except Exception as e:
                logger.warning(f"Text region extraction failed: {e}")
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Text region extraction failed: {e}")
            return []
    
    def _analyze_text_hierarchy(self, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze text hierarchy (size differences, positioning).
        Does NOT identify font families - only evaluates size and position.
        """
        try:
            if not text_regions:
                return {
                    'hierarchy_detected': False,
                    'levels': [],
                    'size_variation': 0.0,
                    'observations': ['No text regions detected']
                }
            
            # Extract font sizes (approximate from bbox height)
            font_sizes = []
            for region in text_regions:
                bbox = region.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    height = bbox[3] - bbox[1]
                    font_sizes.append(height)
            
            if not font_sizes:
                return {
                    'hierarchy_detected': False,
                    'levels': [],
                    'size_variation': 0.0,
                    'observations': ['Could not determine font sizes from text regions']
                }
            
            # Analyze hierarchy
            min_size = min(font_sizes)
            max_size = max(font_sizes)
            size_variation = (max_size - min_size) / max_size if max_size > 0 else 0.0
            
            # Identify hierarchy levels (headline, subtext, body)
            levels = []
            if size_variation > 0.2:  # Significant size variation indicates hierarchy
                sorted_regions = sorted(text_regions, key=lambda r: (r.get('bbox', [0, 0, 0, 0])[1], -(r.get('bbox', [0, 0, 0, 0])[3] - r.get('bbox', [0, 0, 0, 0])[1])))  # Sort by Y position, then by size
                
                # First region is likely headline (top, largest)
                if sorted_regions:
                    headline = sorted_regions[0]
                    levels.append({
                        'level': 'headline',
                        'text': headline.get('text', '')[:50],
                        'size': headline.get('font_size', 0),
                        'position': 'top'
                    })
                
                # Remaining regions
                for i, region in enumerate(sorted_regions[1:], 1):
                    size = region.get('font_size', 0)
                    if size >= max_size * 0.8:
                        level_type = 'subtext'
                    else:
                        level_type = 'body'
                    
                    levels.append({
                        'level': level_type,
                        'text': region.get('text', '')[:50],
                        'size': size,
                        'position': 'middle' if i < len(sorted_regions) - 1 else 'bottom'
                    })
            
            observations = []
            if size_variation > 0.3:
                observations.append(f"Clear text hierarchy detected (size variation: {size_variation:.1%})")
            elif size_variation > 0.1:
                observations.append(f"Moderate text hierarchy (size variation: {size_variation:.1%})")
            else:
                observations.append("Limited text hierarchy - similar font sizes throughout")
            
            return {
                'hierarchy_detected': size_variation > 0.2,
                'levels': levels,
                'size_variation': size_variation,
                'min_size': min_size,
                'max_size': max_size,
                'observations': observations
            }
            
        except Exception as e:
            logger.error(f"Text hierarchy analysis failed: {e}")
            return {
                'hierarchy_detected': False,
                'levels': [],
                'size_variation': 0.0,
                'observations': [f'Analysis error: {str(e)}']
            }
    
    def _analyze_text_contrast(self, image: np.ndarray, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze text contrast against background.
        Does NOT identify font families - only evaluates contrast.
        """
        try:
            if not text_regions:
                return {
                    'contrast_score': 0.0,
                    'low_contrast_regions': [],
                    'observations': ['No text regions to analyze contrast']
                }
            
            # Convert to grayscale for contrast analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            low_contrast_regions = []
            contrast_scores = []
            
            for region in text_regions:
                bbox = region.get('bbox', [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure bbox is within image bounds
                h, w = gray.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract text region
                text_region = gray[y1:y2, x1:x2]
                if text_region.size == 0:
                    continue
                
                # Calculate contrast (standard deviation of pixel values)
                # Higher std = more contrast
                contrast_std = np.std(text_region)
                
                # Normalize to 0-1 scale (assuming good contrast is std > 30)
                contrast_score = min(1.0, contrast_std / 30.0)
                contrast_scores.append(contrast_score)
                
                # Flag low contrast regions
                if contrast_score < 0.5:  # Low contrast threshold
                    low_contrast_regions.append({
                        'text': region.get('text', '')[:50],
                        'contrast_score': contrast_score,
                        'bbox': bbox
                    })
            
            overall_contrast = sum(contrast_scores) / len(contrast_scores) if contrast_scores else 0.0
            
            observations = []
            if overall_contrast >= 0.7:
                observations.append("Good text contrast detected")
            elif overall_contrast >= 0.5:
                observations.append("Moderate text contrast")
            else:
                observations.append(f"Low text contrast detected ({len(low_contrast_regions)} regions)")
            
            return {
                'contrast_score': overall_contrast,
                'low_contrast_regions': low_contrast_regions,
                'observations': observations
            }
            
        except Exception as e:
            logger.error(f"Text contrast analysis failed: {e}")
            return {
                'contrast_score': 0.0,
                'low_contrast_regions': [],
                'observations': [f'Analysis error: {str(e)}']
            }
    
    def _analyze_readability(self, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze text readability (size, spacing).
        Does NOT identify font families - only evaluates size and spacing.
        """
        try:
            if not text_regions:
                return {
                    'readability_score': 0.0,
                    'small_text_regions': [],
                    'observations': ['No text regions to analyze readability']
                }
            
            small_text_regions = []
            readability_scores = []
            
            # Minimum readable size (pixels) - typically 10-12px minimum
            MIN_READABLE_SIZE = 10
            
            for region in text_regions:
                bbox = region.get('bbox', [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                
                # Approximate font size from bbox height
                font_size = bbox[3] - bbox[1]
                
                # Calculate readability score
                if font_size >= 16:
                    readability_score = 1.0
                elif font_size >= 12:
                    readability_score = 0.8
                elif font_size >= MIN_READABLE_SIZE:
                    readability_score = 0.6
                else:
                    readability_score = 0.3
                    small_text_regions.append({
                        'text': region.get('text', '')[:50],
                        'font_size': font_size,
                        'bbox': bbox
                    })
                
                readability_scores.append(readability_score)
            
            overall_readability = sum(readability_scores) / len(readability_scores) if readability_scores else 0.0
            
            observations = []
            if overall_readability >= 0.8:
                observations.append("Good text readability - appropriate font sizes")
            elif overall_readability >= 0.6:
                observations.append("Moderate text readability")
            else:
                observations.append(f"Poor text readability - {len(small_text_regions)} regions with small text")
            
            return {
                'readability_score': overall_readability,
                'small_text_regions': small_text_regions,
                'average_font_size': sum(r.get('font_size', 0) for r in text_regions) / len(text_regions) if text_regions else 0,
                'observations': observations
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {
                'readability_score': 0.0,
                'small_text_regions': [],
                'observations': [f'Analysis error: {str(e)}']
            }
    
    def _calculate_observable_score(self, observations: Dict[str, Any]) -> float:
        """
        Calculate typography score based only on observable metrics.
        Does NOT include font compliance (fonts are unknown).
        """
        try:
            hierarchy = observations.get('text_hierarchy', {})
            contrast = observations.get('contrast', {})
            readability = observations.get('readability', {})
            
            # Weighted average of observable metrics
            hierarchy_score = 0.5 if hierarchy.get('hierarchy_detected', False) else 0.3
            contrast_score = contrast.get('contrast_score', 0.0)
            readability_score = readability.get('readability_score', 0.0)
            
            # Weights: contrast 40%, readability 40%, hierarchy 20%
            observable_score = (
                contrast_score * 0.4 +
                readability_score * 0.4 +
                hierarchy_score * 0.2
            )
            
            return round(observable_score, 2)
            
        except Exception as e:
            logger.error(f"Observable score calculation failed: {e}")
            return 0.0
    
    def _generate_observable_recommendations(self, observations: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on observable metrics only"""
        recommendations = []
        
        hierarchy = observations.get('text_hierarchy', {})
        contrast = observations.get('contrast', {})
        readability = observations.get('readability', {})
        
        if not hierarchy.get('hierarchy_detected', False):
            recommendations.append("Improve text hierarchy - use varying font sizes to establish visual hierarchy")
        
        low_contrast = contrast.get('low_contrast_regions', [])
        if low_contrast:
            recommendations.append(f"Improve text contrast - {len(low_contrast)} regions have low contrast against background")
        
        small_text = readability.get('small_text_regions', [])
        if small_text:
            recommendations.append(f"Improve readability - {len(small_text)} regions have text that may be too small to read")
        
        if not recommendations:
            recommendations.append("Typography observations are acceptable - review hierarchy, contrast, and readability")
        
        return recommendations
    
    def detect_fonts(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Detect fonts in the image using MultilingualFontIdentifier.
        
        Args:
            image: Input image as numpy array
            text_regions: Optional text regions from OCR
            
        Returns:
            Dictionary with detected fonts and script analysis
        """
        if not self.font_identifier:
            return {
                "status": "unavailable",
                "reason": "MultilingualFontIdentifier not initialized",
                "fonts_detected": []
            }
        
        try:
            result = self.font_identifier.analyze(image, detect_fonts=True)
            return result
        except Exception as e:
            logger.error(f"Font detection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fonts_detected": []
            }
    
    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        try:
            logger.info("Cleaning up typography analyzer...")
            
            # Clear FontComplianceChecker reference
            if hasattr(self, 'font_compliance_checker') and self.font_compliance_checker:
                del self.font_compliance_checker
                self.font_compliance_checker = None
            
            # Clear MultilingualFontIdentifier reference
            if hasattr(self, 'font_identifier') and self.font_identifier:
                self.font_identifier.cleanup()
                self.font_identifier = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Typography analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during typography analyzer cleanup: {e}")