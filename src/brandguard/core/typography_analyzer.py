"""
Typography Analysis Module
Handles font identification and typography validation using FontComplianceChecker
"""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class TypographyAnalyzer:
    """Handles typography analysis using FontComplianceChecker from FontTypographyChecker"""
    
    def __init__(self, settings, imported_models: Dict[str, Any], lang: str = 'en'):
        """Initialize the typography analyzer"""
        self.settings = settings
        self.imported_models = imported_models
        self.lang = lang
        self.font_compliance_checker = None
        
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
                
        except Exception as e:
            logger.error(f"Typography analysis initialization failed: {e}")
            import traceback
            logger.error(f"Typography initialization traceback: {traceback.format_exc()}")
            self.font_compliance_checker = None
    
    def analyze_typography(
        self,
        image: np.ndarray,
        text_regions: Optional[List[Dict]] = None,
        rag_context: str = "",
        few_shot_examples: Optional[List[Dict]] = None,
        brand_rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive typography analysis using FontComplianceChecker
        
        Args:
            image: Input image as numpy array
            text_regions: Optional list of text regions to analyze
            
        Returns:
            Dictionary containing typography analysis results
        """
        try:
            logger.info("🔍 Starting typography analysis...")

            if rag_context:
                logger.info(f"Typography RAG context ({len(rag_context)} chars) available")

            if self.font_compliance_checker:
                # Use FontComplianceChecker for comprehensive analysis
                result = self._analyze_with_font_compliance_checker(image)
            else:
                # Fallback to basic analysis
                result = self._fallback_typography_analysis(image, text_regions)

            # Overlay brand profile rules when provided (overrides YAML defaults)
            if brand_rules:
                result = self._apply_brand_typography_rules(result, brand_rules, rag_context)

            return result
            
        except Exception as e:
            logger.error(f"Typography analysis failed: {e}")
            import traceback
            logger.error(f"Typography analysis traceback: {traceback.format_exc()}")
            return {
                'fonts_detected': [],
                'font_compliance': {},
                'typography_score': 0.0,
                'recommendations': ['Typography analysis failed due to technical error'],
                'errors': [str(e)]
            }
    
    def _analyze_with_font_compliance_checker(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze typography using FontComplianceChecker"""
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
            
            # Analyze with FontComplianceChecker
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
            
            # Convert FontComplianceChecker results to our format
            return self._convert_font_compliance_results(analysis_results)
            
        except Exception as e:
            logger.error(f"FontComplianceChecker analysis failed: {e}")
            return self._fallback_typography_analysis(image, None)
    
    def _convert_font_compliance_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert FontComplianceChecker results to our format"""
        try:
            # Extract text regions and font analysis
            text_regions = analysis_results.get('text_regions', [])
            font_analysis = analysis_results.get('font_analysis', {})
            typography_validation = analysis_results.get('typography_validation', {})
            overall_compliance = analysis_results.get('overall_compliance', {})
            
            # Convert text regions to fonts_detected format
            fonts_detected = []
            for region in text_regions:
                font_info = {
                    'font_family': region.get('font_name', 'Unknown'),
                    'font_size': region.get('font_metrics', {}).get('font_size', 12),
                    'confidence': region.get('font_confidence', 0.5),
                    'text': region.get('text', ''),
                    'bbox': region.get('bbox', [0, 0, 0, 0]),
                    'area': region.get('area', 0),
                    'font_approved': region.get('font_approved', False)
                }
                fonts_detected.append(font_info)
            
            # Convert compliance results
            font_compliance = {
                'approved_fonts': [f for f in fonts_detected if f.get('font_approved', False)],
                'non_compliant_fonts': [f for f in fonts_detected if not f.get('font_approved', False)],
                'compliance_score': font_analysis.get('compliance_score', 0.0)
            }
            
            # Calculate typography score
            typography_score = overall_compliance.get('overall_score', 0.0)
            
            # Generate recommendations
            recommendations = overall_compliance.get('recommendations', [])
            if not recommendations:
                recommendations = self._generate_basic_recommendations(font_compliance)
            
            return {
                'fonts_detected': fonts_detected,
                'font_compliance': font_compliance,
                'typography_score': typography_score,
                'recommendations': recommendations,
                'errors': [],
                'text_regions': text_regions,
                'font_analysis': font_analysis,
                'typography_validation': typography_validation,
                'overall_compliance': overall_compliance
            }
            
        except Exception as e:
            logger.error(f"Error converting FontComplianceChecker results: {e}")
            return {
                'fonts_detected': [],
                'font_compliance': {},
                'typography_score': 0.0,
                'recommendations': ['Error processing typography results'],
                'errors': [str(e)]
            }
    
    def _apply_brand_typography_rules(
        self,
        result: Dict[str, Any],
        brand_rules: Dict[str, Any],
        rag_context: str = "",
    ) -> Dict[str, Any]:
        """
        Re-validate detected fonts against explicit brand typography rules.
        Overrides the YAML-based approved fonts list with the brand profile's font rules.
        """
        approved_fonts_lower = set()
        for key in ("bangla_font", "english_font"):
            val = brand_rules.get(key)
            if val:
                approved_fonts_lower.add(val.lower())
        for f in brand_rules.get("approved_fonts", []):
            approved_fonts_lower.add(f.lower())

        if not approved_fonts_lower:
            return result  # No brand rules to apply

        fonts_detected = result.get("fonts_detected", [])
        violations = []
        for font in fonts_detected:
            detected_name = (font.get("font_family") or "").lower()
            is_compliant = any(approved in detected_name or detected_name in approved for approved in approved_fonts_lower)
            font["font_approved"] = is_compliant
            if not is_compliant and font.get("text"):
                violations.append(
                    f"Font '{font.get('font_family', 'unknown')}' on text '{font['text'][:30]}' "
                    f"does not match brand fonts: {list(approved_fonts_lower)}"
                )

        compliant = [f for f in fonts_detected if f.get("font_approved")]
        non_compliant = [f for f in fonts_detected if not f.get("font_approved")]
        brand_score = len(compliant) / max(len(fonts_detected), 1)

        result["font_compliance"] = {
            "approved_fonts": compliant,
            "non_compliant_fonts": non_compliant,
            "compliance_score": round(brand_score, 2),
            "brand_violations": violations,
        }
        result["typography_score"] = round(brand_score, 2)

        if rag_context:
            result["rag_context_used"] = rag_context[:200]

        return result

    def _fallback_typography_analysis(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Fallback typography analysis when FontComplianceChecker is not available"""
        try:
            logger.info("Using fallback typography analysis")
            
            # Basic fallback analysis
            fonts_detected = self._fallback_font_detection(image)
            compliance_results = self._fallback_typography_validation(fonts_detected)
            typography_score = self._calculate_typography_score(compliance_results)
            recommendations = self._generate_typography_recommendations(compliance_results)
            
            return {
                'fonts_detected': fonts_detected,
                'font_compliance': compliance_results,
                'typography_score': typography_score,
                'recommendations': recommendations,
                'errors': ['Using fallback typography analysis - FontComplianceChecker not available']
            }
            
        except Exception as e:
            logger.error(f"Fallback typography analysis failed: {e}")
            return {
                'fonts_detected': [],
                'font_compliance': {},
                'typography_score': 0.0,
                'recommendations': ['Typography analysis failed'],
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
    
    def _fallback_font_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback font detection when real model is not available"""
        try:
            # Basic fallback - return generic font info
            return [{
                'font_family': 'Unknown',
                'font_size': 12,
                'confidence': 0.5,
                'region': 'entire_image'
            }]
        except Exception as e:
            logger.error(f"Fallback font detection failed: {e}")
            return []
    
    def _fallback_typography_validation(self, fonts_detected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback typography validation when real model is not available"""
        try:
            return {
                'approved_fonts': [],
                'non_compliant_fonts': fonts_detected,
                'compliance_score': 0.5  # Neutral score for fallback
            }
        except Exception as e:
            logger.error(f"Fallback typography validation failed: {e}")
            return {
                'approved_fonts': [],
                'non_compliant_fonts': fonts_detected,
                'compliance_score': 0.0
            }
    
    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        try:
            logger.info("Cleaning up typography analyzer...")
            
            # Clear FontComplianceChecker reference
            if hasattr(self, 'font_compliance_checker') and self.font_compliance_checker:
                # FontComplianceChecker doesn't have a cleanup method, just clear the reference
                del self.font_compliance_checker
                self.font_compliance_checker = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Typography analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during typography analyzer cleanup: {e}")