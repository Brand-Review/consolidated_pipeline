"""
Typography Analysis Module
Handles font identification and typography validation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class TypographyAnalyzer:
    """Handles typography analysis including font identification and validation"""
    
    def __init__(self, settings, imported_models: Dict[str, Any]):
        """Initialize the typography analyzer"""
        self.settings = settings
        self.imported_models = imported_models
        self.font_identifier = None
        self.typography_validator = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize typography analysis components"""
        try:
            # Get FontIdentifier from imported models
            if 'FontIdentifier' in self.imported_models and self.imported_models['FontIdentifier']:
                self.font_identifier = self.imported_models['FontIdentifier']()
                logger.info("✅ FontIdentifier initialized with real model")
            else:
                logger.warning("⚠️ FontIdentifier not available, using fallback")
                self.font_identifier = None
            
            # Get TypographyValidator from imported models
            if 'TypographyValidator' in self.imported_models and self.imported_models['TypographyValidator']:
                self.typography_validator = self.imported_models['TypographyValidator']()
                logger.info("✅ TypographyValidator initialized with real model")
            else:
                logger.warning("⚠️ TypographyValidator not available, using fallback")
                self.typography_validator = None
                
        except Exception as e:
            logger.error(f"Typography analysis initialization failed: {e}")
            import traceback
            logger.error(f"Typography initialization traceback: {traceback.format_exc()}")
    
    def analyze_typography(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive typography analysis
        
        Args:
            image: Input image as numpy array
            text_regions: Optional list of text regions to analyze
            
        Returns:
            Dictionary containing typography analysis results
        """
        try:
            logger.info("🔍 Starting typography analysis...")
            
            # Initialize results
            results = {
                'fonts_detected': [],
                'font_compliance': {},
                'typography_score': 0.0,
                'recommendations': [],
                'errors': []
            }
            
            # Detect fonts in the image
            fonts_detected = self._detect_fonts(image, text_regions)
            results['fonts_detected'] = fonts_detected
            
            # Validate typography against brand rules
            compliance_results = self._validate_typography(fonts_detected)
            results['font_compliance'] = compliance_results
            
            # Calculate overall typography score
            typography_score = self._calculate_typography_score(compliance_results)
            results['typography_score'] = typography_score
            
            # Generate recommendations
            recommendations = self._generate_typography_recommendations(compliance_results)
            results['recommendations'] = recommendations
            
            logger.info(f"✅ Typography analysis completed. Score: {typography_score:.2f}")
            return results
            
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
    
    def _detect_fonts(self, image: np.ndarray, text_regions: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Detect fonts in the image"""
        try:
            fonts_detected = []
            
            if self.font_identifier:
                # Use real font identifier
                if text_regions:
                    for region in text_regions:
                        # Extract image region if region contains bbox
                        if 'bbox' in region:
                            x1, y1, x2, y2 = region['bbox']
                            region_image = image[y1:y2, x1:x2]
                            if region_image.size > 0:  # Check if region is valid
                                font_info = self.font_identifier.identify_font(region_image)
                                if font_info:
                                    fonts_detected.append(font_info)
                        else:
                            # If region is already image data
                            font_info = self.font_identifier.identify_font(region)
                            if font_info:
                                fonts_detected.append(font_info)
                else:
                    # Detect fonts in entire image
                    font_info = self.font_identifier.identify_font(image)
                    if font_info:
                        fonts_detected.append(font_info)
            else:
                # Fallback: basic font detection
                fonts_detected = self._fallback_font_detection(image)
            
            return fonts_detected
            
        except Exception as e:
            logger.error(f"Font detection failed: {e}")
            return []
    
    def _validate_typography(self, fonts_detected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate typography against brand rules"""
        try:
            compliance_results = {
                'approved_fonts': [],
                'non_compliant_fonts': [],
                'compliance_score': 0.0
            }
            
            if self.typography_validator and fonts_detected:
                # Use real typography validator
                for font_info in fonts_detected:
                    # Use validate_font_family method instead of validate_font
                    font_name = font_info.get('font_family', 'Unknown')
                    validation_result = self.typography_validator.validate_font_family(font_name)
                    is_compliant = validation_result.get('is_compliant', False)
                    if is_compliant:
                        compliance_results['approved_fonts'].append(font_info)
                    else:
                        compliance_results['non_compliant_fonts'].append(font_info)
                
                # Calculate compliance score
                total_fonts = len(fonts_detected)
                approved_fonts = len(compliance_results['approved_fonts'])
                compliance_results['compliance_score'] = approved_fonts / total_fonts if total_fonts > 0 else 0.0
            else:
                # Fallback validation
                compliance_results = self._fallback_typography_validation(fonts_detected)
            
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
            
            # Clear model references
            if hasattr(self, 'font_detector') and self.font_detector:
                if hasattr(self.font_detector, 'cleanup'):
                    self.font_detector.cleanup()
                del self.font_detector
                self.font_detector = None
            
            if hasattr(self, 'font_validator') and self.font_validator:
                if hasattr(self.font_validator, 'cleanup'):
                    self.font_validator.cleanup()
                del self.font_validator
                self.font_validator = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Typography analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during typography analyzer cleanup: {e}")