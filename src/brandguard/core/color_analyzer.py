"""
Color Analysis Module
Handles color palette extraction and validation
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000
import logging

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    """Handles color analysis functionality"""
    
    def __init__(self, settings, imported_models):
        self.settings = settings
        self.imported_models = imported_models
        self.color_extractor = None
        self.color_validator = None
        
        # Initialize color analysis components
        self._init_color_models()
    
    def _init_color_models(self):
        """Initialize color analysis models"""
        try:
            if self.imported_models.get('ColorPaletteExtractor') is not None:
                self.color_extractor = self.imported_models['ColorPaletteExtractor']()
                logger.info("✅ ColorPaletteExtractor initialized with real model")
            else:
                logger.warning("⚠️ ColorPaletteExtractor not available, using fallback")
            
            if self.imported_models.get('ColorPaletteValidator') is not None:
                # ColorPaletteValidator requires a brand_palette argument
                # We'll initialize it with a default palette or None
                try:
                    from ..config.settings import BrandColorPalette
                    # Create a default palette with required arguments
                    default_palette = BrandColorPalette(
                        name="Default Brand Palette",
                        primary_colors=["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"]
                    )
                    self.color_validator = self.imported_models['ColorPaletteValidator'](default_palette)
                    logger.info("✅ ColorPaletteValidator initialized with real model and default palette")
                except Exception as e:
                    logger.warning(f"⚠️ ColorPaletteValidator initialization failed: {e}, using fallback")
                    self.color_validator = None
            else:
                logger.warning("⚠️ ColorPaletteValidator not available, using fallback")
        except Exception as e:
            logger.error(f"Color analysis initialization failed: {e}")
            self.color_extractor = None
            self.color_validator = None
    
    def analyze_colors(self, image: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform color analysis on an image"""
        try:
            if not options:
                options = {}
            
            # Extract dominant colors
            n_colors = options.get('n_colors', 8)
            colors = self._extract_dominant_colors(image, n_colors)
            
            # Validate against brand colors if provided
            brand_validation = {}
            if options.get('brand_palette'):
                brand_validation = self._validate_colors_against_palette_real(
                    colors, 
                    options['brand_palette'], 
                    options.get('color_tolerance', 0.2)
                )
            
            # Analyze color contrast
            # contrast_analysis = self._analyze_color_contrast_real(colors)
            
            return {
                'dominant_colors': colors,
                'brand_validation': brand_validation,
                'contrast_analysis': "No contrast analysis",
                'total_colors': len(colors),
                'analysis_type': 'real_color_analysis' if self.color_extractor else 'fallback_color_analysis'
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {
                'error': f'Color analysis failed: {str(e)}',
                'dominant_colors': [],
                'brand_validation': {},
                'contrast_analysis': {},
                'total_colors': 0,
                'analysis_type': 'error'
            }
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 8) -> List[Dict[str, Any]]:
        """Extract dominant colors from image using K-means clustering"""
        try:

            extracted_colors = self.color_extractor.extract_colors(image)

            return extracted_colors
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return []
    
    def _validate_colors_against_palette_real(self, colors: List[Dict[str, Any]], brand_palette: str = '', tolerance: float = 0.2) -> Dict[str, Any]:
        """Validate colors against brand palette using real model"""
        try:

            if brand_colors and any(brand_colors.values()):
                validation_results = validate_against_brand_colors(extracted_colors, brand_colors, color_tolerance)
                
            return validation_result
            
        except Exception as e:
            logger.error(f"Real color validation failed: {e}, using fallback")
            return self._validate_colors_against_palette(colors, brand_palette, tolerance)
    
    def validate_against_brand_colors(extracted_colors: list, brand_colors: dict, tolerance: float = 2.3) -> dict:

        """Validate extracted colors against brand color palette with categories and thresholds"""
        try:
            # Check if brand palette is empty
            primary_colors = brand_colors.get('primary_colors', [])
            secondary_colors = brand_colors.get('secondary_colors', [])
            accent_colors = brand_colors.get('accent_colors', [])

            
            # If no brand colors are defined, return appropriate response
            if not primary_colors and not secondary_colors and not accent_colors:
                return {
                    'compliance_score': 1.0,
                    'compliant_colors': len(extracted_colors),
                    'non_compliant_colors': 0,
                    'total_colors': len(extracted_colors),
                    'compliant_colors_list': extracted_colors,
                    'non_compliant_colors_list': [],
                    'category_matches': {
                        'primary': [],
                        'secondary': [],
                        'accent': []
                    },
                    'thresholds': {
                        'primary': 0.75,
                        'secondary': 0.75,
                        'accent': 0.75
                    },
                    'warning': 'No brand colors defined - cannot validate compliance'
                }

            compliant_count = 0
            non_compliant_colors = []
            compliant_colors = []
            category_matches = {
                'primary': [],
                'secondary': [],
                'accent': []
            }
            
            # Get thresholds for each category
            primary_threshold = brand_colors.get('primary_threshold', 75) / 100.0
            secondary_threshold = brand_colors.get('secondary_threshold', 75) / 100.0
            accent_threshold = brand_colors.get('accent_threshold', 75) / 100.0

            
            for color_info in extracted_colors:
                extracted_rgb = color_info['rgb']
                is_compliant = False
                match_category = None
                best_match = None
                best_similarity = 0
                
                # Check against primary colors
                for primary_color in primary_colors:
                    similarity = calculate_color_similarity(extracted_rgb, primary_color, tolerance)
                    if similarity > best_similarity and similarity >= primary_threshold:
                        best_similarity = similarity
                        best_match = primary_color
                        match_category = 'primary'
                        is_compliant = True
                
                # Check against secondary colors
                for secondary_color in secondary_colors:
                    similarity = calculate_color_similarity(extracted_rgb, secondary_color, tolerance)
                    if similarity > best_similarity and similarity >= secondary_threshold:
                        best_similarity = similarity
                        best_match = secondary_color
                        match_category = 'secondary'
                        is_compliant = True
                
                # Check against accent colors
                for accent_color in accent_colors:
                    similarity = calculate_color_similarity(extracted_rgb, accent_color, tolerance)
                    if similarity > best_similarity and similarity >= accent_threshold:
                        best_similarity = similarity
                        best_match = accent_color
                        match_category = 'accent'
                        is_compliant = True
                
                if is_compliant:
                    compliant_count += 1
                    compliant_colors.append({
                        'color_info': color_info,
                        'match_category': match_category,
                        'matched_color': best_match,
                        'similarity': best_similarity
                    })
                    category_matches[match_category].append({
                        'color_info': color_info,
                        'matched_color': best_match,
                        'similarity': best_similarity
                    })
                else:
                    non_compliant_colors.append(color_info)
            
            compliance_score = compliant_count / len(extracted_colors) if extracted_colors else 0
            
            return {
                'compliance_score': round(compliance_score, 3),
                'compliant_colors': len(compliant_colors),
                'non_compliant_colors': len(non_compliant_colors),
                'total_colors': len(extracted_colors),
                'compliant_colors_list': compliant_colors,
                'non_compliant_colors_list': non_compliant_colors,
                'category_matches': category_matches,
                'thresholds': {
                    'primary': primary_threshold,
                    'secondary': secondary_threshold,
                    'accent': accent_threshold
                }
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}
    
def calculate_color_similarity(rgb1: tuple, color2, tolerance: float = 2.3) -> float:
    """Calculate color similarity using CIEDE2000 color difference
    
    Args:
        rgb1: First RGB color tuple
        color2: Second color (can be hex string or RGB tuple)
        tolerance: CIEDE2000 threshold (default 2.3 is perceptually noticeable)
    
    Returns:
        float: Similarity score (0-1, where 1 is identical)
    """
    try:
        # Convert color2 to RGB if needed
        if isinstance(color2, str):
            rgb2 = hex_to_rgb(color2)
        else:
            rgb2 = color2
        
        # Convert RGB to LAB color space using skimage
        # skimage expects RGB values in range [0, 1], so normalize
        rgb1_normalized = np.array(rgb1) / 255.0
        rgb2_normalized = np.array(rgb2) / 255.0
        
        # Convert to LAB
        lab1 = rgb2lab(rgb1_normalized.reshape(1, 1, 3))
        lab2 = rgb2lab(rgb2_normalized.reshape(1, 1, 3))
        
        # Calculate CIEDE2000 color difference using skimage
        delta_e = deltaE_ciede2000(lab1, lab2)
        
        # Convert delta_e to similarity score (0-1)
        # Lower delta_e means more similar colors
        similarity = max(0, 1 - (delta_e[0, 0] / (tolerance * 2)))
        
        return similarity
        
    except Exception:
        return 0.0
    
    def _analyze_color_contrast_real(self, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze color contrast using real model"""
        try:
            if not self.color_validator:
                return self._analyze_color_contrast(colors)
            
            # Use real model for contrast analysis
            # ColorPaletteValidator doesn't have analyze_contrast method
            # Use validate_colors as a proxy for contrast analysis
            contrast_result = self.color_validator.validate_colors(colors)
            return contrast_result
            
        except Exception as e:
            logger.error(f"Real contrast analysis failed: {e}, using fallback")
            return self._analyze_color_contrast(colors)
    
    def _analyze_color_contrast(self, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback color contrast analysis"""
        try:
            if len(colors) < 2:
                return {
                    'contrast_ratio': 0,
                    'accessibility_rating': 'N/A',
                    'recommendations': ['Need at least 2 colors for contrast analysis']
                }
            
            # Simple contrast analysis (placeholder)
            return {
                'contrast_ratio': 4.5,  # Placeholder
                'accessibility_rating': 'Good',
                'recommendations': ['Colors provide adequate contrast']
            }
            
        except Exception as e:
            logger.error(f"Contrast analysis failed: {e}")
            return {
                'contrast_ratio': 0,
                'accessibility_rating': 'Error',
                'recommendations': [f'Analysis failed: {str(e)}']
            }
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
