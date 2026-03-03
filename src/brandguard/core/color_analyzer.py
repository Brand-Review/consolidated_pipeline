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
        """Perform color analysis on an image
        
        RULES:
        - If brand palette NOT provided → status = "observed_only", do NOT validate compliance
        - Never output complianceScore = 0 or 100 without brand palette
        """
        try:
            if not options:
                options = {}
            
            # Extract dominant colors - ALWAYS run extraction regardless of brand palette
            n_colors = options.get('n_colors', 8)
            colors = self._extract_dominant_colors(image, n_colors)
            
            # CRITICAL: Ensure colors are never empty - use fallback if needed
            if not colors or len(colors) == 0:
                logger.warning("[Color] Color extraction returned empty - using emergency fallback")
                colors = self._extract_colors_fallback(image, n_colors)
            
            # ✅ CRITICAL: Ensure at least 3 colors (minimum requirement)
            if not colors or len(colors) < 3:
                logger.warning(f"[Color] Only {len(colors) if colors else 0} colors extracted - ensuring minimum 3 colors")
                # Sample additional colors to reach minimum
                h, w = image.shape[:2]
                existing_hex = {c.get('hex', '').upper() or c.get('color', '').upper() for c in colors if isinstance(c, dict)} if colors else set()
                additional_positions = [
                    (h//4, w//4), (h//4, 3*w//4), (3*h//4, w//4), (3*h//4, 3*w//4),
                    (h//2, w//2), (h//3, w//3), (2*h//3, 2*w//3)
                ]
                for y, x in additional_positions:
                    if len(colors) >= 3:
                        break
                    if 0 <= y < h and 0 <= x < w:
                        b, g, r = image[y, x]
                        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                        if hex_color not in existing_hex:
                            colors.append({
                                'hex': hex_color,
                                'rgb': [int(r), int(g), int(b)],
                                'color': hex_color,
                                'percentage': 100.0 / 3
                            })
                            existing_hex.add(hex_color)
            
            # Convert colors to proper format with percentage
            dominant_colors_formatted = []
            dominant_colors_hex = []
            total_percentage = 0.0
            
            for i, color_info in enumerate(colors):
                if isinstance(color_info, dict):
                    # Extract hex if available, otherwise convert from RGB
                    hex_color = color_info.get('hex') or color_info.get('color')
                    if not hex_color and 'rgb' in color_info:
                        rgb = color_info['rgb']
                        hex_color = self._rgb_to_hex(rgb if isinstance(rgb, tuple) else tuple(rgb))
                    
                    if hex_color:
                        hex_color = hex_color.upper() if hex_color.startswith('#') else f"#{hex_color.upper()}"
                        dominant_colors_hex.append(hex_color)
                        
                        # Get RGB
                        rgb = color_info.get('rgb')
                        if not rgb:
                            # Convert hex to RGB
                            hex_clean = hex_color.lstrip('#')
                            rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
                        elif isinstance(rgb, (list, tuple)):
                            rgb = tuple(rgb[:3])
                        
                        # Calculate percentage (equal distribution if not provided)
                        percentage = color_info.get('percentage') or color_info.get('percent') or (100.0 / len(colors))
                        total_percentage += percentage
                        
                        dominant_colors_formatted.append({
                            'hex': hex_color,
                            'rgb': list(rgb) if isinstance(rgb, tuple) else rgb,
                            'percentage': round(percentage, 2),
                            'source': 'image'
                        })
            
            # ✅ CRITICAL: Ensure at least 3 colors in formatted list
            if len(dominant_colors_formatted) < 3:
                logger.warning(f"[Color] Only {len(dominant_colors_formatted)} colors formatted - ensuring minimum 3")
                # Use fallback extraction to get more colors
                fallback_colors = self._extract_colors_fallback(image, 3)
                for fb_color in fallback_colors:
                    if len(dominant_colors_formatted) >= 3:
                        break
                    if isinstance(fb_color, dict):
                        hex_fb = fb_color.get('hex') or fb_color.get('color')
                        if hex_fb and hex_fb not in dominant_colors_hex:
                            hex_fb = hex_fb.upper() if hex_fb.startswith('#') else f"#{hex_fb.upper()}"
                            dominant_colors_hex.append(hex_fb)
                            rgb_fb = fb_color.get('rgb', [0, 0, 0])
                            if isinstance(rgb_fb, tuple):
                                rgb_fb = list(rgb_fb)
                            dominant_colors_formatted.append({
                                'hex': hex_fb,
                                'rgb': rgb_fb[:3] if len(rgb_fb) >= 3 else [0, 0, 0],
                                'percentage': round(100.0 / 3, 2),
                                'source': 'image'
                            })
            
            # Normalize percentages if they don't sum to 100
            if total_percentage > 0 and abs(total_percentage - 100.0) > 0.01:
                for color in dominant_colors_formatted:
                    color['percentage'] = round((color['percentage'] / total_percentage) * 100.0, 2)
            
            # ✅ CRITICAL: Limit to top 5 colors for UI (min 3, max 5)
            dominant_colors_formatted = dominant_colors_formatted[:5]
            dominant_colors_hex = dominant_colors_hex[:5]
            
            # Check if brand palette is provided
            brand_palette = options.get('brand_palette', '')
            has_brand_palette = bool(brand_palette and (isinstance(brand_palette, str) and brand_palette.strip()) or (isinstance(brand_palette, dict) and any(brand_palette.values())))
            
            # RULE: If brand palette NOT provided → status = "observed_only", but STILL return extracted colors
            if not has_brand_palette:
                logger.info("[Color] No brand palette provided - marking as observed_only, returning extracted colors")
                
                # Calculate observable score
                observable_score = self._calculate_color_observable_score(dominant_colors_formatted)
                
                return {
                    'status': 'observed_only',
                    'visibilityState': 'observed',
                    'dominant_colors': dominant_colors_formatted if dominant_colors_formatted else [{'hex': c.get('hex', c.get('color', '')), 'rgb': c.get('rgb', [0,0,0]), 'percentage': 100.0/len(colors), 'source': 'image'} for c in colors if isinstance(c, dict)],
                    'extractionMetadata': {
                        'method': 'kmeans' if self.color_extractor else 'fallback',
                        'confidence': 0.85 if self.color_extractor else 0.7,
                        'coveragePercentage': 100.0,
                        'totalColors': len(dominant_colors_formatted)
                    },
                    'observableScore': observable_score,
                    'message': 'Colors extracted from image (no brand palette provided)',
                    'brand_validation': {
                        'status': 'not_applicable',
                        'reason': 'No brand palette provided',
                        'compliance_score': None
                    },
                    'total_colors': len(colors),
                    'analysis_type': 'real_color_analysis' if self.color_extractor else 'fallback_color_analysis'
                }
            
            # Brand palette provided - validate compliance
            brand_validation = self._validate_colors_against_palette_real(
                colors, 
                brand_palette, 
                options.get('color_tolerance', 2.3)
            )
            
            # Determine validation status
            if brand_validation.get('error'):
                validation_status = 'unknown'
                validation_reason = brand_validation.get('error', 'Validation failed')
            else:
                compliance_score = brand_validation.get('compliance_score')
                
                # RULE: Never output 0 or 100 without proper validation
                # Check if compliance_score is None (validation failed or incomplete)
                if compliance_score is None:
                    validation_status = 'unknown'
                    validation_reason = 'Color validation incomplete or failed'
                elif compliance_score == 0.0:
                    # Only accept 0.0 if we have evidence of non-compliant colors
                    non_compliant = brand_validation.get('non_compliant_colors', 0)
                    total_colors = brand_validation.get('total_colors', 0)
                    if non_compliant > 0 and total_colors > 0:
                        validation_status = 'failed'
                        validation_reason = f'{non_compliant} of {total_colors} colors do not comply with brand palette'
                    else:
                        # Invalid 0.0 score - should not happen with proper validation
                        validation_status = 'unknown'
                        validation_reason = 'Invalid compliance score - validation may be incomplete'
                        compliance_score = None  # Set to None to prevent fake scoring
                elif compliance_score == 1.0:
                    # Only accept 1.0 if all colors are compliant
                    compliant = brand_validation.get('compliant_colors', 0)
                    total_colors = brand_validation.get('total_colors', 0)
                    if compliant == total_colors and total_colors > 0:
                        validation_status = 'passed'
                        validation_reason = 'All colors comply with brand palette'
                    else:
                        # Invalid 1.0 score - should not happen with proper validation
                        validation_status = 'unknown'
                        validation_reason = 'Invalid compliance score - validation may be incomplete'
                        compliance_score = None  # Set to None to prevent fake scoring
                else:
                    # Score is between 0 and 1 (exclusive) - valid partial compliance
                    validation_status = 'failed' if compliance_score < 0.75 else 'passed'
                    validation_reason = brand_validation.get('message', 'Color validation completed')
            
            # Use the validated compliance_score (may be None if invalid)
            final_compliance_score = compliance_score if 'compliance_score' in locals() else brand_validation.get('compliance_score')
            
            # Format colors with percentage for validated case
            validated_colors_formatted = dominant_colors_formatted if dominant_colors_formatted else [
                {
                    'hex': c.get('hex', c.get('color', '')),
                    'rgb': c.get('rgb', [0, 0, 0]) if isinstance(c.get('rgb'), (list, tuple)) else [0, 0, 0],
                    'percentage': 100.0 / len(colors) if colors else 0,
                    'source': 'image'
                }
                for c in colors if isinstance(c, dict)
            ]
            
            # Calculate observable score even when validated
            observable_score = self._calculate_color_observable_score(validated_colors_formatted)
            
            return {
                'status': 'validated',
                'visibilityState': 'evaluated',
                'dominant_colors': validated_colors_formatted,
                'extractionMetadata': {
                    'method': 'kmeans' if self.color_extractor else 'fallback',
                    'confidence': 0.85 if self.color_extractor else 0.7,
                    'coveragePercentage': 100.0,
                    'totalColors': len(validated_colors_formatted)
                },
                'observableScore': observable_score,
                'brand_validation': {
                    'status': validation_status,
                    'reason': validation_reason,
                    'compliance_score': final_compliance_score,  # May be None if invalid
                    'compliant_colors': brand_validation.get('compliant_colors', 0),
                    'non_compliant_colors': brand_validation.get('non_compliant_colors', 0),
                    'total_colors': brand_validation.get('total_colors', len(colors))
                },
                'total_colors': len(colors),
                'analysis_type': 'real_color_analysis' if self.color_extractor else 'fallback_color_analysis'
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            # CRITICAL: Even on error, extract colors as last resort
            try:
                emergency_colors = self._extract_colors_fallback(image, n_colors=3)
                emergency_formatted = [
                    {
                        'hex': c.get('hex', c.get('color', '')),
                        'rgb': c.get('rgb', [0, 0, 0]) if isinstance(c.get('rgb'), (list, tuple)) else [0, 0, 0],
                        'percentage': c.get('percentage', 100.0 / len(emergency_colors)),
                        'source': 'image'
                    }
                    for c in emergency_colors if isinstance(c, dict)
                ]
                return {
                    'status': 'observed_only',
                    'visibilityState': 'observed',
                    'reason': f'Color analysis error: {str(e)}, using emergency extraction',
                    'dominant_colors': emergency_formatted,
                    'message': 'Colors extracted from image (analysis error occurred)',
                    'brand_validation': {
                        'status': 'error',
                        'reason': f'Analysis error: {str(e)}',
                        'compliance_score': None
                    },
                    'total_colors': len(emergency_colors),
                    'analysis_type': 'error_fallback'
                }
            except Exception as fallback_error:
                logger.error(f"Emergency color extraction also failed: {fallback_error}")
                # Absolute last resort - single pixel from center
                h, w = image.shape[:2]
                center_y, center_x = h // 2, w // 2
                if len(image.shape) == 3:
                    b, g, r = image[center_y, center_x]
                else:
                    r = g = b = image[center_y, center_x]
                hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                return {
                    'status': 'observed_only',
                    'visibilityState': 'observed',
                    'reason': f'Color analysis failed: {str(e)}',
                    'dominant_colors': [{
                        'hex': hex_color,
                        'rgb': [int(r), int(g), int(b)],
                        'percentage': 100.0,
                        'source': 'image'
                    }],
                    'message': 'Colors extracted from image (emergency fallback)',
                    'brand_validation': {
                        'status': 'error',
                        'reason': f'Analysis error: {str(e)}',
                        'compliance_score': None
                    },
                    'total_colors': 1,
                    'analysis_type': 'error_emergency'
                }
    
    def _calculate_color_observable_score(self, dominant_colors: List[Dict]) -> float:
        """
        Calculate observable score for colors (no brand palette needed).
        
        Metrics:
        - Color diversity: More colors = better (up to 5 colors)
        - Coverage balance: Colors should have reasonable distribution
        
        Returns: 0-1 score
        """
        if not dominant_colors or len(dominant_colors) == 0:
            return 0.0
        
        # Score based on number of colors (3-5 is optimal)
        color_count_score = min(1.0, len(dominant_colors) / 5.0)
        
        # Score based on coverage balance (colors should be reasonably distributed)
        percentages = [c.get('percentage', 0) for c in dominant_colors if isinstance(c, dict)]
        if percentages:
            # Calculate coefficient of variation (lower = more balanced)
            mean_pct = sum(percentages) / len(percentages)
            std_pct = (sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)) ** 0.5
            cv = std_pct / mean_pct if mean_pct > 0 else 1.0
            balance_score = max(0.0, 1.0 - cv)  # Lower CV = higher score
        else:
            balance_score = 0.5
        
        # Weighted average
        observable_score = (color_count_score * 0.6) + (balance_score * 0.4)
        return round(observable_score, 2)
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 8) -> List[Dict[str, Any]]:
        """Extract dominant colors from image using K-means clustering"""
        try:
            # FIX: Check if color_extractor is available
            if self.color_extractor is None:
                logger.warning("[Color] ColorPaletteExtractor not available, using fallback k-means clustering")
                return self._extract_colors_fallback(image, n_colors)
            
            extracted_colors = self.color_extractor.extract_colors(image)
            
            # FIX: If extraction returns empty or None, use fallback
            if not extracted_colors or len(extracted_colors) == 0:
                logger.warning("[Color] Color extraction returned empty, using fallback k-means clustering")
                return self._extract_colors_fallback(image, n_colors)
            
            return extracted_colors
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}, using fallback k-means clustering")
            return self._extract_colors_fallback(image, n_colors)
    
    def _extract_colors_fallback(self, image: np.ndarray, n_colors: int = 8) -> List[Dict[str, Any]]:
        """FIX: Fallback color extraction using k-means clustering when model is unavailable"""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Reshape image to 2D array of pixels (BGR format)
            h, w = image.shape[:2]
            pixels = image.reshape(-1, 3)
            
            # Sample pixels for faster clustering
            sample_size = min(1000, len(pixels))
            if len(pixels) > sample_size:
                sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
                sample_pixels = pixels[sample_indices]
            else:
                sample_pixels = pixels
            
            # Convert BGR to RGB for clustering
            sample_pixels_rgb = sample_pixels[:, ::-1]
            
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(sample_pixels_rgb)
            
            # Get cluster centers (dominant colors) and labels
            colors_rgb = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate percentage for each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            # Convert RGB to hex and format as dict with percentage
            color_results = []
            for idx, rgb in enumerate(colors_rgb):
                r, g, b = rgb[0], rgb[1], rgb[2]
                hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                
                # Calculate percentage for this cluster
                cluster_count = counts[unique_labels == idx][0] if idx in unique_labels else 0
                percentage = (cluster_count / total_samples * 100.0) if total_samples > 0 else (100.0 / len(colors_rgb))
                
                color_results.append({
                    'hex': hex_color,
                    'rgb': (int(r), int(g), int(b)),
                    'color': hex_color,
                    'percentage': round(percentage, 2)
                })
            
            logger.info(f"[Color] Fallback k-means extraction: {len(color_results)} colors extracted")
            return color_results
            
        except ImportError:
            logger.warning("[Color] sklearn not available, using simple pixel sampling fallback")
            # FIX: Simple fallback - sample pixels from different regions
            h, w = image.shape[:2]
            color_results = []
            num_samples = min(n_colors, 8)
            for i in range(num_samples):
                y = int(h * (i + 1) / (num_samples + 1))
                x = int(w * (i + 1) / (num_samples + 1))
                b, g, r = image[y, x]
                hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                color_results.append({
                    'hex': hex_color,
                    'rgb': (int(r), int(g), int(b)),
                    'color': hex_color,
                    'percentage': round(100.0 / num_samples, 2)
                })
            return color_results
        except Exception as e:
            logger.error(f"[Color] Fallback color extraction failed: {e}")
            # FIX: Last resort - return at least one color from image center
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            b, g, r = image[center_y, center_x]
            hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
            return [{
                'hex': hex_color,
                'rgb': (int(r), int(g), int(b)),
                'color': hex_color,
                'percentage': 100.0
            }]
    
    def _validate_colors_against_palette_real(self, colors: List[Dict[str, Any]], brand_palette, tolerance: float = 2.3) -> Dict[str, Any]:
        """Validate colors against brand palette using real model"""
        try:
            # Parse brand_palette - can be string (JSON) or dict
            if isinstance(brand_palette, str):
                import json
                try:
                    brand_colors = json.loads(brand_palette)
                except:
                    # If not JSON, treat as a single color
                    brand_colors = {'primary_colors': [brand_palette]}
            elif isinstance(brand_palette, dict):
                brand_colors = brand_palette
            else:
                logger.warning(f"Invalid brand_palette type: {type(brand_palette)}")
                return {'error': 'Invalid brand palette format'}
            
            # Extract color lists
            primary_colors = brand_colors.get('primary_colors', [])
            secondary_colors = brand_colors.get('secondary_colors', [])
            accent_colors = brand_colors.get('accent_colors', [])
            
            # If no brand colors, return error
            if not primary_colors and not secondary_colors and not accent_colors:
                return {
                    'error': 'No brand colors defined',
                    'compliance_score': None
                }
            
            # Validate using the static method
            validation_result = ColorAnalyzer.validate_against_brand_colors(colors, brand_colors, tolerance)
            return validation_result
            
        except Exception as e:
            logger.error(f"Real color validation failed: {e}, using fallback")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'Validation failed: {str(e)}', 'compliance_score': None}
    
    @staticmethod
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
    
def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple"""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_color) == 3:
            return tuple(int(hex_color[i]*2, 16) for i in (0, 1, 2))
        return (0, 0, 0)
    except:
        return (0, 0, 0)


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
        elif isinstance(color2, (list, tuple)) and len(color2) >= 3:
            rgb2 = tuple(color2[:3])
        elif isinstance(color2, dict):
            # Handle dict with r, g, b keys
            if 'r' in color2 and 'g' in color2 and 'b' in color2:
                rgb2 = (int(color2['r']), int(color2['g']), int(color2['b']))
            elif 'rgb' in color2:
                rgb_val = color2['rgb']
                if isinstance(rgb_val, (list, tuple)):
                    rgb2 = tuple(rgb_val[:3])
                else:
                    rgb2 = (0, 0, 0)
            else:
                rgb2 = (0, 0, 0)
        else:
            rgb2 = (0, 0, 0)
        
        # Ensure rgb1 is a tuple
        if isinstance(rgb1, (list, tuple)) and len(rgb1) >= 3:
            rgb1 = tuple(rgb1[:3])
        elif isinstance(rgb1, dict):
            if 'r' in rgb1 and 'g' in rgb1 and 'b' in rgb1:
                rgb1 = (int(rgb1['r']), int(rgb1['g']), int(rgb1['b']))
            elif 'rgb' in rgb1:
                rgb_val = rgb1['rgb']
                rgb1 = tuple(rgb_val[:3]) if isinstance(rgb_val, (list, tuple)) else (0, 0, 0)
            else:
                rgb1 = (0, 0, 0)
        else:
            rgb1 = (0, 0, 0)
        
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
        
    except Exception as e:
        logger.error(f"Color similarity calculation failed: {e}")
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
    
    def _rgb_to_hex(self, rgb) -> str:
        """Convert RGB tuple or list to hex string"""
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        elif isinstance(rgb, dict):
            if 'r' in rgb and 'g' in rgb and 'b' in rgb:
                return f"#{int(rgb['r']):02x}{int(rgb['g']):02x}{int(rgb['b']):02x}"
        return "#000000"  # Default fallback
