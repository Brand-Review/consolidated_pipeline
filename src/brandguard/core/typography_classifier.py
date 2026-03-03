"""
Typography Classifier
Classifies typography characteristics from raster images without guessing font names.

Classification categories:
- serif / sans-serif
- geometric / humanist / grotesk
- weight (light, regular, bold)
- case (upper, lower, title)
- role (headline, body, ui-label)
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TypographyClassifier:
    """Classifies typography characteristics from raster images"""
    
    # Similar fonts database (for suggestions)
    SIMILAR_FONTS_DB = {
        'sans-serif': {
            'geometric': ['Inter', 'Roboto', 'Montserrat', 'Poppins', 'Nunito'],
            'humanist': ['Open Sans', 'Lato', 'Source Sans Pro', 'PT Sans'],
            'grotesk': ['Helvetica', 'Arial', 'Univers', 'Akzidenz Grotesk']
        },
        'serif': {
            'geometric': ['Georgia', 'Merriweather', 'Crimson Text'],
            'humanist': ['Lora', 'Crimson Pro', 'Libre Baskerville'],
            'old-style': ['Times New Roman', 'Garamond', 'Baskerville']
        }
    }
    
    def __init__(self):
        """Initialize typography classifier"""
        pass
    
    def classify_text_region(
        self,
        image: np.ndarray,
        text_region: Dict[str, Any],
        image_height: Optional[int] = None,
        image_width: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Classify typography characteristics for a single text region.
        
        Args:
            image: Full image as numpy array
            text_region: Dict with 'text', 'bbox', 'approximateSizePx'
            image_height: Image height (for normalized bbox conversion)
            image_width: Image width (for normalized bbox conversion)
            
        Returns:
            Classification dict with:
            - fontClassification: Dict with serif, style, weight, case, role
            - similarFonts: List of similar font suggestions (optional)
            - confidence: float (0-1)
        """
        try:
            text = text_region.get('text', '')
            bbox = text_region.get('bbox', [])
            approximate_size_px = text_region.get('approximateSizePx', 12)
            
            if not text or len(bbox) < 4:
                return {
                    'fontClassification': {
                        'serif': None,
                        'style': None,
                        'weight': None,
                        'case': None,
                        'role': None
                    },
                    'similarFonts': [],
                    'confidence': 0.0,
                    'limitations': ['Insufficient text or bounding box data']
                }
            
            # Extract text region from image for analysis
            text_image = self._extract_text_region_image(image, bbox, image_height, image_width)
            
            if text_image is None or text_image.size == 0:
                # Fallback: classify from text content and size only
                return self._classify_from_text_only(text, approximate_size_px, bbox)
            
            # Classify serif/sans-serif
            serif_classification = self._classify_serif(text_image)
            
            # Classify style (geometric/humanist/grotesk)
            style_classification = self._classify_style(text_image, serif_classification['serif'])
            
            # Classify weight
            weight_classification = self._classify_weight(text_image)
            
            # Classify case (from text content)
            case_classification = self._classify_case(text)
            
            # Classify role (from size and position)
            role_classification = self._classify_role(approximate_size_px, bbox, image_height or image.shape[0])
            
            # Build classification
            font_classification = {
                'serif': serif_classification['serif'],
                'style': style_classification['style'],
                'weight': weight_classification['weight'],
                'case': case_classification['case'],
                'role': role_classification['role']
            }
            
            # Calculate overall confidence
            confidence = min(
                serif_classification['confidence'],
                style_classification['confidence'],
                weight_classification['confidence'],
                case_classification['confidence'],
                role_classification['confidence']
            )
            
            # Generate similar font suggestions (optional)
            similar_fonts = self._suggest_similar_fonts(font_classification, confidence)
            
            # Build limitations list
            limitations = [
                'Font family names cannot be reliably detected from raster images',
                'Classification is approximate and based on visual characteristics',
                'Serif detection accuracy depends on image quality and resolution',
                'Style classification (geometric/humanist/grotesk) is approximate'
            ]
            
            return {
                'fontClassification': font_classification,
                'similarFonts': similar_fonts,
                'confidence': round(confidence, 2),
                'limitations': limitations
            }
            
        except Exception as e:
            logger.error(f"Typography classification failed: {e}")
            return {
                'fontClassification': {
                    'serif': None,
                    'style': None,
                    'weight': None,
                    'case': None,
                    'role': None
                },
                'similarFonts': [],
                'confidence': 0.0,
                'limitations': [f'Classification error: {str(e)}']
            }
    
    def _extract_text_region_image(
        self,
        image: np.ndarray,
        bbox: List[float],
        image_height: Optional[int] = None,
        image_width: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Extract text region from image for analysis"""
        try:
            h, w = image.shape[:2]
            
            # Convert normalized bbox to pixel coordinates if needed
            if bbox[0] <= 1.0 and bbox[1] <= 1.0:
                # Normalized coordinates
                if image_height and image_width:
                    x1 = int(bbox[0] * image_width)
                    y1 = int(bbox[1] * image_height)
                    x2 = int(bbox[2] * image_width)
                    y2 = int(bbox[3] * image_height)
                else:
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
            else:
                # Pixel coordinates
                x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Ensure within bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Convert to grayscale if needed
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            return region
            
        except Exception as e:
            logger.warning(f"Failed to extract text region: {e}")
            return None
    
    def _classify_serif(self, text_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify serif vs sans-serif using edge detection and contour analysis.
        
        Serif fonts have small decorative strokes at character endings.
        We detect these by analyzing edge patterns and contour features.
        """
        try:
            # Enhance image for analysis
            if text_image.size == 0:
                return {'serif': None, 'confidence': 0.0}
            
            # Resize if too small (improves detection)
            if text_image.shape[0] < 20 or text_image.shape[1] < 20:
                scale = max(20 / text_image.shape[0], 20 / text_image.shape[1])
                new_h = int(text_image.shape[0] * scale)
                new_w = int(text_image.shape[1] * scale)
                text_image = cv2.resize(text_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply thresholding
            _, binary = cv2.threshold(text_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detect edges
            edges = cv2.Canny(text_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'serif': None, 'confidence': 0.0}
            
            # Analyze contour features for serif detection
            serif_indicators = 0
            total_contours = min(len(contours), 50)  # Limit analysis
            
            for contour in contours[:total_contours]:
                # Serif fonts have more complex contours (more points, more variation)
                if len(contour) > 10:
                    # Calculate contour complexity
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    
                    if perimeter > 0:
                        # Circularity (serif fonts have lower circularity due to decorative strokes)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Serif fonts typically have more irregular shapes
                        if circularity < 0.7:
                            serif_indicators += 1
                
                # Check for small protrusions (serif characteristics)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                
                if hull_area > 0:
                    # Serif fonts have more area difference (protrusions)
                    area_ratio = contour_area / hull_area
                    if area_ratio < 0.85:  # Significant protrusions
                        serif_indicators += 1
            
            # Determine classification
            serif_ratio = serif_indicators / (total_contours * 2) if total_contours > 0 else 0.0
            
            if serif_ratio > 0.3:
                serif = 'serif'
                confidence = min(0.8, 0.5 + serif_ratio)
            else:
                serif = 'sans-serif'
                confidence = min(0.8, 0.5 + (1 - serif_ratio))
            
            return {'serif': serif, 'confidence': round(confidence, 2)}
            
        except Exception as e:
            logger.warning(f"Serif classification failed: {e}")
            return {'serif': None, 'confidence': 0.0}
    
    def _classify_style(
        self,
        text_image: np.ndarray,
        serif_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Classify style: geometric, humanist, or grotesk (for sans-serif)
        or geometric, humanist, old-style (for serif).
        
        This is approximate and based on overall character shape.
        """
        try:
            if text_image.size == 0:
                return {'style': None, 'confidence': 0.0}
            
            # For now, use a simplified classification
            # Geometric: more uniform, circular shapes
            # Humanist: more organic, varied shapes
            # Grotesk: more uniform, less organic than humanist
            
            # This is a simplified heuristic - in production, you might use
            # more sophisticated shape analysis or ML models
            
            # Default to 'geometric' for sans-serif, 'humanist' for serif
            # with low confidence (since this is hard to determine accurately)
            
            if serif_type == 'serif':
                style = 'humanist'  # Most serif fonts are humanist
                confidence = 0.4  # Low confidence
            else:
                style = 'geometric'  # Default for sans-serif
                confidence = 0.4  # Low confidence
            
            return {'style': style, 'confidence': confidence}
            
        except Exception as e:
            logger.warning(f"Style classification failed: {e}")
            return {'style': None, 'confidence': 0.0}
    
    def _classify_weight(self, text_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify font weight: light, regular, bold.
        
        Based on stroke width and pixel density.
        """
        try:
            if text_image.size == 0:
                return {'weight': None, 'confidence': 0.0}
            
            # Apply thresholding
            _, binary = cv2.threshold(text_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate stroke width (average width of character strokes)
            # Use distance transform to find stroke width
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find local maxima (center of strokes)
            # Simplified: use mean of distance transform as proxy for stroke width
            mean_stroke_width = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
            
            # Calculate pixel density (percentage of dark pixels)
            pixel_density = np.sum(binary == 0) / binary.size if binary.size > 0 else 0
            
            # Classify based on stroke width and density
            if mean_stroke_width > 2.5 or pixel_density > 0.4:
                weight = 'bold'
                confidence = 0.7
            elif mean_stroke_width < 1.5 or pixel_density < 0.2:
                weight = 'light'
                confidence = 0.6
            else:
                weight = 'regular'
                confidence = 0.7
            
            return {'weight': weight, 'confidence': round(confidence, 2)}
            
        except Exception as e:
            logger.warning(f"Weight classification failed: {e}")
            return {'weight': None, 'confidence': 0.0}
    
    def _classify_case(self, text: str) -> Dict[str, Any]:
        """
        Classify text case: upper, lower, title.
        
        Based on text content analysis.
        """
        try:
            if not text:
                return {'case': None, 'confidence': 0.0}
            
            # Remove non-alphabetic characters for analysis
            alpha_text = ''.join(c for c in text if c.isalpha())
            
            if not alpha_text:
                return {'case': None, 'confidence': 0.0}
            
            upper_count = sum(1 for c in alpha_text if c.isupper())
            lower_count = sum(1 for c in alpha_text if c.islower())
            total_count = len(alpha_text)
            
            upper_ratio = upper_count / total_count if total_count > 0 else 0
            
            if upper_ratio > 0.8:
                case = 'upper'
                confidence = 0.9
            elif upper_ratio < 0.2:
                case = 'lower'
                confidence = 0.9
            else:
                # Title case: first letter uppercase, rest lowercase (approximately)
                # Or mixed case
                case = 'title'
                confidence = 0.7
            
            return {'case': case, 'confidence': round(confidence, 2)}
            
        except Exception as e:
            logger.warning(f"Case classification failed: {e}")
            return {'case': None, 'confidence': 0.0}
    
    def _classify_role(
        self,
        approximate_size_px: float,
        bbox: List[float],
        image_height: int
    ) -> Dict[str, Any]:
        """
        Classify text role: headline, body, ui-label.
        
        Based on size and position.
        """
        try:
            # Determine position (normalized Y)
            if len(bbox) >= 4:
                if bbox[1] <= 1.0:
                    # Normalized
                    y_position = bbox[1]
                else:
                    # Pixel coordinates
                    y_position = bbox[1] / image_height if image_height > 0 else 0.5
            else:
                y_position = 0.5
            
            # Classify based on size and position
            if approximate_size_px >= 24:
                if y_position < 0.3:
                    role = 'headline'
                    confidence = 0.8
                else:
                    role = 'headline'  # Large text is likely headline
                    confidence = 0.7
            elif approximate_size_px >= 14:
                role = 'body'
                confidence = 0.8
            else:
                role = 'ui-label'
                confidence = 0.7
            
            return {'role': role, 'confidence': round(confidence, 2)}
            
        except Exception as e:
            logger.warning(f"Role classification failed: {e}")
            return {'role': None, 'confidence': 0.0}
    
    def _classify_from_text_only(
        self,
        text: str,
        approximate_size_px: float,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """Fallback classification using only text content and size"""
        case_classification = self._classify_case(text)
        role_classification = self._classify_role(approximate_size_px, bbox, 1000)  # Default height
        
        return {
            'fontClassification': {
                'serif': None,  # Cannot determine without image
                'style': None,
                'weight': None,
                'case': case_classification['case'],
                'role': role_classification['role']
            },
            'similarFonts': [],
            'confidence': min(case_classification['confidence'], role_classification['confidence']),
            'limitations': [
                'Text region image not available - classification limited to case and role',
                'Serif, style, and weight cannot be determined without image analysis'
            ]
        }
    
    def _suggest_similar_fonts(
        self,
        font_classification: Dict[str, Any],
        confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Suggest similar fonts based on classification.
        
        Only returns suggestions if confidence is reasonable (> 0.5).
        """
        try:
            if confidence < 0.5:
                return []  # Don't suggest if confidence is too low
            
            serif = font_classification.get('serif')
            style = font_classification.get('style')
            
            if not serif or not style:
                return []  # Need both serif and style for suggestions
            
            # Get similar fonts from database
            similar_fonts = []
            
            if serif == 'sans-serif' and style in self.SIMILAR_FONTS_DB['sans-serif']:
                font_list = self.SIMILAR_FONTS_DB['sans-serif'][style]
                for font_name in font_list[:3]:  # Top 3 suggestions
                    similar_fonts.append({
                        'name': font_name,
                        'confidence': round(confidence * 0.7, 2),  # Lower confidence for suggestions
                        'reason': f'Similar {style} {serif} font'
                    })
            elif serif == 'serif' and style in self.SIMILAR_FONTS_DB['serif']:
                font_list = self.SIMILAR_FONTS_DB['serif'][style]
                for font_name in font_list[:3]:  # Top 3 suggestions
                    similar_fonts.append({
                        'name': font_name,
                        'confidence': round(confidence * 0.7, 2),
                        'reason': f'Similar {style} {serif} font'
                    })
            
            return similar_fonts
            
        except Exception as e:
            logger.warning(f"Similar fonts suggestion failed: {e}")
            return []

