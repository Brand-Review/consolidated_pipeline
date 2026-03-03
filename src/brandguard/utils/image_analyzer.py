"""
Image Analysis Utilities
Helper functions for detecting visible text, logos, and other image features.
Used to distinguish between "no feature" vs "detection failure".
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def detect_visible_text_regions(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect if visible text exists in an image using basic image processing.
    
    This is used to distinguish between:
    - OCR failure when text exists (critical system error)
    - No text in image (legitimate skip)
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with:
        - has_text: bool - Whether text regions were detected
        - text_regions: List[Dict] - Detected text regions (approximate)
        - confidence: float - Confidence in text detection
    """
    try:
        if image is None or image.size == 0:
            return {
                'has_text': False,
                'text_regions': [],
                'confidence': 0.0,
                'method': 'invalid_image'
            }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Method 1: Detect text-like regions using morphological operations
        # This detects regions with high edge density and regular structure (characteristic of text)
        
        # Apply adaptive thresholding to improve text detection
        # This helps with images that have varying brightness
        try:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        except:
            binary = gray
        
        # Apply edge detection with multiple thresholds for better coverage
        edges_low = cv2.Canny(gray, 30, 100)
        edges_high = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges_low, edges_high)
        
        # Combine with binary threshold for better text detection
        edges_combined = cv2.bitwise_or(edges, binary)
        
        # Dilate edges to connect nearby pixels (characters in words)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges_combined, kernel, iterations=2)
        
        # Find contours (potential text regions)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        image_area = gray.shape[0] * gray.shape[1]
        min_contour_area = max(50, image_area * 0.0005)  # 0.05% of image area, minimum 50 pixels
        min_aspect_ratio = 1.2  # Slightly lower threshold for better detection (was 1.5)
        max_aspect_ratio = 20.0  # Cap very wide regions (might be lines/separators)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Text regions typically have:
            # - Aspect ratio between 1.2 and 20 (wider than tall, but not too extreme)
            # - Height within reasonable range (10px to 50% of image height)
            # - Width at least 20px (single characters are usually wider)
            min_height = max(8, gray.shape[0] * 0.01)  # At least 1% of image height or 8px
            max_height = gray.shape[0] * 0.6  # Up to 60% of image height
            
            if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and 
                min_height <= h <= max_height and
                w >= 15):  # Minimum width for text (characters are usually wider)
                text_regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'area_ratio': area / image_area if image_area > 0 else 0
                })
        
        has_text = len(text_regions) > 0
        
        # Calculate confidence based on number, quality, and size of detected regions
        # More regions = higher confidence, but also consider area coverage
        if text_regions:
            total_area_ratio = sum(r['area_ratio'] for r in text_regions)
            region_count_factor = min(1.0, len(text_regions) * 0.15)  # Each region adds up to 15% confidence
            area_factor = min(0.5, total_area_ratio * 10)  # Area coverage adds up to 50% confidence
            confidence = min(0.95, 0.3 + region_count_factor + area_factor)  # Base 30% + factors, cap at 95%
        else:
            confidence = 0.0
        
        return {
            'has_text': has_text,
            'text_regions': text_regions,
            'confidence': confidence,
            'method': 'morphological_analysis'
        }
        
    except Exception as e:
        logger.error(f"Visible text detection failed: {e}")
        return {
            'has_text': False,
            'text_regions': [],
            'confidence': 0.0,
            'method': 'error',
            'error': str(e)
        }


def detect_detection_failure(
    feature_type: str,
    detection_result: Any,
    has_feature_indicators: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine if a detection failure occurred vs no feature exists.
    
    Args:
        feature_type: Type of feature ('text', 'logo', etc.)
        detection_result: Result from detection (empty if nothing found)
        has_feature_indicators: Indicators that feature might exist
        
    Returns:
        Dictionary with:
        - is_failure: bool - True if detection failed (should exist but wasn't detected)
        - is_absent: bool - True if feature legitimately doesn't exist
        - confidence: float - Confidence in determination
    """
    try:
        if feature_type == 'text':
            has_text = has_feature_indicators.get('has_text', False)
            text_confidence = has_feature_indicators.get('confidence', 0.0)
            ocr_succeeded = bool(detection_result and len(str(detection_result).strip()) > 0)
            
            # RULE: Only flag as critical failure if confidence is reasonably high (>=0.3)
            # This prevents false positives from overly sensitive text detection
            # If indicators suggest text exists (with reasonable confidence) but OCR failed → detection failure
            if has_text and not ocr_succeeded and text_confidence >= 0.3:
                return {
                    'is_failure': True,
                    'is_absent': False,
                    'confidence': text_confidence,
                    'reason': f'Text regions detected (confidence: {text_confidence:.2f}) but OCR failed to extract'
                }
            # If confidence is low, treat as uncertain (not a critical failure)
            elif has_text and not ocr_succeeded and text_confidence < 0.3:
                return {
                    'is_failure': False,  # Not confident enough to flag as system error
                    'is_absent': True,
                    'confidence': 1.0 - text_confidence,
                    'reason': f'Low-confidence text indicators ({text_confidence:.2f}) and OCR returned empty - likely no text'
                }
            # If no indicators and no result → likely no text
            elif not has_text and not ocr_succeeded:
                return {
                    'is_failure': False,
                    'is_absent': True,
                    'confidence': 0.7,
                    'reason': 'No text regions detected and OCR returned empty'
                }
            # If OCR succeeded → no failure
            else:
                return {
                    'is_failure': False,
                    'is_absent': False,
                    'confidence': 1.0,
                    'reason': 'OCR succeeded or no text indicators'
                }
        
        elif feature_type == 'logo':
            has_detections = bool(detection_result and len(detection_result) > 0)
            
            # For logos, we can't easily detect if one "should" exist
            # So we distinguish based on detection method status
            if isinstance(detection_result, dict):
                detection_status = detection_result.get('status', 'unknown')
                if detection_status == 'failed':
                    return {
                        'is_failure': True,
                        'is_absent': False,
                        'confidence': 0.8,
                        'reason': 'Logo detection failed with error status'
                    }
                elif detection_status == 'skipped':
                    return {
                        'is_failure': False,
                        'is_absent': True,
                        'confidence': 0.6,
                        'reason': 'Logo detection was skipped'
                    }
                elif not has_detections:
                    return {
                        'is_failure': False,
                        'is_absent': True,
                        'confidence': 0.7,
                        'reason': 'No logos detected (detection method succeeded but found nothing)'
                    }
            
            # Default: no failure if detections exist or result is empty but valid
            return {
                'is_failure': False,
                'is_absent': not has_detections,
                'confidence': 0.6,
                'reason': 'Detection completed normally'
            }
        
        # Default for other feature types
        return {
            'is_failure': False,
            'is_absent': True,
            'confidence': 0.5,
            'reason': 'Unknown feature type'
        }
        
    except Exception as e:
        logger.error(f"Detection failure analysis failed: {e}")
        return {
            'is_failure': False,
            'is_absent': True,
            'confidence': 0.0,
            'error': str(e)
        }

