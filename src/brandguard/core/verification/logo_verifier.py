"""
Logo Verification Module
Stage B: Verifies that detected objects are actually logos, not UI blocks or other graphics.

Verification requires at least 2 of:
1. High edge density + symmetry
2. Brand text match from OCR
3. Aspect ratio within expected logo bounds
4. Size ratio between minLogoSize and maxLogoSize
5. Visual similarity (embedding or color consistency)

Stage C: Brand Mismatch Detection
- Compares detected logo against user's reference logo
- Uses VLM to identify brand in detected region
- Flags as "wrong brand" if detected brand != expected brand
"""

import cv2
import numpy as np
import logging
import base64
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# Global VLM config (set by logo_analyzer)
VLM_API_URL = "http://localhost:8000/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def set_vlm_config(api_url: str, model: str):
    """Configure VLM settings for brand mismatch detection"""
    global VLM_API_URL, VLM_MODEL
    VLM_API_URL = api_url
    VLM_MODEL = model


class LogoVerifier:
    """
    Verifies that detected objects are actually logos.
    Prevents false positives from UI blocks, badges, CTAs, etc.
    """
    
    def __init__(self, brand_name: Optional[str] = None):
        """
        Initialize logo verifier.
        
        Args:
            brand_name: Optional brand name for text matching (e.g., "AgencyHandy")
        """
        self.brand_name = brand_name
        if brand_name:
            # Normalize brand name for matching
            self.brand_name_normalized = self._normalize_text(brand_name)
        else:
            self.brand_name_normalized = None
    
    def verify_logo(
        self,
        detection: Dict[str, Any],
        image: np.ndarray,
        ocr_text: Optional[str] = None,
        min_logo_size: float = 0.01,
        max_logo_size: float = 0.25
    ) -> Dict[str, Any]:
        """
        Verify if a detection is actually a logo.
        
        Args:
            detection: Detection dict with bbox, confidence, etc.
            image: Full image array
            ocr_text: Optional OCR-extracted text for brand name matching
            min_logo_size: Minimum logo size ratio (0-1)
            max_logo_size: Maximum logo size ratio (0-1)
        
        Returns:
            Dict with:
            - verified: bool - True if verified as logo
            - verification_score: float (0-1) - Confidence in verification
            - verification_reasons: List[str] - Which checks passed
            - class_name: str - "logo" if verified, "unknown_graphic" if not
        """
        try:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                return self._reject_detection("Invalid bbox format")
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Extract ROI (Region of Interest)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return self._reject_detection("Empty ROI")
            
            image_height, image_width = image.shape[:2]
            roi_height, roi_width = roi.shape[:2]
            
            # Calculate verification checks
            checks_passed = []
            verification_score = 0.0
            
            # Check 1: Edge density + symmetry
            edge_score = self._check_edge_density_and_symmetry(roi)
            if edge_score >= 0.6:
                checks_passed.append("high_edge_density_symmetry")
                verification_score += 0.25
            
            # Check 2: Brand text match from OCR
            text_match_score = self._check_brand_text_match(roi, ocr_text, x1, y1, x2, y2, image_width, image_height)
            if text_match_score >= 0.5:
                checks_passed.append("brand_text_match")
                verification_score += 0.25
            
            # Check 3: Aspect ratio within expected logo bounds
            aspect_ratio = roi_width / roi_height if roi_height > 0 else 0
            if 0.8 <= aspect_ratio <= 3.5:
                checks_passed.append("valid_aspect_ratio")
                verification_score += 0.2
            
            # Check 4: Size ratio between minLogoSize and maxLogoSize
            size_ratio = (roi_width * roi_height) / (image_width * image_height)
            if min_logo_size <= size_ratio <= max_logo_size:
                checks_passed.append("valid_size_ratio")
                verification_score += 0.15
            
            # Check 5: Color consistency (simplified visual similarity)
            color_consistency = self._check_color_consistency(roi)
            if color_consistency >= 0.6:
                checks_passed.append("color_consistency")
                verification_score += 0.15
            
            # Verification requires at least 2 checks passed
            verified = len(checks_passed) >= 2
            
            if verified:
                return {
                    'verified': True,
                    'verification_score': min(1.0, verification_score),
                    'verification_reasons': checks_passed,
                    'class_name': 'logo',
                    'original_class': detection.get('class_name', 'unknown')
                }
            else:
                return {
                    'verified': False,
                    'verification_score': verification_score,
                    'verification_reasons': checks_passed,
                    'class_name': 'unknown_graphic',
                    'original_class': detection.get('class_name', 'unknown'),
                    'rejection_reason': f'Only {len(checks_passed)} verification checks passed (need 2+)'
                }
                
        except Exception as e:
            logger.error(f"Logo verification failed: {e}", exc_info=True)
            return self._reject_detection(f"Verification error: {str(e)}")
    
    def _check_edge_density_and_symmetry(self, roi: np.ndarray) -> float:
        """
        Check edge density and symmetry.
        Logos typically have high edge density and some symmetry.
        
        Returns:
            Score 0-1 (higher = more logo-like)
        """
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check horizontal symmetry
            h, w = gray.shape
            if h > 1 and w > 1:
                top_half = gray[:h//2, :]
                bottom_half = gray[h//2:, :]
                if bottom_half.shape[0] != top_half.shape[0]:
                    bottom_half = bottom_half[:top_half.shape[0], :]
                
                # Flip bottom half and compare
                bottom_flipped = cv2.flip(bottom_half, 0)
                if top_half.shape == bottom_flipped.shape:
                    symmetry_h = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_flipped.astype(float))) / 255.0
                else:
                    symmetry_h = 0.0
                
                # Check vertical symmetry
                left_half = gray[:, :w//2]
                right_half = gray[:, w//2:]
                if right_half.shape[1] != left_half.shape[1]:
                    right_half = right_half[:, :left_half.shape[1]]
                
                right_flipped = cv2.flip(right_half, 1)
                if left_half.shape == right_flipped.shape:
                    symmetry_v = 1.0 - np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float))) / 255.0
                else:
                    symmetry_v = 0.0
                
                symmetry_score = (symmetry_h + symmetry_v) / 2.0
            else:
                symmetry_score = 0.0
            
            # Combine edge density and symmetry
            # Logos should have both high edge density and some symmetry
            combined_score = (edge_density * 0.6) + (symmetry_score * 0.4)
            
            return min(1.0, combined_score)
            
        except Exception as e:
            logger.warning(f"Edge density/symmetry check failed: {e}")
            return 0.0
    
    def _check_brand_text_match(
        self,
        roi: np.ndarray,
        ocr_text: Optional[str],
        x1: int, y1: int, x2: int, y2: int,
        image_width: int, image_height: int
    ) -> float:
        """
        Check if brand text appears near the detection.
        
        Args:
            roi: Region of interest
            ocr_text: Full OCR text from image
            x1, y1, x2, y2: Bounding box coordinates
            image_width, image_height: Full image dimensions
        
        Returns:
            Score 0-1 (higher = more likely brand text match)
        """
        if not ocr_text or not self.brand_name_normalized:
            return 0.0
        
        try:
            # Normalize OCR text
            ocr_normalized = self._normalize_text(ocr_text)
            
            # Check if brand name appears in OCR text
            if self.brand_name_normalized in ocr_normalized:
                # Check if text is near the detection (within 2x detection size)
                # This is a simplified check - in production, you'd use OCR bounding boxes
                detection_size = max(x2 - x1, y2 - y1)
                proximity_threshold = detection_size * 2
                
                # For now, if brand name is in OCR, give partial score
                # Full score if we could verify proximity (would need OCR bbox)
                return 0.7  # Partial match - brand name found but proximity not verified
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Brand text match check failed: {e}")
            return 0.0
    
    def _check_color_consistency(self, roi: np.ndarray) -> float:
        """
        Check color consistency within ROI.
        Logos typically have consistent colors (not random UI elements).
        
        Returns:
            Score 0-1 (higher = more consistent colors)
        """
        try:
            if len(roi.shape) != 3:
                return 0.0
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate color variance
            # Lower variance = more consistent colors = more logo-like
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Normalize std (0-180 for H, 0-255 for S and V)
            h_std_norm = h_std / 180.0
            s_std_norm = s_std / 255.0
            v_std_norm = v_std / 255.0
            
            # Average normalized std
            avg_std = (h_std_norm + s_std_norm + v_std_norm) / 3.0
            
            # Lower std = higher consistency score
            consistency_score = 1.0 - min(1.0, avg_std)
            
            return consistency_score
            
        except Exception as e:
            logger.warning(f"Color consistency check failed: {e}")
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, remove special chars)"""
        if not text:
            return ""
        # Lowercase and remove non-alphanumeric
        normalized = re.sub(r'[^a-z0-9]', '', text.lower())
        return normalized
    
    def _reject_detection(self, reason: str) -> Dict[str, Any]:
        """Return rejection result"""
        return {
            'verified': False,
            'verification_score': 0.0,
            'verification_reasons': [],
            'class_name': 'unknown_graphic',
            'rejection_reason': reason
        }


def verify_logo_detections(
    detections: List[Dict[str, Any]],
    image: np.ndarray,
    ocr_text: Optional[str] = None,
    brand_name: Optional[str] = None,
    min_logo_size: float = 0.01,
    max_logo_size: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Verify multiple logo detections.
    
    Args:
        detections: List of detection dicts
        image: Full image array
        ocr_text: Optional OCR text for brand matching
        brand_name: Optional brand name
        min_logo_size: Minimum logo size ratio
        max_logo_size: Maximum logo size ratio
    
    Returns:
        List of detections with verification results added
    """
    if not detections:
        return []
    
    verifier = LogoVerifier(brand_name=brand_name)
    verified_detections = []
    
    for detection in detections:
        verification_result = verifier.verify_logo(
            detection,
            image,
            ocr_text=ocr_text,
            min_logo_size=min_logo_size,
            max_logo_size=max_logo_size
        )
        
        # Update detection with verification results
        detection['verification'] = verification_result
        detection['class_name'] = verification_result['class_name']
        detection['verified'] = verification_result['verified']
        detection['verification_score'] = verification_result['verification_score']
        
        verified_detections.append(detection)
    
    return verified_detections


def _encode_image_base64(image: np.ndarray) -> str:
    """Encode image to base64 for VLM API"""
    try:
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        h, w = image.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""


def _identify_brand_in_region(image: np.ndarray, region_bbox: List[int]) -> Optional[Dict[str, Any]]:
    """
    Use VLM to identify brand in a specific region of an image.
    
    Args:
        image: Full image array
        region_bbox: [x1, y1, x2, y2] coordinates
        
    Returns:
        Dict with brand_name, confidence, reasoning or None if failed
    """
    try:
        # Crop the region
        x1, y1, x2, y2 = [int(c) for c in region_bbox]
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        logo_crop = image[y1:y2, x1:x2]
        if logo_crop.size == 0:
            return None
        
        # Encode and call VLM
        image_b64 = _encode_image_base64(logo_crop)
        if not image_b64:
            return None
        
        prompt = """Analyze this logo and identify the brand name.

Respond in JSON format:
{
  "brand_name": "Brand Name or Unknown",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

If you cannot identify the brand, respond with "Unknown" as the brand_name."""

        payload = {
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{VLM_API_URL.rsplit('/v1', 1)[0]}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"VLM brand identification failed: {response.status_code}")
            return None
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Parse JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        data = json.loads(content.strip())
        
        return {
            'brand_name': data.get('brand_name', 'Unknown'),
            'confidence': data.get('confidence', 0.0),
            'reasoning': data.get('reasoning', '')
        }
        
    except Exception as e:
        logger.warning(f"Brand identification failed: {e}")
        return None


def check_brand_mismatch(
    detections: List[Dict[str, Any]],
    image: np.ndarray,
    expected_brand: str,
    reference_logo: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Check if detected logos match or mismatch with expected brand.
    
    THIS IS THE KEY FUNCTION for your use case:
    - User uploads Nike logo
    - Content shows Puma
    - System detects: WRONG BRAND!
    
    Args:
        detections: List of verified logo detections
        image: Full image array
        expected_brand: Brand name user provided (e.g., "Nike")
        reference_logo: Optional reference logo image for visual comparison
        
    Returns:
        Updated detections with brand_match results
    """
    if not detections or not expected_brand:
        return detections
    
    expected_brand_normalized = expected_brand.lower().strip()
    
    for detection in detections:
        # Try to identify brand in detected region
        brand_info = _identify_brand_in_region(
            image, 
            detection.get('bbox', [])
        )
        
        if brand_info:
            detected_brand = brand_info.get('brand_name', 'Unknown')
            detected_brand_normalized = detected_brand.lower().strip() if detected_brand else ''
            
            # Check for mismatch
            is_match = (
                detected_brand_normalized == expected_brand_normalized or
                expected_brand_normalized in detected_brand_normalized or
                detected_brand_normalized in expected_brand_normalized
            )
            
            detection['brand_match'] = {
                'expected_brand': expected_brand,
                'detected_brand': detected_brand,
                'is_match': is_match,
                'is_mismatch': not is_match,  # THIS IS WHAT YOU WANT!
                'confidence': brand_info.get('confidence', 0.0),
                'reasoning': brand_info.get('reasoning', ''),
                'status': 'correct_brand' if is_match else 'wrong_brand'
            }
            
            if not is_match:
                logger.warning(f"🚫 BRAND MISMATCH: Expected '{expected_brand}' but found '{detected_brand}'")
            else:
                logger.info(f"✅ BRAND MATCH: Found '{expected_brand}'")
        else:
            # Could not identify brand
            detection['brand_match'] = {
                'expected_brand': expected_brand,
                'detected_brand': 'Unknown',
                'is_match': None,
                'is_mismatch': None,
                'confidence': 0.0,
                'reasoning': 'Could not identify brand from logo',
                'status': 'unknown'
            }
    
    return detections

