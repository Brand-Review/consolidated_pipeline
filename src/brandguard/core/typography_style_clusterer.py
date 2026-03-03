"""
Typography Style Clusterer
Clusters OCR text regions into typography styles (2-5 max).

CRITICAL:
- OCR words must NOT carry font or typography metadata
- Cluster tokens into styles instead of per-word output
- Remove OCR confidence from typography output
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class TypographyStyleClusterer:
    """Clusters text regions into typography styles"""
    
    def __init__(self, typography_classifier):
        """Initialize with typography classifier"""
        self.typography_classifier = typography_classifier
    
    def cluster_text_regions(
        self,
        image: np.ndarray,
        text_regions: List[Dict[str, Any]],
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        hierarchy_detected: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Cluster text regions into typography styles (max 3).
        Output fields only:
        - role
        - approxFontSizePx (optional)
        - weight (optional)
        - confidenceLevel (low|medium|high)
        """
        if not text_regions:
            return []

        try:
            if not hierarchy_detected:
                return [{
                    'role': 'ui',
                    'confidenceLevel': 'low'
                }]

            classified = []
            for region in text_regions:
                if not isinstance(region, dict) or 'bbox' not in region:
                    continue
                classification_result = self.typography_classifier.classify_text_region(
                    image, region, image_height, image_width
                )
                font_classification = classification_result.get('fontClassification', {})
                bbox = region.get('bbox', [])
                bbox_h = 0.0
                if isinstance(bbox, list) and len(bbox) == 4:
                    bbox_h = bbox[3] - bbox[1]
                if bbox_h <= 1.0:
                    font_size_ratio = max(0.0, float(bbox_h))
                else:
                    font_size_ratio = float(bbox_h) / float(image_height or 1)
                approx_font_size_px = font_size_ratio * float(image_height or 1)
                classified.append({
                    'approxFontSizePx': round(approx_font_size_px, 1),
                    'fontSizeRatio': font_size_ratio,
                    'role': font_classification.get('role', 'body'),
                    'weight': font_classification.get('weight')
                })

            if not classified:
                return [{'role': 'ui', 'confidenceLevel': 'low'}]

            style_groups = {'heading': [], 'body': [], 'ui': []}
            for region in classified:
                size_ratio = region.get('fontSizeRatio', 0.0)
                if size_ratio >= 0.035:
                    role = 'heading'
                elif size_ratio < 0.020:
                    role = 'ui'
                else:
                    role = 'body'
                style_groups[role].append(region)

            styles = []
            for role in ['heading', 'body', 'ui']:
                regions = style_groups.get(role, [])
                if not regions:
                    continue
                avg_size = sum(r.get('approxFontSizePx', 0.0) for r in regions) / len(regions)
                weights = [r.get('weight') for r in regions if r.get('weight')]
                weight = max(set(weights), key=weights.count) if weights else None
                confidence_level = 'high' if len(regions) >= 5 else 'medium'
                
                # ✅ FIX: Add relativeSize (large/medium/small) instead of exact fontSizePx
                # Determine relative size based on role and average size
                if role == 'heading':
                    relative_size = 'large'
                elif role == 'ui':
                    relative_size = 'small'
                else:
                    relative_size = 'medium'
                
                style = {
                    'role': role,
                    'relativeSize': relative_size,  # ✅ NEW: large | medium | small
                    'confidenceLevel': confidence_level
                }
                if avg_size > 0:
                    style['approxFontSizePx'] = round(avg_size, 1)  # Keep for reference but not primary
                if weight:
                    style['weight'] = weight
                # ✅ NO fontFamily, NO fontName - font identity not exposed
                styles.append(style)

            return styles[:3]  # Max 3 styles (2-5 range, using 3 as default)
        except Exception as e:
            logger.error(f"Typography style clustering failed: {e}")
            return [{'role': 'ui', 'confidenceLevel': 'low'}]

def _create_style_from_regions(
        self,
        regions: List[Dict[str, Any]],
        default_role: str
    ) -> Dict[str, Any]:
        """Create a style object from a list of regions"""
        if not regions:
            return {}
        
        avg_size = np.mean([r.get('approximateSizePx', 12) for r in regions])
        avg_ratio = np.mean([r.get('fontSizeRatio', 0.012) for r in regions])
        avg_confidence = np.mean([r.get('classificationConfidence', 0.5) for r in regions])
        
        roles = [r.get('role', default_role) for r in regions]
        serifs = [r.get('serif') for r in regions if r.get('serif')]
        weights = [r.get('weight') for r in regions if r.get('weight')]
        cases = [r.get('case') for r in regions if r.get('case')]
        
        most_common_role = max(set(roles), key=roles.count) if roles else default_role
        most_common_serif = max(set(serifs), key=serifs.count) if serifs else 'sans-serif'
        most_common_weight = max(set(weights), key=weights.count) if weights else 'regular'
        most_common_case = max(set(cases), key=cases.count) if cases else 'title'
        
        # Map role
        if most_common_role in ['headline', 'heading']:
            role = 'heading'
        elif most_common_role in ['ui-label', 'ui']:
            role = 'ui'
        else:
            role = 'body'
        
        # ✅ FIX: Determine relativeSize based on role
        if role == 'heading':
            relative_size = 'large'
        elif role == 'ui':
            relative_size = 'small'
        else:
            relative_size = 'medium'
        
        return {
            'role': role,
            'relativeSize': relative_size,  # ✅ NEW: large | medium | small
            'approxFontSizePx': round(avg_size, 1),  # Keep for reference
            'fontSizeRatio': round(avg_ratio, 4),
            'weight': most_common_weight,
            'case': most_common_case,
            'serif': most_common_serif,
            # ✅ NO fontFamily, NO fontName - font identity completely removed
            'confidenceLevel': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.5 else 'low',
            'regionCount': len(regions)
        }

