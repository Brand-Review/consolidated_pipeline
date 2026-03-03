"""
Logo Zone Detection
Deterministic logo placement zone analysis - pure geometry, no AI.

Responsibilities:
- Define placement zones (top-left, top-center, top-right, center, etc.)
- Detect which zone a logo is in based on its position
- Calculate zone boundaries

FORBIDDEN:
- No AI-based placement decisions
- No scoring
- No compliance judgments (zones are facts, not compliance rules)
"""

from typing import Dict, Any, List, Tuple, Optional
from .logo_geometry import calculate_center, normalize_bbox
import logging

logger = logging.getLogger(__name__)


# Zone definitions as (x_min, y_min, x_max, y_max) ratios (0.0-1.0)
# Zones are deterministic, geometric boundaries
ZONES = {
    "top-left": (0.0, 0.0, 0.33, 0.33),
    "top-center": (0.33, 0.0, 0.66, 0.33),
    "top-right": (0.66, 0.0, 1.0, 0.33),
    "center-left": (0.0, 0.33, 0.33, 0.66),
    "center": (0.33, 0.33, 0.66, 0.66),
    "center-right": (0.66, 0.33, 1.0, 0.66),
    "bottom-left": (0.0, 0.66, 0.33, 1.0),
    "bottom-center": (0.33, 0.66, 0.66, 1.0),
    "bottom-right": (0.66, 0.66, 1.0, 1.0),
}

# Zone labels for human readability
ZONE_LABELS = {
    "top-left": "Top Left",
    "top-center": "Top Center",
    "top-right": "Top Right",
    "center-left": "Center Left",
    "center": "Center",
    "center-right": "Center Right",
    "bottom-left": "Bottom Left",
    "bottom-center": "Bottom Center",
    "bottom-right": "Bottom Right",
}


class LogoZoneDetector:
    """
    Logo Zone Detector - Determines which zone a logo is in based on geometry.
    
    This is a pure geometric calculation - no AI, no scoring, no compliance judgment.
    Zones are facts about position, not rules about compliance.
    """
    
    def __init__(self):
        """Initialize logo zone detector"""
        self.zones = ZONES.copy()
        self.zone_labels = ZONE_LABELS.copy()
    
    def detect_zone(
        self,
        bbox_normalized: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect which zone a logo is in based on its normalized bounding box.
        
        Args:
            bbox_normalized: Normalized bounding box from normalize_bbox()
            
        Returns:
            Dictionary with zone information:
            {
                "zone": str,  # Zone name (e.g., "top-left", "center")
                "zone_label": str,  # Human-readable zone label
                "center": (float, float),  # Logo center (x_ratio, y_ratio)
                "bbox": Dict[str, float],  # Normalized bounding box
                "overlaps_zones": [str],  # List of zones the logo overlaps (if spanning multiple)
                "confidence": float  # Confidence in zone detection (geometric certainty)
            }
        """
        try:
            # Calculate logo center
            center_x, center_y = calculate_center(bbox_normalized)
            
            # Get bounding box coordinates
            x1 = bbox_normalized.get('x1', bbox_normalized.get('x', 0.0))
            y1 = bbox_normalized.get('y1', bbox_normalized.get('y', 0.0))
            x2 = bbox_normalized.get('x2', bbox_normalized.get('x', 0.0) + bbox_normalized.get('width', 0.0))
            y2 = bbox_normalized.get('y2', bbox_normalized.get('y', 0.0) + bbox_normalized.get('height', 0.0))
            
            # Primary zone: Determine based on center position
            primary_zone = self._get_zone_by_center(center_x, center_y)
            
            # Check if logo overlaps multiple zones
            overlaps_zones = self._get_overlapping_zones(x1, y1, x2, y2)
            
            # If logo overlaps multiple zones, use the one with largest overlap
            if len(overlaps_zones) > 1:
                primary_zone = self._get_zone_by_overlap(x1, y1, x2, y2, overlaps_zones)
            
            # Confidence: High if center is clearly within zone, lower if on boundary
            confidence = self._calculate_zone_confidence(
                center_x, center_y, primary_zone
            )
            
            return {
                "zone": primary_zone,
                "zone_label": self.zone_labels.get(primary_zone, primary_zone),
                "center": (center_x, center_y),
                "bbox": bbox_normalized,
                "overlaps_zones": overlaps_zones,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Logo zone detection failed: {e}", exc_info=True)
            return {
                "zone": "unknown",
                "zone_label": "Unknown",
                "center": (0.5, 0.5),
                "bbox": bbox_normalized,
                "overlaps_zones": [],
                "confidence": 0.0
            }
    
    def _get_zone_by_center(self, center_x: float, center_y: float) -> str:
        """
        Determine zone based on center position.
        
        Args:
            center_x: X coordinate of logo center (0.0-1.0)
            center_y: Y coordinate of logo center (0.0-1.0)
            
        Returns:
            Zone name (e.g., "top-left", "center")
        """
        # Determine horizontal zone
        if center_x < 0.33:
            h_zone = "left"
        elif center_x < 0.66:
            h_zone = "center"
        else:
            h_zone = "right"
        
        # Determine vertical zone
        if center_y < 0.33:
            v_zone = "top"
        elif center_y < 0.66:
            v_zone = "center"
        else:
            v_zone = "bottom"
        
        # Combine to get zone name
        if v_zone == "center" and h_zone == "center":
            return "center"
        elif v_zone == "center":
            return f"{h_zone}-center"
        elif h_zone == "center":
            return f"{v_zone}-center"
        else:
            return f"{v_zone}-{h_zone}"
    
    def _get_overlapping_zones(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> List[str]:
        """
        Get all zones that the logo bounding box overlaps.
        
        Args:
            x1, y1: Top-left corner of bounding box (normalized)
            x2, y2: Bottom-right corner of bounding box (normalized)
            
        Returns:
            List of zone names that the logo overlaps
        """
        overlapping_zones = []
        
        for zone_name, (zone_x1, zone_y1, zone_x2, zone_y2) in self.zones.items():
            # Check if bounding boxes overlap
            if self._boxes_overlap(x1, y1, x2, y2, zone_x1, zone_y1, zone_x2, zone_y2):
                overlapping_zones.append(zone_name)
        
        return overlapping_zones if overlapping_zones else ["center"]
    
    def _boxes_overlap(
        self,
        x1a: float, y1a: float, x2a: float, y2a: float,
        x1b: float, y1b: float, x2b: float, y2b: float
    ) -> bool:
        """
        Check if two bounding boxes overlap.
        
        Args:
            x1a, y1a, x2a, y2a: First bounding box
            x1b, y1b, x2b, y2b: Second bounding box
            
        Returns:
            True if boxes overlap, False otherwise
        """
        # Boxes overlap if they intersect in both x and y dimensions
        x_overlap = not (x2a < x1b or x2b < x1a)
        y_overlap = not (y2a < y1b or y2b < y1a)
        
        return x_overlap and y_overlap
    
    def _get_zone_by_overlap(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        overlapping_zones: List[str]
    ) -> str:
        """
        Determine primary zone based on overlap area.
        
        Args:
            x1, y1, x2, y2: Logo bounding box
            overlapping_zones: List of zones the logo overlaps
            
        Returns:
            Zone name with largest overlap
        """
        max_overlap = 0.0
        primary_zone = overlapping_zones[0]  # Default to first zone
        
        for zone_name in overlapping_zones:
            zone_x1, zone_y1, zone_x2, zone_y2 = self.zones[zone_name]
            
            # Calculate overlap area
            overlap_x1 = max(x1, zone_x1)
            overlap_y1 = max(y1, zone_y1)
            overlap_x2 = min(x2, zone_x2)
            overlap_y2 = min(y2, zone_y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                if overlap_area > max_overlap:
                    max_overlap = overlap_area
                    primary_zone = zone_name
        
        return primary_zone
    
    def _calculate_zone_confidence(
        self,
        center_x: float,
        center_y: float,
        zone_name: str
    ) -> float:
        """
        Calculate confidence in zone detection based on how centered the logo is within the zone.
        
        Args:
            center_x: X coordinate of logo center (0.0-1.0)
            center_y: Y coordinate of logo center (0.0-1.0)
            zone_name: Detected zone name
            
        Returns:
            Confidence value (0.0-1.0)
        """
        if zone_name not in self.zones:
            return 0.5  # Default confidence for unknown zones
        
        zone_x1, zone_y1, zone_x2, zone_y2 = self.zones[zone_name]
        
        # Calculate zone center
        zone_center_x = (zone_x1 + zone_x2) / 2.0
        zone_center_y = (zone_y1 + zone_y2) / 2.0
        
        # Calculate zone size
        zone_width = zone_x2 - zone_x1
        zone_height = zone_y2 - zone_y1
        
        # Calculate distance from logo center to zone center (normalized by zone size)
        dx = abs(center_x - zone_center_x) / max(zone_width, 0.01)
        dy = abs(center_y - zone_center_y) / max(zone_height, 0.01)
        
        # Distance from center (0.0 = perfectly centered, 1.0+ = at edge or outside)
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # Confidence: Higher when closer to zone center
        # At center (distance = 0): confidence = 1.0
        # At edge (distance = 1.0): confidence = 0.5
        # Outside (distance > 1.0): confidence = 0.0
        confidence = max(0.0, 1.0 - (distance * 0.5))
        
        return float(confidence)
    
    def is_in_zone(
        self,
        bbox_normalized: Dict[str, float],
        target_zone: str
    ) -> bool:
        """
        Check if logo is in a specific zone.
        
        Args:
            bbox_normalized: Normalized bounding box
            target_zone: Zone name to check
            
        Returns:
            True if logo is in target zone, False otherwise
        """
        if target_zone not in self.zones:
            logger.warning(f"Unknown zone: {target_zone}")
            return False
        
        zone_info = self.detect_zone(bbox_normalized)
        return zone_info["zone"] == target_zone
    
    def get_all_zones(self) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get all zone definitions.
        
        Returns:
            Dictionary mapping zone names to (x1, y1, x2, y2) tuples
        """
        return dict(self.zones)

