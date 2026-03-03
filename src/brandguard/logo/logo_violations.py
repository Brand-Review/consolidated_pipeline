"""
Logo Placement Violations
Deterministic violation detection - pure math, no AI.

Rules:
- Center placement = violation (unless explicitly allowed)
- Logo too large/small = violation
- Edge padding < min = violation
"""

from typing import Dict, Any, List, Optional, Tuple
from .logo_geometry import normalize_bbox, calculate_center, calculate_size_ratio, calculate_edge_distances, is_centered
from .logo_zones import LogoZoneDetector, ZONES
import logging

logger = logging.getLogger(__name__)


class LogoPlacementViolations:
    """
    Logo Placement Violations Detector
    
    Detects placement violations using deterministic rules (pure math, no AI).
    """
    
    def __init__(
        self,
        allowed_zones: Optional[List[str]] = None,
        min_logo_size: float = 0.01,  # 1% of image area
        max_logo_size: float = 0.25,   # 25% of image area
        min_edge_distance: float = 0.05  # 5% of image dimension
    ):
        """
        Initialize logo placement violations detector.
        
        Args:
            allowed_zones: List of allowed placement zones (default: ["top-left", "top-center", "top-right"])
            min_logo_size: Minimum logo size as ratio of image area (default: 0.01 = 1%)
            max_logo_size: Maximum logo size as ratio of image area (default: 0.25 = 25%)
            min_edge_distance: Minimum distance from edges as ratio (default: 0.05 = 5%)
        """
        self.allowed_zones = allowed_zones or ["top-left", "top-center", "top-right", "bottom-left", "bottom-right"]
        self.min_logo_size = min_logo_size
        self.max_logo_size = max_logo_size
        self.min_edge_distance = min_edge_distance
        self.zone_detector = LogoZoneDetector()
    
    def detect_violations(
        self,
        bbox_normalized: Dict[str, float],
        image_width: int,
        image_height: int,
        allow_center: bool = False
    ) -> Dict[str, Any]:
        """
        Detect logo placement violations.
        
        Args:
            bbox_normalized: Normalized bounding box from normalize_bbox()
            image_width: Image width in pixels
            image_height: Image height in pixels
            allow_center: Whether center placement is allowed (default: False)
            
        Returns:
            Dictionary with violation information:
            {
                "zone": str,  # Detected zone
                "sizeRatio": float,  # Logo size as ratio of image area
                "violations": [str],  # List of violation descriptions
                "hasViolations": bool,  # True if any violations detected
                "placementOk": bool,  # True if placement is valid
                "sizeOk": bool,  # True if size is valid
                "edgePaddingOk": bool  # True if edge padding is valid
            }
        """
        violations = []
        
        # Detect zone
        zone_info = self.zone_detector.detect_zone(bbox_normalized)
        detected_zone = zone_info["zone"]
        
        # Calculate size ratio
        size_ratio = calculate_size_ratio(bbox_normalized)
        
        # Calculate edge distances
        edge_distances = calculate_edge_distances(bbox_normalized)
        
        # Check zone violation
        placement_ok = self._check_zone_violation(detected_zone, allow_center, violations)
        
        # Check center violation (strict)
        if is_centered(bbox_normalized, tolerance=0.1) and not allow_center:
            violations.append("Logo is centered (center placement is not allowed)")
            placement_ok = False
        
        # Check size violations
        size_ok = self._check_size_violations(size_ratio, violations)
        
        # Check edge padding violations
        edge_padding_ok = self._check_edge_padding_violations(edge_distances, violations)
        
        return {
            "zone": detected_zone,
            "sizeRatio": size_ratio,
            "violations": violations,
            "hasViolations": len(violations) > 0,
            "placementOk": placement_ok,
            "sizeOk": size_ok,
            "edgePaddingOk": edge_padding_ok,
            "edgeDistances": edge_distances,
            "zoneInfo": zone_info
        }
    
    def _check_zone_violation(
        self,
        detected_zone: str,
        allow_center: bool,
        violations: List[str]
    ) -> bool:
        """
        Check if detected zone is in allowed zones.
        
        Args:
            detected_zone: Detected zone name
            allow_center: Whether center is allowed
            violations: List to append violations to
            
        Returns:
            True if zone is allowed, False otherwise
        """
        # Always allow allowed zones
        if detected_zone in self.allowed_zones:
            return True
        
        # Check center exception
        if detected_zone == "center" and allow_center:
            return True
        
        # Zone is not allowed
        violations.append(f"Logo in '{detected_zone}' zone (allowed zones: {', '.join(self.allowed_zones)})")
        return False
    
    def _check_size_violations(
        self,
        size_ratio: float,
        violations: List[str]
    ) -> bool:
        """
        Check if logo size is within allowed range.
        
        Args:
            size_ratio: Logo size as ratio of image area
            violations: List to append violations to
            
        Returns:
            True if size is valid, False otherwise
        """
        size_ok = True
        
        # Check minimum size
        if size_ratio < self.min_logo_size:
            violations.append(f"Logo too small ({size_ratio:.2%} of image, minimum {self.min_logo_size:.2%} required)")
            size_ok = False
        
        # Check maximum size
        if size_ratio > self.max_logo_size:
            violations.append(f"Logo too large ({size_ratio:.2%} of image, maximum {self.max_logo_size:.2%} allowed)")
            size_ok = False
        
        return size_ok
    
    def _check_edge_padding_violations(
        self,
        edge_distances: Dict[str, float],
        violations: List[str]
    ) -> bool:
        """
        Check if logo has sufficient edge padding.
        
        Args:
            edge_distances: Dictionary with edge distances (top, bottom, left, right)
            violations: List to append violations to
            
        Returns:
            True if edge padding is valid, False otherwise
        """
        edge_padding_ok = True
        
        # Check each edge
        if edge_distances["top"] < self.min_edge_distance:
            violations.append(f"Insufficient top padding ({edge_distances['top']:.2%}, minimum {self.min_edge_distance:.2%} required)")
            edge_padding_ok = False
        
        if edge_distances["bottom"] < self.min_edge_distance:
            violations.append(f"Insufficient bottom padding ({edge_distances['bottom']:.2%}, minimum {self.min_edge_distance:.2%} required)")
            edge_padding_ok = False
        
        if edge_distances["left"] < self.min_edge_distance:
            violations.append(f"Insufficient left padding ({edge_distances['left']:.2%}, minimum {self.min_edge_distance:.2%} required)")
            edge_padding_ok = False
        
        if edge_distances["right"] < self.min_edge_distance:
            violations.append(f"Insufficient right padding ({edge_distances['right']:.2%}, minimum {self.min_edge_distance:.2%} required)")
            edge_padding_ok = False
        
        return edge_padding_ok

