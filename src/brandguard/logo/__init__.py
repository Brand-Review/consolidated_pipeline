"""
Logo Geometry Module
Deterministic logo placement analysis - pure geometry, no AI judgments.
"""

from .logo_geometry import normalize_bbox, calculate_center, calculate_size_ratio, calculate_edge_distances, is_centered
from .logo_zones import LogoZoneDetector, ZONES
from .logo_violations import LogoPlacementViolations

__all__ = [
    'normalize_bbox', 'calculate_center', 'calculate_size_ratio', 
    'calculate_edge_distances', 'is_centered',
    'LogoZoneDetector', 'ZONES',
    'LogoPlacementViolations'
]

