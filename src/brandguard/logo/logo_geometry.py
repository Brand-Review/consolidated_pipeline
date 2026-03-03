"""
Logo Geometry Utilities
Pure geometric calculations for logo placement analysis - no AI, no scoring.

Responsibilities:
- Normalize bounding boxes to ratios (0.0-1.0)
- Calculate logo center position
- Calculate logo size relative to image
- Calculate edge distances

FORBIDDEN:
- No AI-based placement decisions
- No scoring
- No compliance judgments
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int
) -> Dict[str, float]:
    """
    Normalize bounding box coordinates to ratios (0.0-1.0).
    
    Args:
        bbox: Bounding box as [x, y, width, height] or [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Dictionary with normalized bounding box:
        {
            "x": float,  # x position ratio (0.0-1.0)
            "y": float,  # y position ratio (0.0-1.0)
            "width": float,  # width ratio (0.0-1.0)
            "height": float,  # height ratio (0.0-1.0)
            "x1": float,  # left edge ratio (0.0-1.0)
            "y1": float,  # top edge ratio (0.0-1.0)
            "x2": float,  # right edge ratio (0.0-1.0)
            "y2": float  # bottom edge ratio (0.0-1.0)
        }
    """
    if not bbox or len(bbox) < 4:
        raise ValueError("Bounding box must have at least 4 values")
    
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    
    # Handle two bbox formats:
    # Format 1: [x, y, width, height] (center + size)
    # Format 2: [x1, y1, x2, y2] (corners)
    
    if len(bbox) == 4:
        # Try to detect format based on values
        # If max value > 1.0, assume pixel coordinates
        # Otherwise assume already normalized
        
        if max(bbox) <= 1.0 and min(bbox) >= 0.0:
            # Already normalized, assume [x, y, width, height]
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
        else:
            # Pixel coordinates, detect format
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # Likely [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
            else:
                # Likely [x, y, width, height]
                x, y, w, h = bbox
                x1, y1 = x, y
                x2, y2 = x + w, y + h
            
            # Normalize to ratios
            x = x / image_width
            y = y / image_height
            w = w / image_width
            h = h / image_height
            x1 = x1 / image_width
            y1 = y1 / image_height
            x2 = x2 / image_width
            y2 = y2 / image_height
    
    # Ensure values are in valid range [0.0, 1.0]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    return {
        "x": float(x),
        "y": float(y),
        "width": float(w),
        "height": float(h),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2)
    }


def calculate_center(bbox_normalized: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate center position of normalized bounding box.
    
    Args:
        bbox_normalized: Normalized bounding box from normalize_bbox()
        
    Returns:
        Tuple of (center_x_ratio, center_y_ratio) where both are 0.0-1.0
    """
    x1 = bbox_normalized.get('x1', bbox_normalized.get('x', 0.0))
    y1 = bbox_normalized.get('y1', bbox_normalized.get('y', 0.0))
    width = bbox_normalized.get('width', 0.0)
    height = bbox_normalized.get('height', 0.0)
    
    # Calculate center
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)
    
    # Ensure in valid range
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    
    return (float(center_x), float(center_y))


def calculate_size_ratio(bbox_normalized: Dict[str, float]) -> float:
    """
    Calculate logo size as ratio of total image area.
    
    Args:
        bbox_normalized: Normalized bounding box from normalize_bbox()
        
    Returns:
        Size ratio (0.0-1.0) representing logo area / image area
    """
    width = bbox_normalized.get('width', 0.0)
    height = bbox_normalized.get('height', 0.0)
    
    # Area ratio = (width * height) / (1.0 * 1.0) = width * height
    size_ratio = width * height
    
    # Ensure in valid range
    size_ratio = max(0.0, min(1.0, size_ratio))
    
    return float(size_ratio)


def calculate_edge_distances(bbox_normalized: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate distances from logo edges to image edges.
    
    Args:
        bbox_normalized: Normalized bounding box from normalize_bbox()
        
    Returns:
        Dictionary with edge distances (all ratios 0.0-1.0):
        {
            "top": float,  # Distance from top edge to top of image
            "bottom": float,  # Distance from bottom edge to bottom of image
            "left": float,  # Distance from left edge to left of image
            "right": float  # Distance from right edge to right of image
        }
    """
    x1 = bbox_normalized.get('x1', bbox_normalized.get('x', 0.0))
    y1 = bbox_normalized.get('y1', bbox_normalized.get('y', 0.0))
    x2 = bbox_normalized.get('x2', bbox_normalized.get('x', 0.0) + bbox_normalized.get('width', 0.0))
    y2 = bbox_normalized.get('y2', bbox_normalized.get('y', 0.0) + bbox_normalized.get('height', 0.0))
    
    # Calculate distances to edges
    top_distance = y1  # Distance from top of logo to top of image
    bottom_distance = 1.0 - y2  # Distance from bottom of logo to bottom of image
    left_distance = x1  # Distance from left of logo to left of image
    right_distance = 1.0 - x2  # Distance from right of logo to right of image
    
    # Ensure non-negative
    top_distance = max(0.0, top_distance)
    bottom_distance = max(0.0, bottom_distance)
    left_distance = max(0.0, left_distance)
    right_distance = max(0.0, right_distance)
    
    return {
        "top": float(top_distance),
        "bottom": float(bottom_distance),
        "left": float(left_distance),
        "right": float(right_distance)
    }


def is_centered(bbox_normalized: Dict[str, float], tolerance: float = 0.1) -> bool:
    """
    Check if logo is centered in image (within tolerance).
    
    Args:
        bbox_normalized: Normalized bounding box from normalize_bbox()
        tolerance: Tolerance for center detection (default 0.1 = 10% of image)
        
    Returns:
        True if logo is centered (within tolerance), False otherwise
    """
    center_x, center_y = calculate_center(bbox_normalized)
    
    # Check if center is within tolerance of image center (0.5, 0.5)
    center_threshold = 0.5
    is_x_centered = abs(center_x - center_threshold) <= tolerance
    is_y_centered = abs(center_y - center_threshold) <= tolerance
    
    return is_x_centered and is_y_centered

