"""
Logo Candidate Filtering Layer
Deterministic filtering of YOLO detections before compliance evaluation
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def filter_logo_candidates(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filter logo detections using deterministic rules.
    
    Filters applied in order:
    1. Confidence filter (>= 0.75)
    2. Area ratio filter (minLogoSize to maxLogoSize)
    3. Aspect ratio filter (0.8 to 3.5)
    4. Non-Maximum Suppression (IoU threshold 0.5)
    5. Position bias (confidence adjustment for center zone)
    
    Args:
        detections: List of raw logo detections with bbox and confidence
        image_width: Image width in pixels
        image_height: Image height in pixels
        config: Optional config dict with:
            - logoConfidenceThreshold (default: 0.75)
            - minLogoSize (default: 0.01)
            - maxLogoSize (default: 0.25)
            - allowedZones (optional, for position bias)
    
    Returns:
        Filtered list of detections with added metadata:
        - bbox: [x1, y1, x2, y2]
        - confidence: float (0-1)
        - size_ratio: float (0-1)
        - aspect_ratio: float
        - zone: str (e.g., "top-left", "center")
    """
    try:
        # Defensive checks
        if not detections:
            return []
        
        # Handle None detections
        if detections is None:
            logger.warning("filter_logo_candidates received None detections, returning empty list")
            return []
        
        # Validate image dimensions
        if image_width <= 0 or image_height <= 0:
            logger.error(f"Invalid image dimensions: {image_width}x{image_height}, returning empty list")
            return []
        
        config = config or {}
        # Confidence threshold: fixed at 0.75 per requirements (filters false positives)
        # This is separate from YOLO's confidence threshold
        confidence_threshold = 0.75
        min_logo_size = config.get('minLogoSize', 0.01)
        max_logo_size = config.get('maxLogoSize', 0.25)
        allowed_zones = config.get('allowedZones', [])
        
        image_area = image_width * image_height
        
        if image_area <= 0:
            logger.error(f"Invalid image area: {image_area}, returning empty list")
            return []
        
        # Step 1: Confidence Filter
        # Discard detections with confidence < 0.75
        filtered = []
        for detection in detections:
            if not isinstance(detection, dict):
                logger.warning(f"Skipping invalid detection (not a dict): {type(detection)}")
                continue
            
            try:
                confidence = float(detection.get('confidence', 0.0))
                if confidence >= confidence_threshold:
                    filtered.append(detection)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping detection with invalid confidence: {e}")
                continue
        
        logger.debug(f"After confidence filter (>= {confidence_threshold}): {len(filtered)} detections")
        
        # Step 2: Area Ratio Filter
        # Compute size_ratio and filter by min/max logo size
        area_filtered = []
        for detection in filtered:
            try:
                bbox = detection.get('bbox', [])
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                
                # Extract bbox coordinates (assuming [x1, y1, x2, y2] format)
                # Ensure coordinates are in correct order
                try:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping detection with invalid bbox coordinates: {e}")
                    continue
                
                # Calculate width and height (ensure positive)
                w = max(abs(x2 - x1), 1)  # At least 1 pixel
                h = max(abs(y2 - y1), 1)  # At least 1 pixel
                
                # Calculate size ratio
                detection_area = w * h
                size_ratio = detection_area / image_area if image_area > 0 else 0
                
                # Filter by size
                if min_logo_size <= size_ratio <= max_logo_size:
                    # Add metadata
                    detection['size_ratio'] = size_ratio
                    area_filtered.append(detection)
            except Exception as e:
                logger.warning(f"Error processing detection in area filter: {e}")
                continue
        
        logger.debug(f"After area ratio filter ({min_logo_size}-{max_logo_size}): {len(area_filtered)} detections")
        
        # Step 3: Aspect Ratio Filter
        # Filter by aspect ratio (0.8 to 3.5)
        aspect_filtered = []
        for detection in area_filtered:
            try:
                bbox = detection.get('bbox', [])
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                
                try:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping detection with invalid bbox in aspect filter: {e}")
                    continue
                
                w = max(abs(x2 - x1), 1)  # At least 1 pixel
                h = max(abs(y2 - y1), 1)  # At least 1 pixel
                
                # Calculate aspect ratio (width/height)
                aspect_ratio = w / h
                
                # Filter by aspect ratio
                if 0.8 <= aspect_ratio <= 3.5:
                    detection['aspect_ratio'] = aspect_ratio
                    aspect_filtered.append(detection)
            except Exception as e:
                logger.warning(f"Error processing detection in aspect filter: {e}")
                continue
        
        logger.debug(f"After aspect ratio filter (0.8-3.5): {len(aspect_filtered)} detections")
        
        # Step 4: Non-Maximum Suppression (NMS)
        # Remove overlapping detections using IoU threshold 0.5
        nms_filtered = _apply_nms(aspect_filtered, iou_threshold=0.5)
        
        logger.debug(f"After NMS (IoU 0.5): {len(nms_filtered)} detections")
        
        # Step 5: Position Bias (confidence adjustment)
        # Reduce confidence for center zone detections (multiply by 0.7)
        # Do NOT discard based on position alone
        final_filtered = []
        for detection in nms_filtered:
            bbox = detection.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            # Determine zone
            zone = _determine_zone(bbox, image_width, image_height)
            detection['zone'] = zone
            
            # Apply position bias for center zone
            if zone == 'center':
                original_confidence = detection.get('confidence', 0.0)
                detection['confidence'] = original_confidence * 0.7
                detection['confidence_original'] = original_confidence  # Keep original for reference
                logger.debug(f"Applied position bias to center detection: {original_confidence:.3f} -> {detection['confidence']:.3f}")
            
            final_filtered.append(detection)
        
        logger.info(f"Logo candidate filtering complete: {len(detections)} -> {len(final_filtered)} detections")
        
        return final_filtered
        
    except Exception as e:
        logger.error(f"Error in filter_logo_candidates: {e}", exc_info=True)
        # Return empty list on error to prevent pipeline failure
        # This ensures the analysis can continue even if filtering fails
        return []


def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value (0.0 to 1.0)
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def _apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detections with bbox and confidence
        iou_threshold: IoU threshold for considering detections as overlapping
    
    Returns:
        Filtered list with highest-confidence detection kept for each overlap group
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    sorted_detections = sorted(detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
    
    kept = []
    suppressed = set()
    
    for i, detection in enumerate(sorted_detections):
        if i in suppressed:
            continue
        
        bbox1 = detection.get('bbox', [])
        if len(bbox1) < 4:
            continue
        
        kept.append(detection)
        
        # Suppress overlapping detections
        for j in range(i + 1, len(sorted_detections)):
            if j in suppressed:
                continue
            
            bbox2 = sorted_detections[j].get('bbox', [])
            if len(bbox2) < 4:
                continue
            
            iou = _calculate_iou(bbox1, bbox2)
            if iou >= iou_threshold:
                suppressed.add(j)
    
    return kept


def _determine_zone(bbox: List[float], image_width: int, image_height: int) -> str:
    """
    Determine the zone where a detection is located.
    
    Zones: top-left, top-right, bottom-left, bottom-right, center
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_width: Image width
        image_height: Image height
    
    Returns:
        Zone name as string
    """
    if len(bbox) < 4:
        return 'unknown'
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Calculate center point of bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Define center region (middle 40% of image)
    center_threshold_x = 0.3  # 30% from each edge = 40% center
    center_threshold_y = 0.3
    
    left_bound = image_width * center_threshold_x
    right_bound = image_width * (1 - center_threshold_x)
    top_bound = image_height * center_threshold_y
    bottom_bound = image_height * (1 - center_threshold_y)
    
    # Check if in center
    if (left_bound <= center_x <= right_bound and 
        top_bound <= center_y <= bottom_bound):
        return 'center'
    
    # Determine quadrant
    mid_x = image_width / 2
    mid_y = image_height / 2
    
    if center_x < mid_x:
        if center_y < mid_y:
            return 'top-left'
        else:
            return 'bottom-left'
    else:
        if center_y < mid_y:
            return 'top-right'
        else:
            return 'bottom-right'

