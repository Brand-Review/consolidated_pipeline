"""
Logo Detection Module
Uses multiple approaches for logo detection:
1. Fine-tuned YOLO for logo detection
2. VLM-based detection (Qwen2.5-VL) as fallback/enhancement
3. Heuristic fallback
"""

import logging
import numpy as np
import base64
import json
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import cv2, fallback to PIL if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image
    logger.warning("cv2 not available, using PIL for image loading")


class LogoDetector:
    """
    Logo detection using multiple approaches.
    Priority: Fine-tuned YOLO > VLM (Qwen) > Heuristic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LogoDetector.
        
        Args:
            config: Configuration dict with keys:
                - confidence_threshold: Minimum confidence for detections
                - path: Model path for YOLO
                - use_yolo: Whether to use YOLO model
                - use_qwen: Whether to use Qwen VLM for detection
                - qwen_api_url: API URL for Qwen VLM
                - qwen_model: Model name for Qwen
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.model_path = config.get('path', 'yolov8n.pt')
        self.use_yolo = config.get('use_yolo', True)
        self.use_qwen = config.get('use_qwen', True)
        self.qwen_api_url = config.get('qwen_api_url', 'http://localhost:8000/v1/chat/completions')
        self.qwen_model = config.get('qwen_model', 'Qwen/Qwen2.5-VL-3B-Instruct')
        
        self.model = None
        self.model_loaded = False
        self._qwen_available = None
    
    def _load_image(self, image_source):
        """Load image from various sources"""
        if CV2_AVAILABLE:
            if isinstance(image_source, np.ndarray):
                return image_source
            return cv2.imread(image_source)
        else:
            if isinstance(image_source, np.ndarray):
                return np.array(Image.fromarray(image_source).convert('RGB'))
            img = Image.open(image_source)
            return np.array(img.convert('RGB'))
    
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 for VLM API"""
        try:
            # Ensure RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Resize for API (max 1024px)
            h, w = image.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return ""
    
    def _check_qwen_available(self) -> bool:
        """Check if Qwen VLM is available"""
        if self._qwen_available is not None:
            return self._qwen_available
        
        try:
            import requests
            response = requests.get(f"{self.qwen_api_url.rsplit('/v1', 1)[0]}/v1/models", timeout=5)
            self._qwen_available = response.status_code == 200
            if self._qwen_available:
                logger.info("✅ Qwen VLM is available for logo detection")
            return self._qwen_available
        except Exception as e:
            logger.warning(f"Qwen VLM not available: {e}")
            self._qwen_available = False
            return False
    
    def load_model(self) -> bool:
        """Load the logo detection model"""
        loaded = False
        
        # Try YOLO-based logo detection models
        if self.use_yolo:
            # Try fine-tuned logo detection models
            logo_models = [
                'yolov8n.pt',  # Fallback to COCO if no logo model
            ]
            
            for model_name in logo_models:
                try:
                    from ultralytics import YOLO
                    
                    logger.info(f"Trying logo detection model: {model_name}")
                    self.model = YOLO(model_name)
                    self.model_path = model_name
                    self.model_loaded = True
                    logger.info(f"✅ LogoDetector loaded: {model_name}")
                    loaded = True
                    break
                except ImportError as e:
                    logger.warning(f"⚠️ ultralytics not installed: {e}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                    continue
        
        # Check Qwen availability
        if self.use_qwen and not self._check_qwen_available():
            logger.info("⚠️ Qwen VLM not available for logo detection")
        
        if not loaded and not self.use_qwen:
            logger.warning("⚠️ No detection methods available, using heuristic fallback")
        
        return loaded or self.use_qwen
    
    def detect_with_qwen(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect logos using Qwen2.5-VL vision language model.
        This is more accurate for general logo detection.
        """
        if not self._check_qwen_available():
            return []
        
        try:
            import requests
            
            # Encode image
            image_b64 = self._encode_image_base64(image)
            if not image_b64:
                return []
            
            # Prepare prompt for logo detection
            prompt = """Analyze this image and identify ALL logos/brands present.
            
For each logo found, provide:
1. A brief description of what the logo looks like
2. The bounding box coordinates as "x1,y1,x2,y2" (as percentages 0-100)
3. Your confidence that it's actually a logo (not just text or graphic)

Respond in JSON format:
{
  "logos": [
    {
      "description": "red swoosh symbol",
      "bbox_percent": [10, 20, 30, 40],
      "confidence": 0.9,
      "likely_brand": "Nike if this is the famous Nike swoosh"
    }
  ]
}

If no logos are found, respond: {"logos": []}"""

            # Make API call
            payload = {
                "model": self.qwen_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.qwen_api_url.rsplit('/v1', 1)[0]}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.warning(f"Qwen API error: {response.status_code}")
                return []
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            logos = data.get('logos', [])
            
            # Convert percentage bbox to pixel coordinates
            h, w = image.shape[:2]
            detections = []
            
            for i, logo in enumerate(logos):
                bbox_pct = logo.get('bbox_percent', [0, 0, 100, 100])
                x1_pct, y1_pct, x2_pct, y2_pct = bbox_pct
                
                x1 = int(x1_pct / 100 * w)
                y1 = int(y1_pct / 100 * h)
                x2 = int(x2_pct / 100 * w)
                y2 = int(y2_pct / 100 * h)
                
                # Ensure valid bbox
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': logo.get('confidence', 0.7),
                    'class_id': i,
                    'class_name': 'logo',
                    'description': logo.get('description', ''),
                    'likely_brand': logo.get('likely_brand', ''),
                    'bbox_coords': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    },
                    'method': 'qwen_vlm'
                })
            
            logger.info(f"Qwen detected {len(detections)} logo(s)")
            return detections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen response: {e}")
        except Exception as e:
            logger.error(f"Qwen detection failed: {e}")
        
        return []
    
    def _heuristic_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback heuristic logo detection using edge detection and contours.
        Works without any ML models.
        """
        detections = []
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image
            
            # Simple edge detection using numpy
            edges_h = np.abs(np.diff(gray, axis=1))
            edges_v = np.abs(np.diff(gray, axis=0))
            
            # Pad to original size
            edges_h = np.pad(edges_h, ((0, 0), (0, 1)), mode='constant')
            edges_v = np.pad(edges_v, ((0, 1), (0, 0)), mode='constant')
            
            edges = edges_h + edges_v
            
            # Find regions with high edge density (potential logos)
            h, w = edges.shape
            region_size = min(h, w) // 8
            
            for y in range(0, h - region_size, region_size // 2):
                for x in range(0, w - region_size, region_size // 2):
                    region = edges[y:y+region_size, x:x+region_size]
                    edge_density = np.sum(region > 20) / region.size
                    
                    # High edge density but not full of text
                    if 0.05 < edge_density < 0.4:
                        # Check if it's in typical logo positions (corners)
                        is_corner = (x < w // 4 or x > 3 * w // 4) and (y < h // 4 or y > 3 * h // 4)
                        
                        if is_corner or edge_density < 0.25:
                            detections.append({
                                'bbox': [x, y, x + region_size, y + region_size],
                                'confidence': min(0.5, edge_density * 2),
                                'class_id': 0,
                                'class_name': 'potential_logo',
                                'bbox_coords': {
                                    'x1': x, 'y1': y,
                                    'x2': x + region_size, 'y2': y + region_size
                                },
                                'method': 'heuristic'
                            })
            
            logger.info(f"Heuristic detection found {len(detections)} potential logo(s)")
            return detections
            
        except Exception as e:
            logger.error(f"Heuristic detection failed: {e}")
            return []
    
    def detect_logos(self, image) -> List[Dict[str, Any]]:
        """
        Detect logos in image using best available method.
        
        Priority: YOLO > Qwen VLM > Heuristic
        
        Args:
            image: Input image as numpy array, PIL Image, or file path
            
        Returns:
            List of detections with bbox, confidence, class info
        """
        # Load image if needed
        if not isinstance(image, np.ndarray):
            image = self._load_image(image)
        
        if image is None:
            return []
        
        # Method 1: Try Qwen VLM (most accurate for general logos)
        if self.use_qwen and self._check_qwen_available():
            try:
                qwen_detections = self.detect_with_qwen(image)
                if qwen_detections:
                    logger.info(f"Using Qwen VLM detections: {len(qwen_detections)}")
                    return qwen_detections
            except Exception as e:
                logger.warning(f"Qwen detection failed, trying YOLO: {e}")
        
        # Method 2: Try YOLO (COCO - may detect generic objects, not logos)
        if self.model_loaded and self.model is not None:
            try:
                # Convert to RGB
                if CV2_AVAILABLE and len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Run detection
                results = self.model(image_rgb, conf=self.confidence_threshold, verbose=False)
                
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            box = boxes[i]
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names[class_id] if hasattr(result, 'names') else f"object_{class_id}"
                            
                            # Filter to likely logo classes (can customize this)
                            # COCO doesn't have "logo" class, but we can try
                            detections.append({
                                'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                'confidence': conf,
                                'class_id': class_id,
                                'class_name': class_name,
                                'bbox_coords': {
                                    'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                                    'x2': int(xyxy[2]), 'y2': int(xyxy[3])
                                },
                                'method': 'yolo_coco'
                            })
                
                if detections:
                    logger.info(f"YOLO detected {len(detections)} objects (may include logos)")
                    return detections
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Method 3: Fallback to heuristic
        logger.info("Using heuristic logo detection")
        return self._heuristic_detection(image)
    
    def identify_brand_from_crop(self, logo_crop: np.ndarray, hint: str = "") -> Optional[Dict[str, Any]]:
        """
        Identify brand from logo crop using Qwen VLM.
        
        Args:
            logo_crop: Cropped logo image
            hint: Optional hint about the brand name
            
        Returns:
            Dict with brand prediction or None
        """
        if not self._check_qwen_available():
            return None
        
        try:
            import requests
            
            image_b64 = self._encode_image_base64(logo_crop)
            if not image_b64:
                return None
            
            hint_text = f" The brand might be '{hint}'." if hint else ""
            
            prompt = f"""Analyze this logo image and identify the brand name.{hint_text}

Respond in JSON format:
{{
  "brand_name": "Brand Name or Unknown",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

If you cannot identify the brand, respond with "Unknown" as the brand_name."""

            payload = {
                "model": self.qwen_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.qwen_api_url.rsplit('/v1', 1)[0]}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
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
                'brand': data.get('brand_name', 'Unknown'),
                'confidence': data.get('confidence', 0.0),
                'reasoning': data.get('reasoning', '')
            }
            
        except Exception as e:
            logger.warning(f"Brand identification failed: {e}")
            return None


class LogoPlacementValidator:
    """Validate logo placement based on brand rules"""
    
    def __init__(self, brand_rules):
        self.brand_rules = brand_rules
    
    def validate_placement(self, detections: List[Dict], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Validate logo placement.
        
        Args:
            detections: List of logo detections
            image_shape: (height, width)
            
        Returns:
            Validation result dict
        """
        if not detections:
            return {
                'status': 'no_detections',
                'compliance_score': 0,
                'valid': False,
                'violations': []
            }
        
        height, width = image_shape[:2]
        violations = []
        
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Calculate size ratio
            logo_width = x2 - x1
            logo_height = y2 - y1
            size_ratio = (logo_width * logo_height) / (width * height)
            
            # Check size constraints
            if size_ratio < self.brand_rules.min_logo_size:
                violations.append({
                    'type': 'too_small',
                    'detection': det,
                    'message': f"Logo too small: {size_ratio:.2%} < {self.brand_rules.min_logo_size:.2%}"
                })
            elif size_ratio > self.brand_rules.max_logo_size:
                violations.append({
                    'type': 'too_large',
                    'detection': det,
                    'message': f"Logo too large: {size_ratio:.2%} > {self.brand_rules.max_logo_size:.2%}"
                })
            
            # Check edge distance
            left_dist = x1 / width
            right_dist = (width - x2) / width
            top_dist = y1 / height
            bottom_dist = (height - y2) / height
            
            min_edge_dist = min(left_dist, right_dist, top_dist, bottom_dist)
            if min_edge_dist < self.brand_rules.min_edge_distance:
                violations.append({
                    'type': 'too_close_to_edge',
                    'detection': det,
                    'message': f"Logo too close to edge: {min_edge_dist:.2%} < {self.brand_rules.min_edge_distance:.2%}"
                })
        
        # Calculate compliance score
        if detections:
            compliance_score = max(0, 100 - len(violations) * 20)
        else:
            compliance_score = 0
        
        return {
            'status': 'validated',
            'compliance_score': compliance_score,
            'valid': len(violations) == 0,
            'violations': violations,
            'total_detections': len(detections)
        }


def create_logo_detector(config: Dict[str, Any] = None) -> LogoDetector:
    """Factory function to create LogoDetector"""
    if config is None:
        config = {}
    return LogoDetector(config)
