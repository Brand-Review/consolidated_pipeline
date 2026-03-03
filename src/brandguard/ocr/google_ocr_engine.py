"""
Google Cloud Vision OCR Engine
Primary OCR engine using Google Cloud Vision API for high-accuracy text extraction.

Features:
- High accuracy text detection
- Word-level bounding boxes with confidence scores
- Normalized coordinates (0-1)
- Graceful error handling
"""

import cv2
import numpy as np
import logging
import os
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import io

logger = logging.getLogger(__name__)


# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds
TIMEOUT = 30.0  # seconds

# Google Cloud Vision client (lazy initialization)
_vision_client = None


def _get_vision_client():
    """Get or initialize Google Cloud Vision client"""
    global _vision_client
    
    if _vision_client is not None:
        return _vision_client
    
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
        
        # Try to get credentials from settings first
        credentials_path = None
        try:
            from ..config.settings import settings
            credentials_path = settings.google_application_credentials
        except (ImportError, AttributeError):
            pass
        
        # If settings not available, try reading directly from config file
        if not credentials_path:
            try:
                import yaml
                from pathlib import Path
                # Try to find production.yaml config file
                config_paths = [
                    Path('configs/production.yaml'),
                    Path(__file__).parent.parent.parent.parent / 'configs' / 'production.yaml',
                    Path.cwd() / 'configs' / 'production.yaml',
                ]
                
                for config_path in config_paths:
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        # Check both top-level and nested under api_credentials
                        if 'google_application_credentials' in config_data:
                            credentials_path = config_data['google_application_credentials']
                        elif 'api_credentials' in config_data and config_data['api_credentials']:
                            credentials_path = config_data['api_credentials'].get('google_application_credentials')
                        
                        if credentials_path:
                            break
            except Exception:
                pass
        
        # Fallback to environment variable
        if not credentials_path:
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not credentials_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS not set. "
                "Set it in configs/production.yaml (google_application_credentials) "
                "or as environment variable GOOGLE_APPLICATION_CREDENTIALS."
            )
        
        # Expand user path (~) if present
        credentials_path = os.path.expanduser(credentials_path)
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Google credentials file not found: {credentials_path}. "
                "Please check the path in configs/production.yaml or GOOGLE_APPLICATION_CREDENTIALS environment variable."
            )
        
        logger.info(f"[GoogleOCR] Initializing Google Cloud Vision client with credentials: {credentials_path}")
        
        # Create credentials from service account file
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Create client with explicit credentials
        _vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("[GoogleOCR] Google Cloud Vision client initialized successfully")
        return _vision_client
        
    except ImportError:
        raise ImportError(
            "google-cloud-vision not installed. Install with: pip install google-cloud-vision>=3.7.2"
        )
    except Exception as e:
        logger.error(f"[GoogleOCR] Failed to initialize Google Cloud Vision client: {e}")
        raise


def extract_text_google_vision(
    image: Union[np.ndarray, str, bytes],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract text from image using Google Cloud Vision API.
    
    Args:
        image: Image as numpy array (BGR), file path (str), or image bytes
        image_width: Original image width (for normalization, optional)
        image_height: Original image height (for normalization, optional)
        
    Returns:
        Dictionary with normalized OCR result:
        {
            "text": str,  # Full extracted text
            "words": [
                {
                    "text": str,
                    "confidence": float (0-1),
                    "bbox": [x1, y1, x2, y2]  # Normalized 0-1
                }
            ],
            "confidence": float (0-1),  # Average confidence
            "hasText": bool,  # STRICT: True if text was extracted (based on text presence, not word count)
            "method": "GoogleVisionOCR",
            "error": None | Dict  # Error object if failed
        }
    """
    timestamp = datetime.now().isoformat()
    
    try:
        logger.info("[GoogleOCR] Starting Google Cloud Vision OCR...")
        
        # Get Vision client
        client = _get_vision_client()
        
        # Convert image to bytes if needed
        image_bytes = None
        actual_width = image_width
        actual_height = image_height
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to bytes
            # Google Vision expects RGB, so convert BGR to RGB
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Get dimensions if not provided
            if actual_width is None or actual_height is None:
                actual_height, actual_width = rgb_image.shape[:2]
            
            # Encode as PNG
            success, encoded_image = cv2.imencode('.png', rgb_image)
            if not success:
                raise ValueError("Failed to encode image as PNG")
            image_bytes = encoded_image.tobytes()
            
        elif isinstance(image, str):
            # File path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            
            with open(image, 'rb') as f:
                image_bytes = f.read()
            
            # Get dimensions from file if not provided
            if actual_width is None or actual_height is None:
                img = cv2.imread(image)
                if img is not None:
                    actual_height, actual_width = img.shape[:2]
                else:
                    # Fallback: use PIL
                    from PIL import Image
                    with Image.open(image) as pil_img:
                        actual_width, actual_height = pil_img.size
                        
        elif isinstance(image, bytes):
            # Already bytes
            image_bytes = image
            # Try to get dimensions from image bytes
            if actual_width is None or actual_height is None:
                try:
                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    actual_width, actual_height = pil_img.size
                except:
                    logger.warning("[GoogleOCR] Could not determine image dimensions from bytes")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        if image_bytes is None:
            raise ValueError("Failed to convert image to bytes")
        
        if actual_width is None or actual_height is None:
            raise ValueError("Image dimensions not provided and could not be determined")
        
        # Create Vision API image object
        from google.cloud import vision
        from google.api_core import exceptions as google_exceptions
        vision_image = vision.Image(content=image_bytes)
        
        # Perform text detection with retry logic for transient errors
        logger.info("[GoogleOCR] Calling Google Vision API text_detection...")
        
        # Retry logic for transient errors
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                # Configure retry and timeout
                retry = None
                try:
                    from google.api_core import retry as google_retry
                    # Retry on transient errors
                    retry = google_retry.Retry(
                        predicate=google_retry.if_exception_type(
                            google_exceptions.ServiceUnavailable,
                            google_exceptions.DeadlineExceeded,
                            google_exceptions.InternalServerError,
                        ),
                        initial=INITIAL_RETRY_DELAY,
                        maximum=MAX_RETRY_DELAY,
                        multiplier=2.0,
                        timeout=TIMEOUT,
                    )
                except ImportError:
                    # Fallback if retry module not available
                    pass
                
                if retry:
                    response = client.text_detection(image=vision_image, retry=retry, timeout=TIMEOUT)
                else:
                    # Manual retry with exponential backoff
                    response = client.text_detection(image=vision_image, timeout=TIMEOUT)
                
                # Success - break out of retry loop
                break
                
            except (google_exceptions.ServiceUnavailable, 
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.InternalServerError) as e:
                error_type = type(e).__name__
                
                if attempt < MAX_RETRIES - 1:
                    # Calculate exponential backoff delay
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    logger.warning(
                        f"[GoogleOCR] Transient error ({error_type}): {str(e)}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})..."
                    )
                    time.sleep(delay)
                else:
                    # Last attempt failed
                    logger.error(
                        f"[GoogleOCR] Failed after {MAX_RETRIES} attempts. "
                        f"Last error ({error_type}): {str(e)}"
                    )
                    raise
            except Exception as e:
                # Non-retryable error - raise immediately
                logger.error(f"[GoogleOCR] Non-retryable error: {str(e)}")
                raise
        
        # Ensure response was set (should always be true if we reach here)
        if response is None:
            raise RuntimeError("Failed to get response from Google Vision API after retries")
        
        # Check for errors
        if response.error.message:
            error_msg = f"Google Vision API error: {response.error.message}"
            logger.error(f"[GoogleOCR] {error_msg}")
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "hasText": False,  # STRICT: API error means no text extracted
                "method": "GoogleVisionOCR",
                "error": {
                    "type": "api_error",
                    "message": error_msg,
                    "code": response.error.code
                },
                "timestamp": timestamp
            }
        
        # Extract text and annotations
        texts = response.text_annotations
        
        if not texts:
            logger.warning("[GoogleOCR] No text detected in image")
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "hasText": False,  # STRICT: No text detected
                "method": "GoogleVisionOCR",
                "error": None,
                "timestamp": timestamp
            }
        
        # First annotation contains full text
        full_text = texts[0].description if texts else ""
        
        # Remaining annotations are word-level
        words = []
        confidences = []
        
        # Process word-level annotations (skip first which is full text)
        for annotation in texts[1:]:
            text = annotation.description
            
            # Get bounding polygon
            vertices = annotation.bounding_poly.vertices
            if len(vertices) >= 2:
                # Get bounding box from vertices
                x_coords = [v.x for v in vertices if v.x is not None]
                y_coords = [v.y for v in vertices if v.y is not None]
                
                if x_coords and y_coords:
                    # Normalize to 0-1
                    x1 = float(min(x_coords)) / actual_width
                    y1 = float(min(y_coords)) / actual_height
                    x2 = float(max(x_coords)) / actual_width
                    y2 = float(max(y_coords)) / actual_height
                    
                    # Google Vision doesn't provide confidence in text_detection
                    # Use a default high confidence (0.9) for Google Vision
                    confidence = 0.9
                    
                    words.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
                    confidences.append(confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
        
        # STRICT REQUIREMENT: hasText must be based on actual text presence, not word count
        has_text = bool(full_text and full_text.strip())
        word_count = len(words)
        
        logger.info(f"[GoogleOCR] Success: {word_count} words detected, {len(full_text)} characters, hasText={has_text}")
        
        return {
            "text": full_text,
            "words": words,
            "confidence": avg_confidence,
            "hasText": has_text,  # STRICT: Based on text presence, not word count
            "method": "GoogleVisionOCR",
            "error": None,
            "timestamp": timestamp
        }
        
    except ValueError as e:
        # Configuration/input errors
        error_msg = str(e)
        logger.error(f"[GoogleOCR] Configuration error: {error_msg}")
        return {
            "text": "",
            "words": [],
            "confidence": 0.0,
            "hasText": False,  # STRICT: Configuration error means no text extracted
            "method": "GoogleVisionOCR",
            "error": {
                "type": "configuration_error",
                "message": error_msg
            },
            "timestamp": timestamp
        }
    except FileNotFoundError as e:
        # File not found
        error_msg = str(e)
        logger.error(f"[GoogleOCR] File not found: {error_msg}")
        return {
            "text": "",
            "words": [],
            "confidence": 0.0,
            "hasText": False,  # STRICT: File not found means no text extracted
            "method": "GoogleVisionOCR",
            "error": {
                "type": "file_not_found",
                "message": error_msg
            },
            "timestamp": timestamp
        }
    except Exception as e:
        # Check if this is a transient Google API error
        error_type_name = type(e).__name__
        is_transient_error = (
            'ServiceUnavailable' in error_type_name or
            'DeadlineExceeded' in error_type_name or
            'InternalServerError' in error_type_name or
            'sendmsg: Broken pipe' in str(e) or
            '503' in str(e)
        )
        
        if is_transient_error:
            # Transient network/API errors (should have been retried)
            error_msg = str(e)
            logger.error(f"[GoogleOCR] Transient API error after retries ({error_type_name}): {error_msg}", exc_info=True)
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "hasText": False,  # STRICT: API error means no text extracted
                "method": "GoogleVisionOCR",
                "error": {
                    "type": "api_unavailable",
                    "message": f"Google Vision API temporarily unavailable: {error_msg}",
                    "retryable": True
                },
                "timestamp": timestamp
            }
        
        # Other errors (network, auth, etc.)
        error_msg = str(e)
        logger.error(f"[GoogleOCR] Error: {error_msg}", exc_info=True)
        return {
            "text": "",
            "words": [],
            "confidence": 0.0,
            "hasText": False,  # STRICT: Unknown error means no text extracted
            "method": "GoogleVisionOCR",
            "error": {
                "type": "unknown_error",
                "message": error_msg
            },
            "timestamp": timestamp
        }

