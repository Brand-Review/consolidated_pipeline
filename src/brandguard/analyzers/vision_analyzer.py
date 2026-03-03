"""
Vision Signal Analyzer
Analyzes images for logos, colors, and visible text presence using gpt-4o-vision-latest.

Contract (from ANALYZER_CONTRACT.md):
- Logo detection and position
- Color extraction
- Visible text presence (NO OCR, just detection)
- NEVER compute spelling, scores, or compliance judgments
"""

import cv2
import numpy as np
import base64
import io
import os
import logging
import requests
import tempfile
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    Vision Signal Analyzer using gpt-4o-vision-latest via OpenRouter.
    
    Responsibilities:
    - Logo detection and position
    - Color extraction
    - Visible text presence (NO OCR, just detection)
    
    FORBIDDEN:
    - No spelling checking
    - No scores
    - No compliance judgments
    - No OCR text extraction (just detection of visible text)
    """
    
    def __init__(self, api_key: Optional[str] = None, fallback_model: str = "openai/gpt-4o-vision-small"):
        """
        Initialize Vision Analyzer.
        
        Args:
            api_key: OpenRouter API key (if not provided, will try to get from environment)
            fallback_model: Fallback model if primary model is unavailable
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.model_name = "openai/gpt-4o-vision-latest"  # Primary model as per spec
        self.fallback_model = fallback_model
        self.api_endpoint = f"{self.base_url}/chat/completions"
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
    
    def analyze_vision(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze vision signals from image.
        
        Contract (from ANALYZER_CONTRACT.md):
        {
            "status": "passed|failed|skipped|unknown",
            "logos": [LogoSignal],
            "colors": Dict[str, ColorSignal],
            "visibleTextDetected": bool,
            "confidence": float,
            "failure": null | FailureDict
        }
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            Dictionary with vision signals following the contract
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Validate input
            if image is None or not isinstance(image, np.ndarray):
                return {
                    "status": "failed",
                    "logos": [],
                    "colors": {},
                    "visibleTextDetected": False,
                    "confidence": 0.0,
                    "failure": {
                        "reason": "Invalid input: image must be a numpy array",
                        "failure_type": "invalid_input",
                        "recommendations": [
                            "Verify image format is supported",
                            "Check that image is properly loaded"
                        ]
                    },
                    "timestamp": timestamp
                }
            
            # Check if API is available
            if not self.is_available():
                return self._create_fallback_result(timestamp, "api_not_available")
            
            # Convert image to base64
            image_base64 = self._encode_image(image)
            
            # Create vision analysis prompt (signal-only, no scoring)
            prompt = self._create_vision_prompt()
            
            # Call OpenRouter API with image
            try:
                response = self._call_openrouter_api_with_image(prompt, image_base64, self.model_name)
                logger.debug("Vision analyzer: OpenRouter API response received")
                
                # Parse response
                return self._parse_vision_response(response, timestamp)
                
            except requests.exceptions.Timeout:
                logger.warning("Vision analyzer: OpenRouter API request timed out, trying fallback model")
                try:
                    response = self._call_openrouter_api_with_image(prompt, image_base64, self.fallback_model)
                    return self._parse_vision_response(response, timestamp)
                except Exception as e:
                    return self._create_fallback_result(timestamp, "timeout", str(e))
                    
            except requests.exceptions.ConnectionError:
                return self._create_fallback_result(timestamp, "connection_error", "Connection failed - check network")
                
            except requests.exceptions.HTTPError as e:
                error_msg = self._get_http_error_message(e)
                return self._create_fallback_result(timestamp, "http_error", error_msg)
                
            except Exception as e:
                error_msg = f"Vision analysis failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return self._create_fallback_result(timestamp, "unknown_error", error_msg)
                
        except Exception as e:
            logger.error(f"Vision analyzer: Unexpected error: {e}", exc_info=True)
            return self._create_fallback_result(timestamp, "unknown_error", str(e))
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available"""
        if not self.api_key:
            return False
        
        if not self.api_key.strip():
            return False
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _create_vision_prompt(self) -> str:
        """
        Create vision analysis prompt - signal-only, no scoring, no compliance.
        
        Prompt must ONLY extract signals:
        - Logo positions and bounding boxes
        - Dominant colors
        - Visible text presence (NOT extraction, just detection)
        """
        return """Analyze this image and extract ONLY visual signals. Do NOT score, judge compliance, or extract text content.

Extract the following signals:

1. LOGOS: Detect any logos and report their positions as normalized bounding boxes [x, y, width, height] where values are 0.0-1.0 relative to image dimensions. If no logo, return empty array.

2. COLORS: Extract dominant colors as hex codes. Report top 5 dominant colors with coverage percentages.

3. VISIBLE TEXT: Detect if text is visible in the image (yes/no with confidence). Do NOT extract text content - just detect presence.

Return ONLY a valid JSON object with this exact structure:
{
  "logos": [
    {
      "detected": true,
      "bbox": [x_ratio, y_ratio, width_ratio, height_ratio],
      "confidence": 0.0-1.0
    }
  ],
  "colors": {
    "dominant": [
      {"hex": "#HEXCODE", "coverage": 0.0-1.0, "confidence": 0.0-1.0}
    ]
  },
  "visibleTextDetected": true/false,
  "visibleTextConfidence": 0.0-1.0
}

IMPORTANT:
- Do NOT include scoring, compliance judgments, or text extraction
- Do NOT extract actual text content - only detect if text is visible
- If no logo detected, return empty logos array
- If no colors detected, return empty colors array
- Bounding boxes MUST be normalized ratios (0.0-1.0), not pixel coordinates"""
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string for API"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    
    def _call_openrouter_api_with_image(self, prompt: str, image_base64: str, model: str) -> Dict[str, Any]:
        """
        Call OpenRouter API with image.
        
        Args:
            prompt: Text prompt
            image_base64: Base64-encoded image
            model: Model name to use
            
        Returns:
            API response dictionary
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent signal extraction
            "max_tokens": 1000
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/brandguard/consolidated_pipeline",  # Optional for OpenRouter
            "X-Title": "BrandGuard Vision Analyzer"  # Optional for OpenRouter
        }
        
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=headers,
            timeout=(10, 60)  # 10s connect, 60s read
        )
        
        response.raise_for_status()
        return response.json()
    
    def _parse_vision_response(self, response: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Parse OpenRouter API response into vision signals.
        
        Args:
            response: OpenRouter API response
            timestamp: ISO 8601 timestamp
            
        Returns:
            Parsed vision signals following the contract
        """
        try:
            # Extract content from response
            choices = response.get('choices', [])
            if not choices:
                return {
                    "status": "failed",
                    "logos": [],
                    "colors": {},
                    "visibleTextDetected": False,
                    "confidence": 0.0,
                    "failure": {
                        "reason": "API response missing choices",
                        "failure_type": "model_error",
                        "recommendations": ["Check OpenRouter API response format"]
                    },
                    "timestamp": timestamp
                }
            
            content = choices[0].get('message', {}).get('content', '')
            if not content:
                return {
                    "status": "failed",
                    "logos": [],
                    "colors": {},
                    "visibleTextDetected": False,
                    "confidence": 0.0,
                    "failure": {
                        "reason": "API response missing content",
                        "failure_type": "model_error",
                        "recommendations": ["Check OpenRouter API response format"]
                    },
                    "timestamp": timestamp
                }
            
            # Parse JSON from content
            # Try to extract JSON if wrapped in markdown code blocks
            content_clean = content.strip()
            if "```json" in content_clean:
                json_start = content_clean.find("```json") + 7
                json_end = content_clean.find("```", json_start)
                content_clean = content_clean[json_start:json_end].strip()
            elif "```" in content_clean:
                json_start = content_clean.find("```") + 3
                json_end = content_clean.find("```", json_start)
                content_clean = content_clean[json_start:json_end].strip()
            
            try:
                parsed = json.loads(content_clean)
            except json.JSONDecodeError:
                # Try parsing the entire content as JSON
                parsed = json.loads(content)
            
            # Extract signals
            logos = parsed.get('logos', [])
            colors = parsed.get('colors', {})
            visible_text_detected = parsed.get('visibleTextDetected', False)
            visible_text_confidence = parsed.get('visibleTextConfidence', 0.0)
            
            # Format logos
            formatted_logos = []
            for logo in logos:
                if isinstance(logo, dict) and logo.get('detected', False):
                    formatted_logos.append({
                        "detected": True,
                        "bbox": logo.get('bbox', [0, 0, 0, 0]),
                        "confidence": float(logo.get('confidence', 0.0))
                    })
            
            # Format colors
            formatted_colors = {}
            dominant_colors = colors.get('dominant', [])
            if dominant_colors:
                formatted_colors['dominant'] = [
                    {
                        "hex": color.get('hex', '#000000'),
                        "coverage": float(color.get('coverage', 0.0)),
                        "confidence": float(color.get('confidence', 0.0))
                    }
                    for color in dominant_colors[:5]  # Top 5 colors
                ]
            
            # Calculate overall confidence
            confidences = []
            if formatted_logos:
                confidences.extend([logo.get('confidence', 0.0) for logo in formatted_logos])
            if formatted_colors.get('dominant'):
                confidences.extend([color.get('confidence', 0.0) for color in formatted_colors['dominant']])
            if visible_text_detected:
                confidences.append(visible_text_confidence)
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "status": "passed",
                "logos": formatted_logos,
                "colors": formatted_colors,
                "visibleTextDetected": visible_text_detected,
                "confidence": overall_confidence,
                "failure": None,
                "timestamp": timestamp
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Vision analyzer: Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {content[:500] if 'content' in locals() else 'N/A'}")
            return {
                "status": "failed",
                "logos": [],
                "colors": {},
                "visibleTextDetected": False,
                "confidence": 0.0,
                "failure": {
                    "reason": f"Failed to parse API response as JSON: {str(e)}",
                    "failure_type": "parse_error",
                    "recommendations": ["Check OpenRouter API response format", "Verify model output is valid JSON"]
                },
                "timestamp": timestamp
            }
        except Exception as e:
            logger.error(f"Vision analyzer: Failed to parse response: {e}", exc_info=True)
            return {
                "status": "failed",
                "logos": [],
                "colors": {},
                "visibleTextDetected": False,
                "confidence": 0.0,
                "failure": {
                    "reason": f"Failed to parse vision response: {str(e)}",
                    "failure_type": "parse_error",
                    "recommendations": ["Check API response format", "Review logs for details"]
                },
                "timestamp": timestamp
            }
    
    def _create_fallback_result(self, timestamp: str, error_reason: str, error_msg: Optional[str] = None) -> Dict[str, Any]:
        """Create fallback result when API is unavailable"""
        failure = {
            "reason": error_msg or self._get_error_message(error_reason),
            "failure_type": error_reason,
            "recommendations": self._get_error_recommendations(error_reason)
        }
        
        return {
            "status": "failed",
            "logos": [],
            "colors": {},
            "visibleTextDetected": False,
            "confidence": 0.0,
            "failure": failure,
            "timestamp": timestamp
        }
    
    def _get_http_error_message(self, e: requests.exceptions.HTTPError) -> str:
        """Get user-friendly HTTP error message"""
        if e.response:
            if e.response.status_code == 401:
                return "API key invalid or expired (401 Unauthorized)"
            elif e.response.status_code == 429:
                return "Rate limit exceeded (429 Too Many Requests)"
            else:
                return f"HTTP error: {e.response.status_code}"
        return "HTTP error occurred"
    
    def _get_error_message(self, error_reason: str) -> str:
        """Get user-friendly error message"""
        messages = {
            "api_not_available": "OpenRouter API key not configured or invalid",
            "timeout": "Request timed out",
            "connection_error": "Connection failed - check network",
            "http_error": "HTTP error occurred",
            "parse_error": "Failed to parse API response",
            "unknown_error": "Unexpected error occurred"
        }
        return messages.get(error_reason, "Unknown error")
    
    def _get_error_recommendations(self, error_reason: str) -> List[str]:
        """Get actionable recommendations for error"""
        recommendations = {
            "api_not_available": [
                "Set OPENROUTER_API_KEY environment variable",
                "Verify API key is valid at https://openrouter.ai/keys",
                "Check API key has sufficient credits"
            ],
            "timeout": [
                "Check network connectivity",
                "Retry the request",
                "Check OpenRouter API status"
            ],
            "connection_error": [
                "Check network connectivity",
                "Verify firewall settings",
                "Check OpenRouter API status"
            ],
            "http_error": [
                "Check API key is valid",
                "Verify account has credits",
                "Check rate limits"
            ],
            "parse_error": [
                "Check API response format",
                "Verify model output is valid JSON",
                "Review logs for detailed error"
            ],
            "unknown_error": [
                "Review logs for detailed error",
                "Check OpenRouter API status",
                "Verify image format is supported"
            ]
        }
        return recommendations.get(error_reason, ["Check logs for details"])

