"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description:Base Pipeline Orchestrator 
Core functionality for coordinating all BrandGuard models
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .model_imports import import_all_models, get_imported_models, is_models_loaded
from .color_analyzer import ColorAnalyzer
from .logo_analyzer import LogoAnalyzer
from .typography_analyzer import TypographyAnalyzer
from .copywriting_analyzer import CopywritingAnalyzer

logger = logging.getLogger(__name__)

class BasePipelineOrchestrator:
    """
    Base orchestrator that coordinates all BrandGuard models
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.analysis_results = {}
        self.current_analysis_id = None
        
        # Import and initialize models
        self.MODELS_LOADED = import_all_models()
        self.imported_models = get_imported_models()
        
        # Initialize analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize all analyzer components"""
        try:
            # Initialize color analyzer
            self.color_analyzer = ColorAnalyzer(self.settings, self.imported_models)
            
            # Initialize logo analyzer
            self.logo_analyzer = LogoAnalyzer(self.settings, self.imported_models)

            # Initialize typography analyzer
            self.typography_analyzer = TypographyAnalyzer(self.settings, self.imported_models)

            # Initialize copywriting analyzer
            self.copywriting_analyzer = CopywritingAnalyzer(self.settings, self.imported_models)
            
            logger.info("All analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            import traceback
            logger.error(f"Analyzer initialization traceback: {traceback.format_exc()}")
    
    def analyze_content(self, 
                       input_source: str, 
                       source_type: str = 'image',
                       analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method that coordinates all models
        
        Args:
            input_source: Path to file, text content, or URL
            source_type: Type of input ('image', 'document', 'text', 'url')
            analysis_options: Configuration options for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # CRITICAL: Defensive routing - check if input_source is an image URL/file
            # This prevents image URLs from reaching url_analysis even if source_type is wrong
            # MUST happen BEFORE generating analysis ID or initializing results
            if input_source:
                input_lower = str(input_source).lower()
                url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
                image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
                is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                          any(url_path.endswith(ext) for ext in image_extensions)
                
                if is_image and source_type != 'image':
                    # CRITICAL: Force image URLs/files to use image analysis
                    logger.error(
                        f"[Routing] CRITICAL BUG: Image URL/file '{input_source[:100]}' was routed to "
                        f"'{source_type}' but should be 'image'. FORCING correction in base orchestrator!"
                    )
                    source_type = 'image'
            
            # Generate analysis ID
            self.current_analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Initialize results structure
            results = {
                'analysis_id': self.current_analysis_id,
                'timestamp': datetime.now().isoformat(),
                'input_source': input_source,
                'source_type': source_type,  # Use corrected source_type
                'analysis_type': f'{source_type}_analysis',  # Set analysis_type based on corrected source_type
                'model_results': {},
                'overall_compliance': 0.0,
                'summary': '',
                'recommendations': []
            }
            
            logger.info(f"[Routing] Base orchestrator - input_source: '{input_source[:100] if input_source else 'None'}', "
                       f"source_type: '{source_type}', analysis_type: '{results.get('analysis_type')}'")
            
            # Route to appropriate analysis method
            # CRITICAL: Final check before routing - image URLs must NEVER reach url_analysis
            if input_source:
                input_lower = str(input_source).lower()
                url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
                image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
                is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                          any(url_path.endswith(ext) for ext in image_extensions)
                
                if is_image and source_type != 'image':
                    # CRITICAL: This is a double-check - image URLs must use image analysis
                    logger.error(
                        f"[Routing] CRITICAL BUG: Image URL '{input_source[:100]}' reached routing with "
                        f"source_type='{source_type}' instead of 'image'. FORCING correction!"
                    )
                    source_type = 'image'
            
            if source_type == 'image':
                analysis_result = self._analyze_image(input_source, analysis_options)
            elif source_type == 'document':
                analysis_result = self._analyze_document(input_source, analysis_options)
            elif source_type == 'text':
                analysis_result = self._analyze_text(input_source, analysis_options)
            elif source_type == 'url':
                # CRITICAL: This should NEVER happen for image URLs due to multiple defensive checks above
                # If we reach here with an image URL, it's a critical bug
                if input_source:
                    input_lower = str(input_source).lower()
                    url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
                    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
                    is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                              any(url_path.endswith(ext) for ext in image_extensions)
                    
                    if is_image:
                        # CRITICAL: Image URL reached url_analysis - force correction
                        logger.error(
                            f"[Routing] CRITICAL BUG: Image URL '{input_source[:100]}' reached url_analysis! "
                            f"FORCING correction to image_analysis!"
                        )
                        source_type = 'image'
                        analysis_result = self._analyze_image(input_source, analysis_options)
                    else:
                        # Non-image URL - return not_supported
                        analysis_result = self._analyze_url(input_source, analysis_options)
                        if analysis_result.get('status') == 'not_supported':
                            return {
                                'status': 'not_supported',
                                'message': analysis_result.get('message', 'Webpage URL analysis is not yet supported'),
                                'analysis_id': self.current_analysis_id,
                                'timestamp': datetime.now().isoformat()
                            }
                else:
                    analysis_result = self._analyze_url(input_source, analysis_options)
                    if analysis_result.get('status') == 'not_supported':
                        return {
                            'status': 'not_supported',
                            'message': analysis_result.get('message', 'Webpage URL analysis is not yet supported'),
                            'analysis_id': self.current_analysis_id,
                            'timestamp': datetime.now().isoformat()
                        }
            else:
                return {'error': f'Unsupported source type: {source_type}'}
            
            # CRITICAL: Validate that analysis actually ran before merging results
            # Check if analysis_result has actual results (not just placeholder)
            model_results = analysis_result.get('model_results', {})
            if not model_results or (isinstance(model_results, dict) and len(model_results) == 0):
                # No actual analysis ran - check if it's a not_supported response
                if analysis_result.get('status') == 'not_supported':
                    return {
                        'status': 'not_supported',
                        'message': analysis_result.get('message', 'This input type is not yet supported'),
                        'analysis_id': self.current_analysis_id,
                        'timestamp': datetime.now().isoformat()
                    }
                # If it's an error response, return it
                if 'error' in analysis_result:
                    logger.error(f"[Analysis] Analysis returned error: {analysis_result.get('error')}")
                    return analysis_result
                # If model_results is empty but no error, log a warning
                logger.warning(f"[Analysis] Analysis completed but model_results is empty. This may indicate analysis modules failed silently.")
            
            # Normalize analyzer outputs to ensure consistent schema for scoring
            raw_model_results = analysis_result.get('model_results', {})
            normalized_model_results = self._normalize_model_results(raw_model_results)
            
            # Merge results (only if analysis actually ran)
            results.update(analysis_result)
            results['model_results'] = normalized_model_results
            
            # CRITICAL: Ensure analysis_type matches the actual source_type used
            # This prevents returning incorrect analysis_type if routing was corrected
            results['source_type'] = source_type  # Use corrected source_type
            results['analysis_type'] = f'{source_type}_analysis'  # Set based on corrected source_type
            
            logger.info(f"[Routing] Final result - source_type: '{source_type}', analysis_type: '{results['analysis_type']}'")
            
            # FIX: Remove HARD GUARD - always compute compliance from available signals
            # FIX: No early returns - analyzers run independently
            model_results = results.get('model_results', {})
            analysis_options = results.get('analysis_options', {}) or {}
            analysis_mode = analysis_options.get('analysisMode')
            observational_only = analysis_mode == "observational_only"
            
            # FIX: Check if image load failed (only case where we can't proceed)
            if 'error' in results and 'Could not load image' in results.get('error', ''):
                logger.error("[Analysis] Image load failed - cannot proceed with analysis")
                return {
                    'status': 'failed',
                    'message': results.get('error', 'Image load failed'),
                    'analysis_id': self.current_analysis_id,
                    'source_type': source_type,
                    'analysis_type': f'{source_type}_analysis',
                    'timestamp': datetime.now().isoformat(),
                    'critical_signal_failure': True,
                    'overall_compliance': None
                }
            
            # FIX: Calculate overall compliance from available signals
            # FIX: Never return null - compute from available analyzers
            overall_result = self._calculate_overall_compliance(results)
            
            # FIX: Extract compliance and critical failure status
            overall_compliance = overall_result.get('overall_compliance')
            critical_signal_failure = overall_result.get('critical_signal_failure', False)
            
            # FIX: Store results
            results['overall_compliance'] = overall_compliance
            results['critical_signal_failure'] = critical_signal_failure
            results['compliance_breakdown'] = overall_result.get('breakdown', {})
            
            # Store analysis_options in results for schema transformation
            if 'analysis_options' not in results:
                results['analysis_options'] = analysis_options or {}
            
            # Generate summary and recommendations (only if analysis actually ran)
            summary_data = self._generate_summary_and_recommendations(results)
            results.update(summary_data)
            
            # Post-process recommendations (deduplicate, tag sources, remove user-blaming, force null compliance)
            try:
                from .recommendation_postprocessor import RecommendationPostProcessor
                postprocessor = RecommendationPostProcessor()
                analyzer_statuses = postprocessor.extract_analyzer_statuses(results)
                results = postprocessor.process(results, analyzer_statuses)
            except ImportError:
                logger.warning("[Analysis] RecommendationPostProcessor not available, skipping post-processing")
            except Exception as e:
                logger.warning(f"[Analysis] Recommendation post-processing failed: {e}")
            
            # Store results
            self.analysis_results[self.current_analysis_id] = results
            
            # Transform to target schema format if this is an image analysis
            if results.get('analysis_type') == 'image_analysis':
                transformed_results = self._transform_to_target_schema(results)
                return transformed_results
            
            return results
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_image(self, image_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze image content - handles both file paths and URLs"""
        import requests
        from PIL import Image
        import io
        
        image_bytes = None
        image = None
        temp_file = None
        
        try:
            # ====================================================================
            # STEP 1: LOAD IMAGE BYTES (FIRST PLACE IN PIPELINE)
            # ====================================================================
            # CRITICAL: Download image bytes from URL or read from file
            # This is the FIRST place where inputSource (S3 URL) is loaded
            
            if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
                # URL input - download image bytes
                logger.info(f"[ImageLoad] Downloading image from URL: {image_path[:100]}...")
                try:
                    # CRITICAL: Download with timeout=10 as specified
                    response = requests.get(image_path, timeout=10)
                    
                    # CRITICAL: Log HTTP status and response body on failure
                    if response.status_code != 200:
                        response_body = response.text[:500] if response.text else "(empty response body)"
                        error_msg = (
                            f"IMAGE_LOAD_FAILED: HTTP {response.status_code} when downloading image from URL\n"
                            f"URL: {image_path[:200]}\n"
                            f"HTTP Status: {response.status_code}\n"
                            f"Response Body: {response_body}"
                        )
                        logger.error(f"[ImageLoad] {error_msg}")
                        return {
                            'error': 'IMAGE_LOAD_FAILED',
                            'error_type': 'IMAGE_LOAD_FAILED',
                            'message': error_msg,
                            'url': image_path,
                            'http_status': response.status_code,
                            'response_body': response_body,
                            'status': 'failed'
                        }
                    
                    # Verify it's actually an image, not HTML
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type or 'application/xhtml' in content_type:
                        response_body = response.text[:500] if response.text else "(empty response body)"
                        error_msg = (
                            f"IMAGE_LOAD_FAILED: URL points to HTML, not an image\n"
                            f"URL: {image_path[:200]}\n"
                            f"Content-Type: {content_type}\n"
                            f"Response Body: {response_body}"
                        )
                        logger.error(f"[ImageLoad] {error_msg}")
                        return {
                            'error': 'IMAGE_LOAD_FAILED',
                            'error_type': 'IMAGE_LOAD_FAILED',
                            'message': error_msg,
                            'url': image_path,
                            'content_type': content_type,
                            'response_body': response_body,
                            'status': 'failed'
                        }
                    
                    # Get image bytes
                    image_bytes = response.content
                    logger.info(f"[ImageLoad] Image downloaded successfully: {len(image_bytes)} bytes, Content-Type: {content_type}")
                    
                except requests.exceptions.Timeout as e:
                    error_msg = (
                        f"IMAGE_LOAD_FAILED: Timeout (10s) when downloading image from URL\n"
                        f"URL: {image_path[:200]}\n"
                        f"Exception: {str(e)}"
                    )
                    logger.error(f"[ImageLoad] {error_msg}")
                    return {
                        'error': 'IMAGE_LOAD_FAILED',
                        'error_type': 'IMAGE_LOAD_FAILED',
                        'message': error_msg,
                        'url': image_path,
                        'exception': str(e),
                        'status': 'failed'
                    }
                except requests.exceptions.RequestException as e:
                    error_msg = (
                        f"IMAGE_LOAD_FAILED: Request failed when downloading image from URL\n"
                        f"URL: {image_path[:200]}\n"
                        f"Exception: {str(e)}"
                    )
                    logger.error(f"[ImageLoad] {error_msg}")
                    return {
                        'error': 'IMAGE_LOAD_FAILED',
                        'error_type': 'IMAGE_LOAD_FAILED',
                        'message': error_msg,
                        'url': image_path,
                        'exception': str(e),
                        'status': 'failed'
                    }
                except Exception as e:
                    error_msg = (
                        f"IMAGE_LOAD_FAILED: Unexpected error when downloading image from URL\n"
                        f"URL: {image_path[:200]}\n"
                        f"Exception: {str(e)}"
                    )
                    logger.error(f"[ImageLoad] {error_msg}", exc_info=True)
                    return {
                        'error': 'IMAGE_LOAD_FAILED',
                        'error_type': 'IMAGE_LOAD_FAILED',
                        'message': error_msg,
                        'url': image_path,
                        'exception': str(e),
                        'status': 'failed'
                    }
            else:
                # File path input - read image bytes
                logger.info(f"[ImageLoad] Reading image from file: {image_path[:100]}...")
                try:
                    if not os.path.exists(image_path):
                        error_msg = f"IMAGE_LOAD_FAILED: Image file not found: {image_path}"
                        logger.error(f"[ImageLoad] {error_msg}")
                        return {
                            'error': 'IMAGE_LOAD_FAILED',
                            'error_type': 'IMAGE_LOAD_FAILED',
                            'message': error_msg,
                            'file_path': image_path,
                            'status': 'failed'
                        }
                    
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                    logger.info(f"[ImageLoad] Image file read successfully: {len(image_bytes)} bytes")
                    
                except Exception as e:
                    error_msg = (
                        f"IMAGE_LOAD_FAILED: Failed to read image file\n"
                        f"File: {image_path}\n"
                        f"Exception: {str(e)}"
                    )
                    logger.error(f"[ImageLoad] {error_msg}", exc_info=True)
                    return {
                        'error': 'IMAGE_LOAD_FAILED',
                        'error_type': 'IMAGE_LOAD_FAILED',
                        'message': error_msg,
                        'file_path': image_path,
                        'exception': str(e),
                        'status': 'failed'
                    }
            
            # ====================================================================
            # STEP 2: CONVERT BYTES TO IMAGE FORMATS
            # ====================================================================
            # CRITICAL: Convert downloaded bytes to:
            # - PIL Image (for logo, color, typography analyzers)
            # - numpy array (for OpenCV operations)
            # - raw bytes (for Google Vision OCR)
            
            if image_bytes is None or len(image_bytes) == 0:
                error_msg = "IMAGE_LOAD_FAILED: Image bytes are empty or None"
                logger.error(f"[ImageLoad] {error_msg}")
                return {
                    'error': 'IMAGE_LOAD_FAILED',
                    'error_type': 'IMAGE_LOAD_FAILED',
                    'message': error_msg,
                    'status': 'failed'
                }
            
            try:
                # Convert bytes to PIL Image first (validates image format)
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if needed (handles RGBA, P, etc.)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert PIL Image to numpy array (BGR for OpenCV)
                import numpy as np
                rgb_array = np.array(pil_image)
                image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                
                logger.info(f"[ImageLoad] Image converted successfully: PIL mode={pil_image.mode}, numpy shape={image.shape}")
                
            except Exception as e:
                error_msg = (
                    f"IMAGE_LOAD_FAILED: Failed to decode image bytes\n"
                    f"Bytes length: {len(image_bytes) if image_bytes else 0}\n"
                    f"Exception: {str(e)}"
                )
                logger.error(f"[ImageLoad] {error_msg}", exc_info=True)
                return {
                    'error': 'IMAGE_LOAD_FAILED',
                    'error_type': 'IMAGE_LOAD_FAILED',
                    'message': error_msg,
                    'exception': str(e),
                    'status': 'failed'
                }
            
            # ====================================================================
            # STEP 3: IMAGE VALIDATION GUARD
            # ====================================================================
            # CRITICAL: Validate image immediately after load
            # If image loads successfully, it MUST be valid for analysis
            if image is None or image.size == 0:
                error_msg = f"IMAGE_LOAD_FAILED: Image is None or empty (size={image.size if image is not None else 0})"
                logger.error(f"[ImageLoad] {error_msg}")
                return {
                    'error': 'IMAGE_LOAD_FAILED',
                    'error_type': 'IMAGE_LOAD_FAILED',
                    'message': error_msg,
                    'status': 'failed'
                }
            
            if len(image.shape) < 2:
                error_msg = f"IMAGE_LOAD_FAILED: Invalid image shape: {image.shape}"
                logger.error(f"[ImageLoad] {error_msg}")
                return {
                    'error': 'IMAGE_LOAD_FAILED',
                    'error_type': 'IMAGE_LOAD_FAILED',
                    'message': error_msg,
                    'status': 'failed'
                }
            
            if image.shape[0] <= 50 or image.shape[1] <= 50:
                error_msg = f"IMAGE_LOAD_FAILED: Image too small: {image.shape[0]}x{image.shape[1]} (minimum 50x50 required)"
                logger.error(f"[ImageLoad] {error_msg}")
                return {
                    'error': 'IMAGE_LOAD_FAILED',
                    'error_type': 'IMAGE_LOAD_FAILED',
                    'message': error_msg,
                    'status': 'failed'
                }
            
            h, w = image.shape[:2]
            logger.info(f"[ImageLoad] Image loaded and validated successfully: {w}x{h}, shape={image.shape}, dtype={image.dtype}")
            
            # CRITICAL: Store image_bytes for Google OCR
            # Google OCR will receive bytes directly, not URLs or file paths
            # This ensures Google OCR is called ONLY AFTER image bytes are successfully loaded
            # image_bytes is stored in the function scope and will be available to OCR
            
        except Exception as e:
            # CRITICAL: If image loading fails, return structured failure immediately
            # Do NOT allow fallback analyzers to run if image loading fails
            error_msg = (
                f"IMAGE_LOAD_FAILED: Unexpected error during image loading\n"
                f"Input: {image_path[:200] if isinstance(image_path, str) else str(image_path)[:200]}\n"
                f"Exception: {str(e)}"
            )
            logger.error(f"[ImageLoad] {error_msg}", exc_info=True)
            return {
                'error': 'IMAGE_LOAD_FAILED',
                'error_type': 'IMAGE_LOAD_FAILED',
                'message': error_msg,
                'exception': str(e),
                'status': 'failed'
            }
        
        # CRITICAL: Early return if image loading failed
        # Do NOT proceed with analyzers if image is None or bytes are None
        if image is None or image_bytes is None:
            error_msg = "IMAGE_LOAD_FAILED: Image or image_bytes is None after loading"
            logger.error(f"[ImageLoad] {error_msg}")
            return {
                'error': 'IMAGE_LOAD_FAILED',
                'error_type': 'IMAGE_LOAD_FAILED',
                'message': error_msg,
                'status': 'failed'
            }
        
        try:
            
            # ====================================================================
            # STEP 1 — IMAGE LOAD GUARD (MANDATORY)
            # ====================================================================
            # CRITICAL: Validate image immediately after load
            # If image loads successfully, it MUST be valid for analysis
            if image is None or image.size == 0:
                raise RuntimeError(f"Image failed to load: image is None or empty (size={image.size if image is not None else 0})")
            
            if len(image.shape) < 2:
                raise RuntimeError(f"Invalid image shape: {image.shape}")
            
            if image.shape[0] <= 50 or image.shape[1] <= 50:
                raise RuntimeError(f"Image too small: {image.shape[0]}x{image.shape[1]} (minimum 50x50 required)")
            
            h, w = image.shape[:2]
            logger.info(f"[ImageGuard] Image loaded successfully: {w}x{h}, shape={image.shape}, dtype={image.dtype}")
            
            # Clean up temp file if we created one
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"[Routing] Failed to delete temp file: {e}")
            
            # Initialize results - FIX: All analyzers run independently, no dependencies
            model_results = {}
            
            # CRITICAL: Initialize ocr_result early to avoid UnboundLocalError
            # Logo detection may reference it before OCR runs
            ocr_result = None
            ocr_text = None
            ocr_words = []
            
            # ====================================================================
            # STAGE 1: HARD SIGNALS - COLOR EXTRACTION (LAYER 1: NON-LLM)
            # FIX: Use HardSignalExtractor for colors (KMeans, always works)
            # FIX: Always returns colors if image loads
            # ====================================================================
            # CRITICAL: Validate image before color extraction
            try:
                assert image is not None, "CRITICAL: Image is None before color extraction"
                assert len(image.shape) >= 2, f"CRITICAL: Invalid image shape: {image.shape if image is not None else 'None'}"
                assert image.shape[0] > 100, f"CRITICAL: Image height too small: {image.shape[0]} (minimum 100 required)"
                assert image.shape[1] > 100, f"CRITICAL: Image width too small: {image.shape[1]} (minimum 100 required)"
            except AssertionError as e:
                error_msg = f"[CRITICAL] Image validation failed before color extraction: {str(e)}"
                logger.critical(error_msg)
                raise ValueError(error_msg) from e
            
            try:
                if not analysis_options or analysis_options.get('color_analysis', {}).get('enabled', True):
                    logger.info("[Analysis] Running hard signal color extraction (non-LLM)...")
                    
                    # FIX: Use HardSignalExtractor for color extraction
                    try:
                        from ..signals.hard_signal_extractor import HardSignalExtractor
                        signal_extractor = HardSignalExtractor()
                        color_signals = signal_extractor.extract_color_signals(image)
                        
                        color_colors = color_signals.get('colors', [])
                        color_status = color_signals.get('status', 'failed')
                        color_raw_signals = color_signals.get('rawSignalsPresent', False)
                        
                        # CRITICAL: If colors array is empty but rawSignalsPresent=True, extract colors directly
                        if not color_colors and color_raw_signals:
                            logger.warning("[Analysis] Color signals present but colors array empty - extracting directly")
                            # Fallback: extract colors directly from image
                            h, w = image.shape[:2]
                            positions = [
                                (h//4, w//4), (h//4, w//2), (h//4, 3*w//4),
                                (h//2, w//4), (h//2, w//2), (h//2, 3*w//4),
                                (3*h//4, w//4), (3*h//4, w//2), (3*h//4, 3*w//4)
                            ]
                            for y, x in positions[:8]:
                                if 0 <= y < h and 0 <= x < w:
                                    if len(image.shape) == 3:
                                        b, g, r = image[y, x]
                                    else:
                                        r = g = b = image[y, x]
                                    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                                    color_colors.append({
                                        "hex": hex_color,
                                        "rgb": [int(r), int(g), int(b)],
                                        "percent": 11.11
                                    })
                            color_status = "observed"  # Update status
                        
                        # CRITICAL: Ensure colors array is never empty
                        if not color_colors:
                            logger.error("[Analysis] CRITICAL: Color extraction returned empty array - using emergency fallback")
                            # Emergency fallback: sample center pixel
                            h, w = image.shape[:2]
                            center_y, center_x = h // 2, w // 2
                            if len(image.shape) == 3:
                                b, g, r = image[center_y, center_x]
                            else:
                                r = g = b = image[center_y, center_x]
                            hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                            color_colors = [{
                                "hex": hex_color,
                                "rgb": [int(r), int(g), int(b)],
                                "percent": 100.0
                            }]
                            color_status = "observed"
                        
                        # FIX: Convert to format expected by color analyzer
                        dominant_colors_hex = [c.get('hex', '') for c in color_colors if c.get('hex')]
                        
                        # FIX: Get brand validation from color analyzer if brand palette provided
                        color_options = (analysis_options or {}).get('color_analysis', {})
                        brand_palette = color_options.get('brand_palette', '')
                        has_brand_palette = bool(brand_palette and brand_palette.strip())
                        
                        # FIX: Map HardSignalExtractor status to contract: observed -> observed, fail -> fail, missing -> missing
                        if color_status == "observed" and color_colors:
                            # FIX: Colors extracted - validate against brand palette if provided
                            if has_brand_palette:
                                # FIX: Validate colors against brand palette (soft reasoning - can use LLM)
                                try:
                                    brand_validation = self.color_analyzer._validate_colors_against_palette_real(
                                        [{'hex': hex_color, 'rgb': next((c['rgb'] for c in color_colors if c.get('hex') == hex_color), None)} for hex_color in dominant_colors_hex],
                                        brand_palette,
                                        color_options.get('color_tolerance', 2.3)
                                    )
                                    
                                    # Determine status based on validation
                                    if brand_validation.get('compliance_score', 0) >= 0.8:
                                        color_analyzer_status = "pass"  # Colors match brand
                                    elif brand_validation.get('compliance_score', 0) >= 0.5:
                                        color_analyzer_status = "observed"  # Colors partially match
                                    else:
                                        color_analyzer_status = "fail"  # Colors don't match
                                    
                                    color_result = {
                                        'status': color_analyzer_status,  # FIX: Use contract status
                                        'dominant_colors': dominant_colors_hex,
                                        'colors': color_colors,  # FIX: Include detailed color info
                                        'brand_validation': brand_validation,
                                        'analyzerStatus': color_analyzer_status,
                                        'confidence': color_signals.get('confidence', 0.85),
                                        'rawSignalPresent': True,
                                        'reason': brand_validation.get('reason')
                                    }
                                except Exception as e:
                                    logger.warning(f"[Analysis] Brand validation failed: {e}, using observed status")
                                    color_result = {
                                        'status': 'observed',  # FIX: Use observed when validation fails
                                        'dominant_colors': dominant_colors_hex,
                                        'colors': color_colors,
                                        'brand_validation': {
                                            'status': 'error',
                                            'reason': f'Brand validation error: {str(e)}',
                                            'compliance_score': None
                                        },
                                        'analyzerStatus': 'observed',
                                        'confidence': color_signals.get('confidence', 0.85),
                                        'rawSignalPresent': True,
                                        'reason': f'Colors extracted but validation failed: {str(e)}'
                                    }
                            else:
                                # FIX: No brand palette - observed only
                                color_result = {
                                    'status': 'observed',  # FIX: Use "observed" (STRICT CONTRACT)
                                    'dominant_colors': dominant_colors_hex,
                                    'colors': color_colors,
                                    'brand_validation': {
                                        'status': 'missing',
                                        'reason': 'Brand color palette not provided - cannot validate compliance',
                                        'compliance_score': None
                                    },
                                    'analyzerStatus': 'observed',
                                    'confidence': color_signals.get('confidence', 0.85),
                                    'rawSignalPresent': True,
                                    'reason': 'Colors extracted but no brand palette provided'
                                }
                        elif color_status == "fail":
                            # CRITICAL: Even on failure, extract colors directly (NEVER return empty)
                            logger.warning("[Analysis] Color extraction failed - using emergency fallback")
                            h, w = image.shape[:2]
                            emergency_colors = []
                            positions = [
                                (h//4, w//4), (h//4, w//2), (h//4, 3*w//4),
                                (h//2, w//4), (h//2, w//2), (h//2, 3*w//4),
                                (3*h//4, w//4), (3*h//4, w//2)
                            ]
                            for y, x in positions:
                                if 0 <= y < h and 0 <= x < w:
                                    if len(image.shape) == 3:
                                        b, g, r = image[y, x]
                                    else:
                                        r = g = b = image[y, x]
                                    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                                    emergency_colors.append({
                                        "hex": hex_color,
                                        "rgb": [int(r), int(g), int(b)],
                                        "percent": 12.5
                                    })
                            
                            emergency_hex = [c.get('hex', '') for c in emergency_colors if c.get('hex')]
                            
                            color_result = {
                                'status': 'observed',  # NOT "fail" - colors extracted via fallback
                                'dominant_colors': emergency_hex,  # CRITICAL: Never empty
                                'colors': emergency_colors,  # CRITICAL: Never empty
                                'brand_validation': {
                                    'status': 'error',
                                    'reason': 'Color extraction failed, using emergency fallback',
                                    'compliance_score': None
                                },
                                'analyzerStatus': 'observed',  # NOT "fail"
                                'confidence': 0.5,  # Lower confidence for fallback
                                'rawSignalPresent': True,  # CRITICAL: Colors exist
                                'reason': color_signals.get('reason', 'Color extraction failed but colors extracted via emergency fallback')
                            }
                        else:  # missing status
                            # CRITICAL: Even if "missing", extract colors directly (NEVER return empty)
                            logger.warning("[Analysis] Color status missing - using emergency fallback")
                            h, w = image.shape[:2]
                            emergency_colors = []
                            positions = [
                                (h//4, w//4), (h//4, w//2), (h//4, 3*w//4),
                                (h//2, w//4), (h//2, w//2), (h//2, 3*w//4),
                                (3*h//4, w//4), (3*h//4, w//2)
                            ]
                            for y, x in positions:
                                if 0 <= y < h and 0 <= x < w:
                                    if len(image.shape) == 3:
                                        b, g, r = image[y, x]
                                    else:
                                        r = g = b = image[y, x]
                                    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                                    emergency_colors.append({
                                        "hex": hex_color,
                                        "rgb": [int(r), int(g), int(b)],
                                        "percent": 12.5
                                    })
                            
                            emergency_hex = [c.get('hex', '') for c in emergency_colors if c.get('hex')]
                            
                            color_result = {
                                'status': 'observed',  # NOT "missing" - colors extracted
                                'dominant_colors': emergency_hex,  # CRITICAL: Never empty
                                'colors': emergency_colors,  # CRITICAL: Never empty
                                'brand_validation': {
                                    'status': 'missing',
                                    'reason': 'No brand palette provided',
                                    'compliance_score': None
                                },
                                'analyzerStatus': 'observed',  # NOT "missing"
                                'confidence': 0.7,  # Lower confidence for fallback
                                'rawSignalPresent': True,  # CRITICAL: Colors exist
                                'reason': color_signals.get('reason', 'Colors extracted via emergency fallback')
                            }
                    except ImportError:
                        # FIX: Fallback to color analyzer if HardSignalExtractor not available
                        logger.warning("[Analysis] HardSignalExtractor not available, using color analyzer fallback")
                        color_options = (analysis_options or {}).get('color_analysis', {})
                        color_result = self.color_analyzer.analyze_colors(image, color_options)
                        dominant_colors_hex = color_result.get('dominant_colors', [])
                        
                        # FIX: Ensure colors are always returned
                        if not dominant_colors_hex:
                            dominant_colors_hex = self._extract_colors_fallback(image, min_colors=3)
                            if dominant_colors_hex:
                                color_result['dominant_colors'] = dominant_colors_hex
                                color_result['fallback_used'] = True
                    
                    # FIX: Store raw output even on failure - never empty object
                    color_result['analyzerStatus'] = color_result.get('analyzerStatus', color_result.get('status', 'passed'))
                    color_result['confidence'] = color_result.get('confidence', 0.85)
                    color_result['rawSignalPresent'] = len(color_result.get('dominant_colors', [])) > 0
                    # 🔥 Add visibilityState to raw output
                    color_result['visibilityState'] = self._get_visibility_state(color_result['analyzerStatus'], analyzer_type="color")
                    
                    model_results['color_analysis'] = color_result
                    logger.info(f"[Analysis] Color extraction completed: {len(color_result.get('dominant_colors', []))} colors, status={color_result.get('analyzerStatus')}")
                else:
                    logger.info("[Analysis] Color analysis disabled")
                    model_results['color_analysis'] = {
                        'status': 'skipped',
                        'reason': 'Color analysis disabled in options',
                        'dominant_colors': [],
                        'analyzerStatus': 'skipped',
                        'confidence': 0.0,
                        'rawSignalPresent': False
                    }
            except Exception as e:
                logger.error(f"[Analysis] Color extraction failed: {e}", exc_info=True)
                # FIX: Store failure output, don't skip - try last resort extraction
                try:
                    last_resort_colors = self._extract_colors_fallback(image, min_colors=1)
                    model_results['color_analysis'] = {
                        'status': 'failed',
                        'reason': f'Color extraction error: {str(e)}',
                        'errors': [str(e)],
                        'dominant_colors': last_resort_colors if last_resort_colors else [],
                        'analyzerStatus': 'failed',
                        'confidence': 0.5 if last_resort_colors else 0.0,
                        'rawSignalPresent': bool(last_resort_colors)
                    }
                except:
                    model_results['color_analysis'] = {
                        'status': 'failed',
                        'reason': f'Color extraction error: {str(e)}',
                        'errors': [str(e)],
                        'dominant_colors': [],
                        'analyzerStatus': 'failed',
                        'confidence': 0.0,
                        'rawSignalPresent': False
                    }
            
            # ====================================================================
            # STAGE 2: HARD SIGNALS - LOGO DETECTION (LAYER 1: NON-LLM)
            # FIX: Use HardSignalExtractor for logo detection (heuristic + object detection)
            # FIX: Returns normalized bboxes (0-1), zones, size ratios
            # ====================================================================
            # CRITICAL: Validate image before logo detection
            try:
                assert image is not None, "CRITICAL: Image is None before logo detection"
                assert len(image.shape) >= 2, f"CRITICAL: Invalid image shape: {image.shape if image is not None else 'None'}"
                assert image.shape[0] > 100, f"CRITICAL: Image height too small: {image.shape[0]} (minimum 100 required)"
                assert image.shape[1] > 100, f"CRITICAL: Image width too small: {image.shape[1]} (minimum 100 required)"
            except AssertionError as e:
                error_msg = f"[CRITICAL] Image validation failed before logo detection: {str(e)}"
                logger.critical(error_msg)
                raise ValueError(error_msg) from e
            
            try:
                if not analysis_options or analysis_options.get('logo_analysis', {}).get('enabled', True):
                    logger.info("[Analysis] Running hard signal logo detection (non-LLM)...")
                    h, w = image.shape[:2]
                    
                    # STRICT REQUIREMENT: If OCR.hasText = true, infer logo region from brand name text
                    ocr_has_text = ocr_result.get('hasText', False) if ocr_result else False
                    ocr_words_list = ocr_result.get('words', []) if ocr_result else []
                    
                    # FIX: Use HardSignalExtractor for logo detection
                    try:
                        from ..signals.hard_signal_extractor import HardSignalExtractor
                        signal_extractor = HardSignalExtractor()
                        logo_signals = signal_extractor.extract_logo_signals(image, w, h)
                        
                        # STRICT: If OCR.hasText = true, infer logo region from brand name text
                        if ocr_has_text and ocr_words_list:
                            # Try to find brand name in OCR text and infer logo position
                            brand_name = analysis_options.get('brand_name') if analysis_options else None
                            if brand_name:
                                # Find brand name in OCR words
                                brand_word_bboxes = []
                                for word_info in ocr_words_list:
                                    word_text = word_info.get('word', '').lower()
                                    if brand_name.lower() in word_text or word_text in brand_name.lower():
                                        bbox = word_info.get('bbox', [])
                                        if len(bbox) == 4:
                                            brand_word_bboxes.append(bbox)
                                
                                if brand_word_bboxes:
                                    # Infer logo region from brand name text position
                                    # Logo is typically near brand name text
                                    first_bbox = brand_word_bboxes[0]
                                    x1, y1, x2, y2 = first_bbox
                                    
                                    # Infer logo bbox: slightly above and to the left of brand text
                                    logo_bbox = [
                                        max(0.0, x1 - 0.1),  # Left of text
                                        max(0.0, y1 - 0.15),  # Above text
                                        x1,  # Right edge at text start
                                        y1  # Bottom edge at text top
                                    ]
                                    
                                    # Determine zone from normalized bbox
                                    center_x = (logo_bbox[0] + logo_bbox[2]) / 2
                                    center_y = (logo_bbox[1] + logo_bbox[3]) / 2
                                    
                                    # STRICT: Validate placement using normalized bbox zones
                                    zone = None
                                    if center_y < 0.33:  # Top third
                                        if center_x < 0.33:
                                            zone = "top-left"
                                        elif center_x < 0.66:
                                            zone = "top-center"
                                        else:
                                            zone = "top-right"
                                    elif center_y < 0.66:  # Middle third
                                        if center_x < 0.33:
                                            zone = "middle-left"
                                        elif center_x < 0.66:
                                            zone = "center"
                                        else:
                                            zone = "middle-right"
                                    else:  # Bottom third
                                        if center_x < 0.33:
                                            zone = "bottom-left"
                                        elif center_x < 0.66:
                                            zone = "bottom-center"
                                        else:
                                            zone = "bottom-right"
                                    
                                    # Add inferred logo detection
                                    inferred_detection = {
                                        'bbox': logo_bbox,
                                        'confidence': 0.7,  # Medium confidence for inferred
                                        'zone': zone,
                                        'sizeRatio': (logo_bbox[2] - logo_bbox[0]) * (logo_bbox[3] - logo_bbox[1]),
                                        'method': 'inferred_from_text',
                                        'reason': f'Logo inferred from brand name "{brand_name}" text position'
                                    }
                                    
                                    # Add to existing detections or create new list
                                    existing_detections = logo_signals.get('detections', [])
                                    if not existing_detections:
                                        # No object detection found, use inferred
                                        logo_signals['detections'] = [inferred_detection]
                                        logo_signals['detected'] = True
                                        logo_signals['status'] = 'observed'
                                        logo_signals['rawSignalsPresent'] = True
                                        logger.info(f"[Analysis] Logo inferred from brand name text at {zone} zone")
                                    else:
                                        # Merge with existing detections
                                        logo_signals['detections'].append(inferred_detection)
                                        logger.info(f"[Analysis] Added inferred logo detection to existing detections")
                        
                        logo_detected = logo_signals.get('detected', False)
                        logo_detections = logo_signals.get('detections', [])
                        logo_status = logo_signals.get('status', 'failed')
                        logo_raw_signals = logo_signals.get('rawSignalsPresent', False)
                        
                        # CRITICAL: NEVER erase signals - if low_confidence or rawSignalsPresent, logo exists
                        # Map HardSignalExtractor status - preserve signals
                        if logo_status == "observed" and logo_detected:
                            analyzer_status = "observed"  # Signals detected
                        elif logo_status == "low_confidence" or (logo_raw_signals and logo_detections):
                            # CRITICAL: Logo-like regions exist - set detected=True
                            analyzer_status = "observed"  # NOT "missing" - signals exist
                            logo_detected = True  # CRITICAL: Override to True if signals exist
                            logger.warning(f"[Analysis] Logo low_confidence but signals present - setting detected=True")
                        elif logo_status == "missing":
                            analyzer_status = "missing"  # Only if truly no signals
                        elif logo_status == "fail":
                            # Check if signals exist despite error
                            if logo_raw_signals or logo_detections:
                                analyzer_status = "observed"  # Signals exist despite error
                                logo_detected = True
                            else:
                                analyzer_status = "fail"  # Error occurred, no signals
                        else:
                            # Unknown status - check rawSignalsPresent
                            if logo_raw_signals or logo_detections:
                                analyzer_status = "observed"
                                logo_detected = True
                            else:
                                analyzer_status = "missing"
                        
                        logo_result = {
                            'status': analyzer_status,
                            'detected': logo_detected,  # CRITICAL: True if signals exist
                            'detections': logo_detections,
                            'logo_detections': logo_detections,
                            'analyzerStatus': analyzer_status,
                            'confidence': logo_detections[0].get('confidence', 0.85) if logo_detections else 0.0,
                            'rawSignalPresent': logo_detected or logo_raw_signals,  # CRITICAL: Preserve signal indicator
                            'reason': logo_signals.get('reason'),
                            'zone': logo_detections[0].get('zone') if logo_detections else None
                        }
                        
                        # FIX: Add placement validation if logo detected
                        if logo_detected and logo_detections:
                            from ..logo.logo_zones import get_zone_name
                            from ..logo.logo_geometry import normalize_bbox, calculate_size_ratio
                            
                            placement_violations = []
                            for det in logo_detections:
                                bbox = det.get('bbox', [])
                                if len(bbox) == 4:
                                    # Check if logo is in center (violation)
                                    center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                                    center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
                                    
                                    if 0.33 <= center_x <= 0.66 and 0.33 <= center_y <= 0.66:
                                        placement_violations.append({
                                            'type': 'center_placement',
                                            'reason': 'Logo placed in center zone (not recommended)',
                                            'severity': 'medium',
                                            'confidence': det.get('confidence', 0.85)
                                        })
                                    
                                    # Check size violations
                                    size_ratio = det.get('sizeRatio', 0)
                                    if size_ratio < 0.01:
                                        placement_violations.append({
                                            'type': 'size_too_small',
                                            'reason': f'Logo too small (size ratio: {size_ratio:.4f})',
                                            'severity': 'high',
                                            'confidence': det.get('confidence', 0.85)
                                        })
                                    elif size_ratio > 0.25:
                                        placement_violations.append({
                                            'type': 'size_too_large',
                                            'reason': f'Logo too large (size ratio: {size_ratio:.4f})',
                                            'severity': 'medium',
                                            'confidence': det.get('confidence', 0.85)
                                        })
                            
                            # V1 RULE: Logo compliance score must ALWAYS be None (observational only)
                            logo_result['placement_validation'] = {
                                'status': 'not_applicable',  # V1: Always not_applicable - logo is observational only
                                'violations': placement_violations,
                                'compliance_score': None,  # V1: Always None - logo never affects compliance score
                                'reason': 'Logo detection is observational only in V1. Reference logo required for placement validation.'
                            }
                        else:
                            logo_result['placement_validation'] = {
                                'status': 'not_applicable',
                                'violations': [],
                                'compliance_score': None
                            }
                    except ImportError:
                        # FIX: Fallback to logo analyzer if HardSignalExtractor not available
                        logger.warning("[Analysis] HardSignalExtractor not available, using logo analyzer fallback")
                        logo_options = (analysis_options or {}).get('logo_analysis', {})
                        if 'ocr_text' in logo_options:
                            del logo_options['ocr_text']
                        if 'extracted_text' in logo_options:
                            del logo_options['extracted_text']
                        
                        brand_name = analysis_options.get('brand_name') if analysis_options else None
                        if brand_name:
                            logo_options['brand_name'] = brand_name
                        
                        logo_result = self.logo_analyzer.analyze_logos(image, logo_options)
                        
                        # FIX: Normalize detections
                        detections = logo_result.get('logo_detections', [])
                        logo_detected = len(detections) > 0
                        logo_result['detected'] = logo_detected
                        
                        # FIX: Normalize bboxes and add zones
                        normalized_detections = []
                        for det in detections:
                            if isinstance(det, dict):
                                bbox = det.get('bbox', det.get('bounding_box', []))
                                if len(bbox) == 4:
                                    # Normalize if not already normalized
                                    if bbox[0] > 1.0 or bbox[1] > 1.0:
                                        bbox = [
                                            bbox[0] / w, bbox[1] / h,
                                            bbox[2] / w, bbox[3] / h
                                        ]
                                    
                                    # Calculate zone
                                    center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                                    center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
                                    from ..logo.logo_zones import get_zone_name
                                    zone = get_zone_name(center_x, center_y)
                                    
                                    # Calculate size ratio
                                    size_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    
                                    normalized_detections.append({
                                        'bbox': bbox,
                                        'confidence': float(det.get('confidence', det.get('score', 0.5))),
                                        'zone': zone,
                                        'sizeRatio': size_ratio
                                    })
                        
                        logo_result['detections'] = normalized_detections
                    
                    # FIX: Store raw output with analyzerStatus and visibilityState
                    logo_result['analyzerStatus'] = logo_result.get('analyzerStatus', 'passed' if logo_result.get('detected') else 'not_detected')
                    logo_result['confidence'] = logo_result.get('confidence', 0.85 if logo_result.get('detected') else 0.0)
                    logo_result['rawSignalPresent'] = logo_result.get('detected', False)
                    # 🔥 Add visibilityState to raw output
                    logo_result['visibilityState'] = self._get_visibility_state(logo_result['analyzerStatus'], analyzer_type="logo")
                    
                    model_results['logo_analysis'] = logo_result
                    logger.info(f"[Analysis] Logo detection completed: detected={logo_result.get('detected', False)}, status={logo_result.get('analyzerStatus')}")
                else:
                    logger.info("[Analysis] Logo analysis disabled")
                    model_results['logo_analysis'] = {
                        'status': 'skipped',
                        'reason': 'Logo analysis disabled in options',
                        'detected': False,
                        'detections': [],
                        'analyzerStatus': 'skipped',
                        'confidence': 0.0,
                        'rawSignalPresent': False
                    }
            except Exception as e:
                logger.error(f"[Analysis] Logo detection failed: {e}", exc_info=True)
                # FIX: Store failure output, don't skip
                model_results['logo_analysis'] = {
                    'status': 'failed',
                    'reason': f'Logo detection error: {str(e)}',
                    'errors': [str(e)],
                    'detected': False,
                    'detections': [],
                    'analyzerStatus': 'failed',
                    'confidence': 0.0,
                    'rawSignalPresent': False
                }
            
            # ====================================================================
            # STAGE 3: HARD SIGNALS - OCR (LAYER 1: NON-LLM)
            # FIX: Use HardSignalExtractor for OCR (PaddleOCR/EasyOCR/Tesseract only)
            # FIX: Returns word-level bounding boxes and confidence per word
            # FIX: OCR failure only affects copywriting, never blocks other analyzers
            # ====================================================================
            # CRITICAL: Validate image before OCR
            try:
                assert image is not None, "CRITICAL: Image is None before OCR"
                assert len(image.shape) >= 2, f"CRITICAL: Invalid image shape: {image.shape if image is not None else 'None'}"
                assert image.shape[0] > 100, f"CRITICAL: Image height too small: {image.shape[0]} (minimum 100 required)"
                assert image.shape[1] > 100, f"CRITICAL: Image width too small: {image.shape[1]} (minimum 100 required)"
            except AssertionError as e:
                error_msg = f"[CRITICAL] Image validation failed before OCR: {str(e)}"
                logger.critical(error_msg)
                raise ValueError(error_msg) from e
            
            # ocr_result already initialized above to avoid UnboundLocalError
            # Only reset ocr_text and ocr_words if ocr_result is still None
            if ocr_result is None:
                ocr_text = None
                ocr_words = []
            try:
                logger.info("[Analysis] Running hard signal OCR (non-LLM)...")
                
                # FIX: Use HardSignalExtractor for OCR (no LLMs)
                try:
                    from ..signals.hard_signal_extractor import HardSignalExtractor
                    h, w = image.shape[:2]
                    signal_extractor = HardSignalExtractor()
                    ocr_signals = signal_extractor.extract_ocr_signals(image, w, h)
                    
                    ocr_text = ocr_signals.get('text', '')
                    ocr_words = ocr_signals.get('words', [])
                    ocr_status = ocr_signals.get('status', 'failed')
                    ocr_confidence = ocr_signals.get('confidence', 0.0)
                    
                    # CRITICAL: Handle status values - NEVER erase signals
                    # Check rawSignalsPresent FIRST - if True, signal exists regardless of text
                    raw_signals_present = ocr_signals.get('rawSignalsPresent', False)
                    
                    if ocr_status == "partial" or (ocr_status == "low_confidence" and raw_signals_present):
                        # CRITICAL: Partial or low_confidence with signals = text exists (even if partial)
                        logger.warning(f"[Analysis] OCR partial/low_confidence ({ocr_confidence:.2f}) but signals present")
                        ocr_result = {
                            'hasText': bool(ocr_text and ocr_text.strip()),  # True if any text extracted
                            'text': ocr_text if ocr_text else "[Text Detected]",  # CRITICAL: Never empty if signals exist
                            'confidence': ocr_confidence,
                            'failure': None,
                            'status': 'partial',  # Use "partial" not "low_confidence" or "fail"
                            'words': ocr_words,
                            'rawSignalPresent': True  # CRITICAL: Preserve signal indicator
                        }
                    elif ocr_status == "pass":
                        # STRICT: Use hasText from OCR signals
                        ocr_has_text = ocr_signals.get('hasText', bool(ocr_text and ocr_text.strip()))
                        ocr_result = {
                            'hasText': ocr_has_text,  # STRICT: Use hasText from OCR signals
                            'text': ocr_text,
                            'confidence': ocr_confidence,
                            'failure': None,
                            'status': 'pass',
                            'words': ocr_words,
                            'rawSignalPresent': True
                        }
                    elif ocr_status == "observed":
                        # STRICT: Use hasText from OCR signals
                        ocr_has_text = ocr_signals.get('hasText', bool(ocr_text and ocr_text.strip()))
                        ocr_result = {
                            'hasText': ocr_has_text,  # STRICT: Use hasText from OCR signals
                            'text': ocr_text,
                            'confidence': ocr_confidence,
                            'failure': None,
                            'status': 'observed',
                            'words': ocr_words,
                            'rawSignalPresent': True
                        }
                    elif ocr_status == "missing":
                        # Only "missing" if truly no signals
                        # STRICT: Use hasText from OCR signals
                        ocr_has_text = ocr_signals.get('hasText', False)
                        ocr_result = {
                            'hasText': ocr_has_text,  # STRICT: Use hasText from OCR signals
                            'text': '',
                            'confidence': 0.0,
                            'failure': {
                                'error': 'OCR_FAILED',
                                'error_type': 'OCR_FAILED',
                                'reason': ocr_signals.get('reason', 'No text detected'),
                                'message': f"OCR_FAILED: {ocr_signals.get('reason', 'No text detected')}",
                                'status': 'failed'
                            },
                            'status': 'missing',
                            'words': [],
                            'rawSignalPresent': False
                        }
                    else:  # "fail" or unknown status
                        # CRITICAL: Check rawSignalsPresent - if True, convert to "partial"
                        # STRICT: Use hasText from OCR signals
                        ocr_has_text = ocr_signals.get('hasText', bool(ocr_text and ocr_text.strip()))
                        if raw_signals_present:
                            logger.warning(f"[Analysis] OCR status={ocr_status} but rawSignalsPresent=True - converting to partial")
                            ocr_result = {
                                'hasText': ocr_has_text,  # STRICT: Use hasText from OCR signals
                                'text': ocr_text if ocr_text else "[Text Detected]",  # Never empty
                                'confidence': max(0.3, ocr_confidence),  # Minimum confidence for partial
                                'failure': None,
                                'status': 'partial',  # NOT "fail" - signal exists
                                'words': ocr_words,
                                'rawSignalPresent': True
                            }
                        else:
                            ocr_result = {
                                'hasText': ocr_has_text,  # STRICT: Use hasText from OCR signals
                                'text': '',
                                'confidence': 0.0,
                                'failure': {
                                    'error': 'OCR_FAILED',
                                    'error_type': 'OCR_FAILED',
                                    'reason': ocr_signals.get('reason', 'OCR failed'),
                                    'message': f"OCR_FAILED: {ocr_signals.get('reason', 'OCR failed')}",
                                    'status': 'failed'
                                },
                                'status': 'fail',
                                'words': [],
                                'rawSignalPresent': False
                            }
                except ImportError:
                    # FIX: Fallback to OCR engine if HardSignalExtractor not available
                    logger.warning("[Analysis] HardSignalExtractor not available, using OCR engine fallback")
                    from ..ocr.ocr_engine import OCREngine
                    ocr_engine = OCREngine()  # FIX: No VLLM analyzer
                    ocr_result = ocr_engine.extract_text(image)
                    ocr_text = ocr_result.get('text') if ocr_result.get('hasText') else None
                    ocr_words = []  # OCR engine doesn't return word-level bboxes yet
                
                # FIX: Store OCR result separately (independent from copywriting)
                model_results['ocr_result'] = {
                    'hasText': ocr_result.get('hasText', bool(ocr_text)),
                    'text': ocr_text or '',
                    'confidence': ocr_result.get('confidence', 0.85 if ocr_text else 0.0),
                    'failure': ocr_result.get('failure'),
                    'analyzerStatus': ocr_result.get('status', 'passed' if ocr_result.get('hasText') else 'failed'),
                    'rawSignalPresent': bool(ocr_text),
                    'words': ocr_words,  # FIX: Word-level bounding boxes
                    'methods_tried': ocr_result.get('methods_tried', [])
                }
                
                # 🔥 CRITICAL FIX: Promote OCR result to FIRST-CLASS ANALYZER OUTPUT
                # OCR success must be registered as copywriting analyzer output immediately
                if ocr_result.get('hasText', False) and ocr_text and ocr_text.strip():
                    word_count = len(ocr_text.split())
                    model_results['copywriting_analysis'] = {
                        'status': 'success',  # OCR success = copywriting success
                        'extractedText': ocr_text,
                        'text_content': ocr_text,  # Alias
                        'wordCount': word_count,
                        'word_count': word_count,  # Alias
                        'source': 'google_ocr',
                        'rawSignalPresent': True,
                        'analyzerStatus': 'success',
                        'issues': [],  # Will be populated by copywriting analyzer
                        'copywriting_score': None,  # Will be computed by copywriting analyzer
                        'ocr_success': True  # Flag to indicate OCR provided the text
                    }
                    logger.info(f"[Analysis] ✅ OCR promoted to copywriting analyzer: {word_count} words, status=success")
                
                logger.info(f"[Analysis] OCR completed: hasText={ocr_result.get('hasText')}, confidence={ocr_result.get('confidence', 0.0):.2f}, words={len(ocr_words)}")
            except Exception as e:
                # CRITICAL: Structured failure for OCR errors
                error_msg = f"OCR_FAILED: Unexpected error during OCR extraction: {str(e)}"
                logger.error(f"[Analysis] {error_msg}", exc_info=True)
                # FIX: Store OCR failure but don't block other analyzers
                model_results['ocr_result'] = {
                    'hasText': False,
                    'text': '',
                    'confidence': 0.0,
                    'failure': {
                        'error': 'OCR_FAILED',
                        'error_type': 'OCR_FAILED',
                        'reason': f'OCR error: {str(e)}',
                        'message': error_msg,
                        'exception': str(e),
                        'status': 'failed'
                    },
                    'analyzerStatus': 'failed',
                    'rawSignalPresent': False,
                    'words': [],
                    'errors': [str(e)]
                }
            
            # ====================================================================
            # STAGE 4: TYPOGRAPHY ANALYSIS (INDEPENDENT)
            # FIX: Typography analysis runs independently, no dependencies
            # ====================================================================
            try:
                if not analysis_options or analysis_options.get('typography_analysis', {}).get('enabled', True):
                    logger.info("[Analysis] Running typography analysis (independent)...")
                    typography_options = (analysis_options or {}).get('typography_analysis', {})
                    
                    # 🔥 FIX: Pass OCR text regions to typography analyzer (decouple from Tesseract)
                    ocr_text_regions = None
                    if ocr_result and ocr_result.get('words'):
                        # Convert OCR words to text regions format
                        ocr_words = ocr_result.get('words', [])
                        text_regions_from_ocr = []
                        for word_info in ocr_words:
                            if isinstance(word_info, dict):
                                # ✅ CRITICAL: Filter out empty text BEFORE storing
                                word_text = word_info.get('word', '') or word_info.get('text', '')
                                if not word_text or not word_text.strip():
                                    continue  # Skip empty text regions
                                
                                bbox = word_info.get('bbox', [])
                                if bbox and len(bbox) >= 4:
                                    # Convert normalized bbox to pixel coordinates
                                    img_h, img_w = image.shape[:2]
                                    x1 = int(bbox[0] * img_w) if bbox[0] <= 1.0 else int(bbox[0])
                                    y1 = int(bbox[1] * img_h) if bbox[1] <= 1.0 else int(bbox[1])
                                    x2 = int(bbox[2] * img_w) if bbox[2] <= 1.0 else int(bbox[2])
                                    y2 = int(bbox[3] * img_h) if bbox[3] <= 1.0 else int(bbox[3])
                                    
                                    # Calculate approximate font size from bbox height
                                    approximate_font_size_px = y2 - y1 if y2 > y1 else 12
                                    
                                    text_regions_from_ocr.append({
                                        'text': word_text.strip(),  # ✅ Non-empty text only
                                        'bbox': [x1, y1, x2, y2],
                                        'approximateSizePx': approximate_font_size_px  # ✅ Only size, NO typography metadata
                                        # ✅ CRITICAL: OCR words must NOT carry font or typography metadata
                                        # ✅ NO fontClassification, NO confidence - these belong in typography layer
                                    })
                        if text_regions_from_ocr:
                            ocr_text_regions = text_regions_from_ocr
                            logger.info(f"[Analysis] Using {len(ocr_text_regions)} OCR text regions for typography analysis (filtered empty text)")
                        else:
                            logger.warning("[Analysis] All OCR words had empty text - no text regions created")
                    
                    # FIX: Typography analyzer uses OCR regions if available, otherwise extracts from image
                    typography_result = self.typography_analyzer.analyze_typography(image, ocr_text_regions, typography_options)
                    
                    # FIX: Store raw output with analyzerStatus and visibilityState
                    typography_result['analyzerStatus'] = typography_result.get('status', 'passed')
                    # ✅ NO confidence field - OCR confidence stays in OCR layer only
                    typography_result['rawSignalPresent'] = bool(typography_result.get('detected_fonts') or typography_result.get('fonts', []))
                    # 🔥 Add visibilityState to raw output
                    typography_result['visibilityState'] = self._get_visibility_state(typography_result['analyzerStatus'], analyzer_type="typography")
                    
                    model_results['typography_analysis'] = typography_result
                    logger.info(f"[Analysis] Typography analysis completed: status={typography_result.get('analyzerStatus')}")
                else:
                    logger.info("[Analysis] Typography analysis disabled")
                    model_results['typography_analysis'] = {
                        'status': 'skipped',
                        'reason': 'Typography analysis disabled in options',
                        'fonts': [],
                        'analyzerStatus': 'skipped',
                        'rawSignalPresent': False
                        # ✅ NO confidence field - OCR confidence stays in OCR layer only
                    }
            except Exception as e:
                logger.error(f"[Analysis] Typography analysis failed: {e}", exc_info=True)
                # FIX: Store failure output, don't skip
                model_results['typography_analysis'] = {
                    'status': 'failed',
                    'reason': f'Typography analysis error: {str(e)}',
                    'errors': [str(e)],
                    'fonts': [],
                    'analyzerStatus': 'failed',
                    'confidence': 0.0,
                    'rawSignalPresent': False
                }
            
            # ====================================================================
            # STAGE 5: COPYWRITING ANALYSIS (LAYER 2: SOFT REASONING + HARD SIGNALS)
            # FIX: Layer 1 (Hard Signals): Dictionary-based spelling detection
            # FIX: Layer 2 (Soft Reasoning): LLM-based copy quality, brand voice (if enabled)
            # FIX: OCR failure only affects copywriting, never blocks overall pipeline
            # ====================================================================
            try:
                if not analysis_options or analysis_options.get('copywriting_analysis', {}).get('enabled', True):
                    logger.info("[Analysis] Running copywriting analysis (marketing copy only)...")
                    from ..ocr.ocr_extractor import extract_ocr
                    from .text_classifier import classify_text_blocks

                    ocr_blocks = extract_ocr(image)
                    classified_blocks = classify_text_blocks(ocr_blocks)
                    marketing_blocks = [b for b in classified_blocks if b.get('label') == 'marketing_copy']

                    copywriting_result = self.copywriting_analyzer.analyze_copywriting(image, marketing_blocks)
                    copywriting_result['analyzerStatus'] = 'success'
                    copywriting_result['source'] = 'ocr'
                    copywriting_result['rawSignalPresent'] = bool(marketing_blocks)
                    copywriting_result['issueCount'] = len(copywriting_result.get('issues', []))

                    model_results['copywriting_analysis'] = copywriting_result
                else:
                    logger.info("[Analysis] Copywriting analysis disabled")
                    model_results['copywriting_analysis'] = {
                        'status': 'not_applicable',
                        'reason': 'Copywriting analysis disabled in options',
                        'analyzerStatus': 'skipped',
                        'rawSignalPresent': False,
                        'issues': []
                    }
            except Exception as e:
                logger.error("[Analysis] Copywriting analysis failed: {}".format(e), exc_info=True)
                model_results['copywriting_analysis'] = {
                    'status': 'not_applicable',
                    'reason': 'Copywriting analysis failed',
                    'analyzerStatus': 'failed',
                    'rawSignalPresent': False,
                    'issues': []
                }
            logger.info(f"[Analysis] Total model results: {len(model_results)} modules completed")
            
            # FIX: All analyzers run independently - no short-circuit logic
            # FIX: Return results even if some analyzers failed
            return {
                'model_results': model_results,
                'analysis_type': 'image_analysis'
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            # FIX: Return error but don't crash - return partial results if any exist
            if 'model_results' in locals() and model_results:
                logger.warning(f"[Analysis] Returning partial results despite error: {e}")
                return {
                    'model_results': model_results,
                    'analysis_type': 'image_analysis',
                    'error': f'Analysis partially failed: {str(e)}'
                }
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _extract_colors_fallback(self, image: np.ndarray, min_colors: int = 3) -> List[str]:
        """
        FIX: Fallback color extraction using k-means clustering.
        This ensures dominant colors are always returned even if color analyzer fails.
        
        Args:
            image: Image as numpy array (BGR format)
            min_colors: Minimum number of colors to extract
            
        Returns:
            List of hex color codes
        """
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Reshape image to 2D array of pixels
            h, w = image.shape[:2]
            pixels = image.reshape(-1, 3)
            
            # Sample pixels for faster clustering
            sample_size = min(1000, len(pixels))
            sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
            sample_pixels = pixels[sample_indices]
            
            # Convert BGR to RGB
            sample_pixels = sample_pixels[:, ::-1]
            
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=min_colors, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)
            
            # Get cluster centers (dominant colors)
            colors_rgb = kmeans.cluster_centers_.astype(int)
            
            # Convert RGB to hex
            hex_colors = []
            for rgb in colors_rgb:
                hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}".upper()
                hex_colors.append(hex_color)
            
            logger.info(f"[Analysis] Fallback color extraction: {len(hex_colors)} colors via k-means")
            return hex_colors
            
        except ImportError:
            logger.warning("[Analysis] sklearn not available for fallback color extraction, using simple pixel sampling")
            # FIX: Simple fallback - extract colors from different regions of image
            h, w = image.shape[:2]
            colors = []
            for i in range(min_colors):
                y = int(h * (i + 1) / (min_colors + 1))
                x = int(w * (i + 1) / (min_colors + 1))
                b, g, r = image[y, x]
                hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                colors.append(hex_color)
            logger.info(f"[Analysis] Simple fallback extraction: {len(colors)} colors from pixel sampling")
            return colors
        except Exception as e:
            logger.error(f"[Analysis] Fallback color extraction failed: {e}, using last-resort extraction")
            # FIX: Last resort - extract at least one color from image center
            try:
                h, w = image.shape[:2]
                colors = []
                # Extract from center and corners
                positions = [
                    (h // 2, w // 2),  # Center
                    (h // 4, w // 4),  # Top-left
                    (h // 4, 3 * w // 4),  # Top-right
                    (3 * h // 4, w // 4),  # Bottom-left
                    (3 * h // 4, 3 * w // 4),  # Bottom-right
                ]
                for y, x in positions[:min_colors]:
                    if 0 <= y < h and 0 <= x < w:
                        b, g, r = image[y, x]
                        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                        if hex_color not in colors:  # Avoid duplicates
                            colors.append(hex_color)
                if colors:
                    logger.info(f"[Analysis] Last-resort extraction: {len(colors)} colors from fixed positions")
                    return colors
                else:
                    # Absolute last resort - single color from center
                    y, x = h // 2, w // 2
                    b, g, r = image[y, x]
                    hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
                    logger.warning(f"[Analysis] Last-resort: returning single color from center: {hex_color}")
                    return [hex_color]
            except Exception as e2:
                logger.error(f"[Analysis] Last-resort color extraction also failed: {e2}")
                # FIX: Return at least one default color (gray) to never return empty
                return ["#808080"]  # Gray as absolute fallback
    
    def _analyze_document(self, document_path: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            # TODO: Implement document analysis
            return {
                'model_results': {},
                'analysis_type': 'document_analysis',
                'message': 'Document analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {'error': f'Document analysis failed: {str(e)}'}
    
    def _analyze_text(self, text_content: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            # TODO: Implement text analysis
            return {
                'model_results': {},
                'analysis_type': 'text_analysis',
                'message': 'Text analysis not yet implemented'
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {'error': f'Text analysis failed: {str(e)}'}
    
    def _analyze_url(self, url: str, analysis_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze URL content (webpage analysis).
        
        NOTE: This is a placeholder for future webpage analysis functionality.
        Currently, webpage URL analysis is not yet implemented.
        
        CRITICAL: We do NOT return compliance scores, buckets, or summaries here
        because no actual analysis has been performed. Returning fake scores would
        be misleading and could cause incorrect compliance decisions.
        """
        try:
            # Webpage URL analysis is not yet implemented
            # Return explicit not_supported response - do NOT return compliance scores
            return {
                'status': 'not_supported',
                'message': 'Webpage URL analysis is not yet supported',
                # Explicitly do NOT include:
                # - compliance_score (would be fake/placeholder)
                # - bucket (would be misleading)
                # - overall_compliance (would be 0, implying failure)
                # - summary (would imply analysis ran)
            }
            
        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return {'error': f'URL analysis failed: {str(e)}'}
    
    def _normalize_model_results(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize analyzer outputs into a stable schema expected by scoring.
        This prevents compliance collapsing to zero due to schema mismatch.
        """
        normalized = {}
        
        # Color
        if 'color_analysis' in model_results:
            color = model_results.get('color_analysis', {})
            normalized['color_analysis'] = {
                'brand_validation': {
                    'compliance_score': color
                    .get('brand_validation', {})
                    .get('compliance_score', 0)
                }
            }
        
        # Logo - CRITICAL: Preserve logo_detections with bbox for frontend display
        if 'logo_analysis' in model_results:
            logo = model_results.get('logo_analysis', {})
            normalized['logo_analysis'] = {
                'scores': {
                    'overall': logo.get('scores', {}).get('overall', 0)
                },
                'placement_validation': logo.get('placement_validation', {}),
                # CRITICAL: Preserve logo_detections with bbox for frontend visualization
                'logo_detections': logo.get('logo_detections', [])
            }
        
        # Typography (optional, future-safe)
        if 'typography_analysis' in model_results:
            typo = model_results.get('typography_analysis', {})
            normalized['typography_analysis'] = typo
        
        # Copywriting (optional)
        if 'copywriting_analysis' in model_results:
            copy = model_results.get('copywriting_analysis', {})
            normalized['copywriting_analysis'] = copy
        
        return normalized
    
    def _calculate_color_observable_score(self, dominant_colors: List[Dict]) -> float:
        """
        Calculate observable score for colors (no brand palette needed).
        
        Metrics:
        - Color diversity: More colors = better (up to 5 colors)
        - Coverage balance: Colors should have reasonable distribution
        
        Returns: 0-1 score
        """
        if not dominant_colors or len(dominant_colors) == 0:
            return 0.0
        
        # Score based on number of colors (3-5 is optimal)
        color_count_score = min(1.0, len(dominant_colors) / 5.0)
        
        # Score based on coverage balance (colors should be reasonably distributed)
        percentages = [c.get('percentage', 0) for c in dominant_colors if isinstance(c, dict)]
        if percentages:
            # Calculate coefficient of variation (lower = more balanced)
            mean_pct = sum(percentages) / len(percentages)
            std_pct = (sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)) ** 0.5
            cv = std_pct / mean_pct if mean_pct > 0 else 1.0
            balance_score = max(0.0, 1.0 - cv)  # Lower CV = higher score
        else:
            balance_score = 0.5
        
        # Weighted average
        observable_score = (color_count_score * 0.6) + (balance_score * 0.4)
        return round(observable_score, 2)
    
    def _calculate_typography_observable_score(self, observations: Dict) -> float:
        """
        Calculate observable score for typography (no brand fonts needed).
        
        Uses hierarchy, contrast, and readability metrics.
        
        Returns: 0-1 score
        """
        if not observations:
            return 0.0
        
        hierarchy = observations.get('text_hierarchy', {})
        contrast = observations.get('contrast', {})
        readability = observations.get('readability', {})
        
        hierarchy_score = 0.5 if (isinstance(hierarchy, dict) and hierarchy.get('hierarchy_detected', False)) else 0.3
        contrast_score = contrast.get('contrast_score', 0.0) if isinstance(contrast, dict) else 0.0
        readability_score = readability.get('readability_score', 0.0) if isinstance(readability, dict) else 0.0
        
        # Weights: contrast 40%, readability 40%, hierarchy 20%
        observable_score = (
            contrast_score * 0.4 +
            readability_score * 0.4 +
            hierarchy_score * 0.2
        )
        
        return round(max(0.0, min(1.0, observable_score)), 2)
    
    def _calculate_logo_observable_score(self, logo_analysis: Dict) -> float:
        """
        Calculate observable score for logos (no verification needed).
        
        Metrics:
        - Detection confidence: Higher confidence = better
        - Zone appropriateness: Top zones preferred
        - Size appropriateness: Not too small, not too large
        
        Returns: 0-1 score
        """
        detections = logo_analysis.get('detections', []) or logo_analysis.get('logo_detections', [])
        
        if not detections or len(detections) == 0:
            return 0.0
        
        # Score based on detection confidence
        confidences = [d.get('confidence', 0.0) for d in detections if isinstance(d, dict)]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Score based on zone (top zones preferred)
        zones = [d.get('zone', '') or d.get('position', '') for d in detections if isinstance(d, dict)]
        top_zones = ['top-left', 'top-center', 'top-right']
        zone_score = 1.0 if any(z in top_zones for z in zones) else 0.7
        
        # Score based on size (reasonable size preferred)
        size_ratios = [d.get('sizeRatio', 0) for d in detections if isinstance(d, dict)]
        if size_ratios:
            avg_size = sum(size_ratios) / len(size_ratios)
            # Optimal size: 0.01-0.10 (1-10% of image)
            if 0.01 <= avg_size <= 0.10:
                size_score = 1.0
            elif 0.005 <= avg_size < 0.01 or 0.10 < avg_size <= 0.15:
                size_score = 0.7
            else:
                size_score = 0.5
        else:
            size_score = 0.5
        
        # Weighted average
        observable_score = (avg_confidence * 0.5) + (zone_score * 0.3) + (size_score * 0.2)
        return round(max(0.0, min(1.0, observable_score)), 2)
    
    def _calculate_copywriting_observable_score(self, copywriting_analysis: Dict) -> float:
        """
        Calculate observable score for copywriting (always computed).
        
        Uses existing copywritingScore from analyzer.
        
        Returns: 0-1 score
        """
        score = copywriting_analysis.get('copywritingScore') or copywriting_analysis.get('copywriting_score', 0)
        
        # Convert 0-100 to 0-1 if needed
        if isinstance(score, (int, float)) and score > 1.0:
            score = score / 100.0
        
        return round(max(0.0, min(1.0, float(score))), 2)
    
    def _calculate_overall_compliance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall compliance using TWO-CHANNEL scoring:
        1. Observable Score (ALWAYS computed) - 0-1 scale
        2. Brand Compliance Score (ONLY when config exists) - 0-1 scale
        
        Final overallCompliance (0-100):
        - Use brandComplianceScore if available
        - Otherwise use observableScore × 100
        - Never return 0 if any signals exist (minimum 10)
        """
        model_results = results.get("model_results", {})
        ocr_result = model_results.get('ocr_result', {})
        
        # FIX: Check analyzer statuses to determine critical failure
        analyzer_statuses = {}
        analyzer_confidences = {}
        has_raw_signals = {}
        
        # FIX: Extract status from each analyzer
        for key, value in model_results.items():
            if isinstance(value, dict):
                analyzer_statuses[key] = value.get('analyzerStatus', value.get('status', 'unknown'))
                analyzer_confidences[key] = value.get('confidence', 0.0)
                has_raw_signals[key] = value.get('rawSignalPresent', False)
        
        # STEP 4: Fix OCR failure logic
        # Check if OCR failed but visible text exists (critical signal failure)
        copywriting_analysis = model_results.get("copywriting_analysis", {})
        ocr_failure_with_visible_text = (
            copywriting_analysis.get("ocr_failure", False) and
            copywriting_analysis.get("visible_text_detected", False) and
            copywriting_analysis.get("analyzerStatus") == "failed"
        )
        
        # FIX: Critical signal failure ONLY if ALL analyzers failed AND image load failed
        # OR if OCR failed despite visible text (STEP 4 requirement)
        all_failed = all(
            status in ['failed', 'not_detected'] and not has_raw_signals.get(key, False)
            for key, status in analyzer_statuses.items()
        )
        
        # FIX: Check if image load failed (only case for true critical failure)
        image_load_failed = 'error' in results or not model_results
        
        # STEP 4: Set criticalSignalFailure = true if OCR failed despite visible text
        critical_signal_failure = (image_load_failed and all_failed) or ocr_failure_with_visible_text
        
        if ocr_failure_with_visible_text:
            logger.error("[Compliance] CRITICAL SIGNAL FAILURE: OCR failed despite visible text detected")
        
        if critical_signal_failure:
            logger.error("[Compliance] CRITICAL SIGNAL FAILURE: All analyzers failed AND image load failed")
            return {
                'overall_compliance': None,
                'critical_signal_failure': True,
                'breakdown': {},
                'reason': 'Image load failed and all analyzers failed'
            }
        
        # ============================================================
        # TWO-CHANNEL SCORING MODEL
        # ============================================================
        # Channel A: Observable Score (ALWAYS computed) - 0-1 scale
        # Channel B: Brand Compliance Score (ONLY when config exists) - 0-1 scale
        # Final overallCompliance: Use brandComplianceScore if available, else observableScore × 100
        # ============================================================
        
        # Observable weights (for Channel A)
        OBSERVABLE_WEIGHTS = {
            "color": 0.3,
            "typography": 0.3,
            "logo": 0.2,
            "copywriting": 0.2
        }
        
        # Brand compliance weights (for Channel B)
        BRAND_WEIGHTS = {
            "color": 0.3,
            "typography": 0.3,
            "logo": 0.2,
            "copywriting": 0.2
        }
        
        # Initialize scoring structures
        observable_scores = {}
        brand_scores = {}
        breakdown = {}
        failed_penalty_count = 0

        def _adjust_observable_score(score: float, status: str, has_signal: bool) -> float:
            """
            Apply status-based adjustments for OBSERVABLE scores only.
            
            CRITICAL: Observable scores are informational, not judgmental.
            - observed/observed_only/unknown: Keep computed score (signals exist, just not validated)
            - missing/skipped/not_detected: 0 score when no signals
            - failed: 0 score (penalty applied separately)
            
            NOTE: This does NOT affect brand compliance - observed signals never contribute to brand scores.
            """
            if status in ["failed", "error"]:
                return 0.0
            if status in ["missing", "skipped", "not_detected"] and not has_signal:
                return 0.0
            # ✅ FIX: observed/observed_only/unknown signals still have observable scores
            # They just don't contribute to brand compliance (handled separately)
            return score

        # ============================================================
        # COLOR ANALYZER SCORING
        # ============================================================
        color_analysis = model_results.get("color_analysis", {})
        color_status = color_analysis.get("analyzerStatus", color_analysis.get("status", "unknown"))
        dominant_colors = color_analysis.get("dominant_colors", [])
        
        # ✅ ALWAYS compute observable score (Channel A)
        color_observable = self._calculate_color_observable_score(dominant_colors)
        color_has_signal = bool(dominant_colors)
        color_observable = _adjust_observable_score(color_observable, color_status, color_has_signal)
        if color_status == "failed":
            failed_penalty_count += 1
        observable_scores["color"] = color_observable
        
        # ✅ Compute brand compliance score (Channel B) - ONLY if validated
        color_brand = None
        if color_status == "validated":
            brand_validation = color_analysis.get("brand_validation", {})
            compliance_score = brand_validation.get("compliance_score")
            if compliance_score is not None:
                # Convert to 0-1 scale if needed
                color_brand = float(compliance_score) if compliance_score <= 1.0 else float(compliance_score) / 100.0
                brand_scores["color"] = color_brand
        
        # Build breakdown
            color_visibility_state = self._get_visibility_state(color_status, analyzer_type="color")
            breakdown["color"] = {
            "observableScore": color_observable,
            "brandScore": color_brand,
                "status": color_status,
            "visibilityState": color_visibility_state,
            "confidence": color_analysis.get("confidence", 0.85 if dominant_colors else 0.0),
            "dominantColors": dominant_colors  # ✅ Always include
        }

        # ============================================================
        # LOGO ANALYZER SCORING
        # ============================================================
        logo_analysis = model_results.get("logo_analysis", {})
        logo_status = logo_analysis.get("analyzerStatus", logo_analysis.get("status", "not_detected"))
        
        # ✅ ALWAYS compute observable score (Channel A)
        logo_observable = self._calculate_logo_observable_score(logo_analysis)
        logo_has_signal = bool(logo_analysis.get("detections", []))
        logo_observable = _adjust_observable_score(logo_observable, logo_status, logo_has_signal)
        if logo_status == "failed":
            failed_penalty_count += 1
        observable_scores["logo"] = logo_observable
        
        # V1 RULE: Logo is observational only - logo brand compliance score must ALWAYS be None
        logo_brand = None
        logger.debug(f"[Compliance] Logo brand score skipped: V1 rule - logo is observational only")
        
        # Build breakdown
        logo_visibility_state = self._get_visibility_state(logo_status, analyzer_type="logo")
        breakdown["logo"] = {
            "observableScore": logo_observable,
            "brandScore": None,  # V1: Always None - logo is observational only
            "status": logo_status,
            "visibilityState": logo_visibility_state,
                "confidence": logo_analysis.get("confidence", 0.0),
            "zone": logo_analysis.get("zone"),
            "detections": logo_analysis.get("detections", [])  # ✅ Always include
        }

        # ============================================================
        # TYPOGRAPHY ANALYZER SCORING
        # ============================================================
        typography_analysis = model_results.get("typography_analysis", {})
        typography_status = typography_analysis.get("analyzerStatus", typography_analysis.get("status", "observed"))
        observations = typography_analysis.get("observations", {})
        
        # ✅ ALWAYS compute observable score (Channel A)
        # Use pre-computed observableScore if available, otherwise calculate
        typography_observable = typography_analysis.get("observableScore")
        if typography_observable is None:
            typography_observable = self._calculate_typography_observable_score(observations)
        else:
            # Ensure it's 0-1 scale
            if isinstance(typography_observable, (int, float)) and typography_observable > 1.0:
                typography_observable = typography_observable / 100.0
        typography_has_signal = bool(typography_analysis.get("typographyStyles", []))
        typography_observable = _adjust_observable_score(
            float(typography_observable) if typography_observable is not None else 0.0,
            typography_status,
            typography_has_signal
        )
        if typography_status == "failed":
            failed_penalty_count += 1
        observable_scores["typography"] = typography_observable
        
        # V1 RULE: Typography is observational only - font detection from raster images is unreliable
        # Typography brand compliance score must ALWAYS be None
        typography_brand = None
        logger.debug(f"[Compliance] Typography brand score skipped: V1 rule - typography is observational only")
        
        # Build breakdown
        typography_visibility_state = self._get_visibility_state(typography_status, analyzer_type="typography")
        breakdown["typography"] = {
            "observableScore": observable_scores["typography"],
            "brandScore": None,  # V1: Always None - typography is observational only
            "status": "observed",  # V1: Always "observed" - font detection unreliable
            "visibilityState": typography_visibility_state,
            "typographyStyles": typography_analysis.get("typographyStyles", []),  # ✅ Clustered styles (2-5 max)
            "textRegionsCount": sum(style.get('regionCount', 0) for style in typography_analysis.get("typographyStyles", [])),
            "explanation": "Typography analysis is observational only. Font family detection from raster images is not reliable."
        }

        # ============================================================
        # COPYWRITING ANALYZER SCORING
        # ============================================================
        copywriting_analysis = model_results.get("copywriting_analysis", {})
        copywriting_status = copywriting_analysis.get("analyzerStatus", copywriting_analysis.get("status", "skipped"))
        ocr_failure = copywriting_analysis.get("ocr_failure", False)
        ocr_success = copywriting_analysis.get("ocr_success", False)
        extracted_text = copywriting_analysis.get("extractedText", "") or copywriting_analysis.get("extracted_text", "") or copywriting_analysis.get("text_content", "")
        
        # ✅ ALWAYS compute observable score (Channel A)
        copywriting_observable = self._calculate_copywriting_observable_score(copywriting_analysis)
        copywriting_has_signal = bool(extracted_text)
        copywriting_observable = _adjust_observable_score(copywriting_observable, copywriting_status, copywriting_has_signal)
        if copywriting_status == "failed":
            failed_penalty_count += 1
        observable_scores["copywriting"] = copywriting_observable
        
        # ✅ Compute brand compliance score (Channel B) - ALWAYS available for copywriting
        # FIX: Only apply penalties to confirmed errors, not likely OCR artifacts
        copywriting_brand = copywriting_observable  # Copywriting observable = brand score (always computed)
        
        # Apply spelling error penalty - ONLY for confirmed errors, not OCR artifacts
        if copywriting_status in ["clean", "needs_revision"]:
            issues = copywriting_analysis.get("issues", []) or []
            # Count only confirmed errors (not likely_ocr_artifact)
            confirmed_errors = [
                issue for issue in issues
                if isinstance(issue, dict) and issue.get("errorType") != "likely_ocr_artifact"
            ]
            issue_count = len(confirmed_errors)
            if issue_count:
                error_penalty = min(0.2, issue_count * 0.02)  # Max 20% reduction (0-1 scale)
                copywriting_brand = max(0.0, copywriting_brand - error_penalty)
                logger.info(
                    f"[Compliance] Reduced copywriting score by {error_penalty*100:.1f}% due to {issue_count} confirmed errors (excluded {len(issues) - issue_count} OCR artifacts)"
                )
        
        brand_scores["copywriting"] = copywriting_brand
        
        # Build breakdown
        copywriting_visibility_state = self._get_visibility_state(copywriting_status)
        breakdown['copywriting'] = {
            "observableScore": copywriting_observable,
            "brandScore": copywriting_brand,
            "status": copywriting_status,
            "visibilityState": copywriting_visibility_state,
            "issueCount": len(copywriting_analysis.get('issues', []))
        }
        
        # ============================================================
        # FINAL SCORE CALCULATION (TWO-CHANNEL MODEL)
        # ============================================================
        
        # Step 1: Calculate observable weighted average (Channel A)
        observable_score = sum(
            observable_scores.get(k, 0.0) * OBSERVABLE_WEIGHTS.get(k, 0.0)
            for k in OBSERVABLE_WEIGHTS.keys()
        )

        # Apply failed signal penalty (5% per failed analyzer, max 20%)
        failed_penalty = min(0.2, failed_penalty_count * 0.05)
        if failed_penalty > 0:
            observable_score = max(0.0, observable_score - failed_penalty)
        
        analysis_options = results.get('analysis_options', {}) or {}
        analysis_mode = analysis_options.get('analysisMode')

        # Step 2: Calculate brand compliance weighted average (Channel B) - ONLY if any brand scores exist
        brand_compliance_score = None
        if brand_scores and analysis_mode != "observational_only":
            # Normalize brand weights
            total_brand_weight = sum(BRAND_WEIGHTS.get(k, 0.0) for k in brand_scores.keys())
            if total_brand_weight > 0:
                normalized_brand_weights = {
                    k: BRAND_WEIGHTS.get(k, 0.0) / total_brand_weight
                    for k in brand_scores.keys()
                }
                brand_compliance_score = sum(
                    brand_scores[k] * normalized_brand_weights[k]
                    for k in brand_scores.keys()
                )
        
        # Step 3: Determine overall compliance
        if analysis_mode == "observational_only":
            overall_compliance = observable_score * 100
        elif brand_compliance_score is not None:
            overall_compliance = brand_compliance_score * 100  # Scale to 0-100
        else:
            overall_compliance = observable_score * 100  # Scale to 0-100
        
        # Step 4: Check for signals
        # ✅ FIX: Do NOT enforce minimum score - missing brand config should not be penalized
        # If brand_compliance_score is None, observable_score is used (which may be low if signals are weak)
        # This is correct behavior - we don't want to artificially inflate scores
        has_signals = (
            len(dominant_colors) > 0 or
            len(typography_analysis.get("typographyStyles", [])) > 0 or
            len(logo_analysis.get("detections", [])) > 0 or
            extracted_text or
            ocr_success
        )
        
        # Clamp to 0-100
        if overall_compliance is not None:
            overall_compliance = max(0, min(100, round(overall_compliance, 1)))
        
        return {
            'overall_compliance': overall_compliance,
            'observable_score': round(observable_score, 2),
            'brand_compliance_score': round(brand_compliance_score, 2) if brand_compliance_score is not None else None,
            'critical_signal_failure': False,
            'breakdown': breakdown
        }
    
    def _generate_summary_and_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX: Generate summary and recommendations from available signals.
        
        RULES:
        - Never return empty objects silently
        - Always include analyzerStatus and confidence
        - Provide reasons for failures
        - Compute from available signals (never null compliance except true critical failure)
        """
        try:
            model_results = results.get('model_results', {})
            
            # FIX: Always generate summary, even if some analyzers failed
            overall_compliance = results.get('overall_compliance', 0)
            critical_signal_failure = results.get('critical_signal_failure', False)
            
            # FIX: Handle null compliance (only if ALL analyzers failed AND image load failed)
            if overall_compliance is None and critical_signal_failure:
                # FIX: True critical failure - ALL analyzers failed AND image load failed
                recommendations = [
                    'Image load failed and all analyzers failed - unable to analyze',
                    'Please verify image file is valid and accessible',
                    'Check system logs for details on critical errors',
                    'Review analyzer configuration and system health'
                ]
                
                return {
                    'summary': 'Analysis could not be completed due to critical system errors (image load failed and all analyzers failed)',
                    'recommendations': recommendations
                }
            
            # FIX: Generate summary from available signals (never null compliance)
            # FIX: If some analyzers failed, still compute compliance from available ones
            if overall_compliance is None or overall_compliance == "UNKNOWN":
                # FIX: This should not happen with new logic, but handle gracefully
                overall_compliance = 0
            
            analysis_options = results.get('analysis_options', {}) or {}
            analysis_mode = analysis_options.get('analysisMode')
            color_options = analysis_options.get('color_analysis', {}) if isinstance(analysis_options, dict) else {}
            typography_options = analysis_options.get('typography_analysis', {}) if isinstance(analysis_options, dict) else {}
            logo_options = analysis_options.get('logo_analysis', {}) if isinstance(analysis_options, dict) else {}
            brand_palette = color_options.get('brand_palette')
            brand_fonts = typography_options.get('expected_fonts') or typography_options.get('brand_fonts')
            logo_rules = logo_options.get('allowedZones') or logo_options.get('logo_rules')
            has_brand_config = bool(brand_palette or brand_fonts or logo_rules)
            
            # Generate summary based on compliance score
            if analysis_mode == "observational_only":
                summary = "Analysis is observational only. Complete brand setup to enable compliance checks."
            elif isinstance(overall_compliance, (int, float)):
                if not has_brand_config:
                    summary = "Partial analysis completed. Brand rules not provided."
                elif overall_compliance >= 85:
                    summary = "Excellent brand compliance"
                elif overall_compliance >= 60:
                    summary = "Good brand compliance with minor issues"
                else:
                    summary = "Brand compliance issues detected"
            else:
                summary = "Analysis completed - compliance computed from available signals"
            
            # FIX: Generate recommendations from analyzer statuses (never empty)
            recommendations = []
            
            # FIX: Check each analyzer status and provide recommendations
            # Color recommendations
            color_analysis = model_results.get('color_analysis', {})
            color_status = color_analysis.get('analyzerStatus', color_analysis.get('status', 'unknown'))
            if color_status == 'failed':
                recommendations.append(f"Color analysis failed: {color_analysis.get('reason', 'Unknown error')}")
            elif color_status == 'observed_only':
                recommendations.append('Color analysis: Brand palette not provided - colors observed only')
            
            # Logo recommendations
            logo_analysis = model_results.get('logo_analysis', {})
            logo_status = logo_analysis.get('analyzerStatus', logo_analysis.get('status', 'unknown'))
            logo_detected = logo_analysis.get('detected', False)
            if logo_status == 'failed':
                recommendations.append(f"Logo analysis failed: {logo_analysis.get('reason', 'Unknown error')}")
            elif logo_status == 'not_detected' and not logo_detected:
                # CRITICAL: Only suggest "logo not detected" if NO logo-like regions exist
                logo_raw_signal = logo_analysis.get('rawSignalPresent', False)
                logo_low_confidence = logo_status == 'low_confidence'
                if not logo_raw_signal and not logo_low_confidence:
                    recommendations.append('Logo not detected in image - verify logo is visible')
                else:
                    # Logo-like regions exist - suggest verification
                    recommendations.append('Logo-like regions detected - verify logo identity')
            
            # Copywriting recommendations
            copywriting_analysis = model_results.get('copywriting_analysis', {})
            copywriting_status = copywriting_analysis.get('analyzerStatus', copywriting_analysis.get('status', 'unknown'))
            
            # STRICT REQUIREMENT: Never emit "OCR failed to extract text" if hasText = true
            ocr_result = model_results.get('ocr_result', {})
            ocr_has_text = ocr_result.get('hasText', False)
            
            if copywriting_status == 'failed':
                ocr_failure = copywriting_analysis.get('ocr_failure', False)
                if ocr_failure:
                    # STRICT: Only say OCR failed if hasText = false
                    if not ocr_has_text:
                        recommendations.append('OCR failed to extract text - check image quality and OCR configuration')
                    else:
                        # hasText = true but OCR failed - this is a system error
                        recommendations.append('OCR detected text but extraction failed - system error, check OCR configuration')
                else:
                    recommendations.append(f"Copywriting analysis failed: {copywriting_analysis.get('reason', 'Unknown error')}")
            elif copywriting_status == 'skipped':
                # STRICT: Only say insufficient text if hasText = false
                if not ocr_has_text:
                    recommendations.append(f"Copywriting skipped: {copywriting_analysis.get('reason', 'Insufficient text')}")
                else:
                    # hasText = true but skipped - this shouldn't happen
                    recommendations.append('Copywriting skipped despite text detection - review analysis configuration')
            
            # Typography recommendations
            typography_analysis = model_results.get('typography_analysis', {})
            typography_status = typography_analysis.get('analyzerStatus', typography_analysis.get('status', 'unknown'))
            if typography_status == 'failed':
                recommendations.append(f"Typography analysis failed: {typography_analysis.get('reason', 'Unknown error')}")
            elif typography_status == 'observed':
                recommendations.append('Typography analysis: Brand font guidelines not provided - fonts observed only')
            
            # FIX: Always include at least one recommendation (never empty)
            if not recommendations:
                if overall_compliance >= 85:
                    recommendations.append('Content meets brand guidelines')
                else:
                    recommendations.append('Review analysis results for improvement opportunities')
            
            return {
                'summary': summary,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return {
                'summary': 'Analysis completed with errors',
                'recommendations': ['Review analysis results for details', f'Error: {str(e)}']
            }
    
    def _get_visibility_state(self, status: str, analyzer_type: Optional[str] = None) -> str:
        """
        Map analyzer status to visibilityState.
        
        Args:
            status: Analyzer status string
            analyzer_type: Optional analyzer type ("logo", "color", "typography", "copywriting")
                          Used for special cases (e.g., logo "not_detected" = "evaluated")
        
        Returns:
            "evaluated" | "observed" | "missing"
        """
        # 🔥 SPECIAL CASE: Logo "not_detected" = "evaluated" (analyzer evaluated and determined no logo)
        if analyzer_type == "logo" and status == "not_detected":
            return "evaluated"
        
        if status in ["passed", "failed", "detected", "validated", "analyzed", "completed", "success", "not_detected"]:
            return "evaluated"
        elif status in ["unknown", "observed_only", "low_confidence", "observed"]:
            return "observed"
        else:
            return "missing"
    
    def _transform_to_target_schema(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX: Transform analysis results to match golden schema format.
        
        Golden Schema:
        {
          "overallCompliance": 72,
          "bucket": "review",
          "logo": {
            "detected": true,
            "zone": "top-center",
            "violations": []
          },
          "copywriting": {
            "status": "needs_revision",
            "issues": [{"type": "spelling", "original": "Enre", "suggestion": "Entire", "confidence": "high"}]
          },
          "colors": {
            "dominant": ["#6C4DFF", "#FFFFFF"],
            "status": "observed"
          },
          "criticalSignalFailure": false
        }
        """
        try:
            model_results = results.get('model_results', {})
            
            # Get analysis_options from results if available
            analysis_options = results.get('analysis_options', {}) or {}
            analysis_mode = analysis_options.get('analysisMode')
            observational_only = analysis_mode == "observational_only"
            
            # FIX: Extract analyzer results (independent - no dependencies)
            logo_analysis = model_results.get('logo_analysis', {}) or {}
            copywriting_analysis = model_results.get('copywriting_analysis', {}) or {}
            color_analysis = model_results.get('color_analysis', {}) or {}
            typography_analysis = model_results.get('typography_analysis', {}) or {}
            
            # FIX: Critical signal failure ONLY if ALL analyzers failed AND image load failed
            # FIX: Use the critical_signal_failure already determined by _calculate_overall_compliance
            critical_signal_failure = results.get('critical_signal_failure', False)
            
            # FIX: Verify critical signal failure is correct (should only be true if ALL failed AND image load failed)
            # FIX: Individual analyzer failures do NOT cause critical signal failure
            # This is already set correctly by _calculate_overall_compliance, so just use it
            
            # FIX: Transform logo analysis to match golden schema
            logo_status = logo_analysis.get('analyzerStatus', logo_analysis.get('status', 'not_detected'))
            if observational_only:
                logo_status = 'observed'
            logo_detected = logo_analysis.get('detected', False)
            
            # FIX: Extract zone from detections or placement validation
            logo_zone = None
            logo_violations = []
            
            # FIX: Get zone from zone_info or placement validation
            detections = logo_analysis.get('detections', []) or logo_analysis.get('logo_detections', [])
            detections_cleaned = []
            for det in detections:
                if isinstance(det, dict):
                    det_clean = {k: v for k, v in det.items() if k != 'confidence'}
                    detections_cleaned.append(det_clean)
                else:
                    detections_cleaned.append(det)
            if detections and isinstance(detections, list) and len(detections) > 0:
                # FIX: Extract zone from first detection
                first_detection = detections[0] if isinstance(detections[0], dict) else {}
                logo_zone = first_detection.get('zone') or first_detection.get('position', 'unknown')
            
            # FIX: Get violations from placement validation
            placement_validation = logo_analysis.get('placement_validation', {})
            if placement_validation:
                logo_violations = placement_validation.get('violations', [])
            
            # FIX: Format logo to match golden schema: {detected, zone, violations}
            logo_golden = {
                'detected': logo_detected,
                'zone': logo_zone if logo_detected else None,
                'violations': logo_violations if logo_detected else []
            }
            
            # ✅ Also keep detailed format for compatibility
            # ✅ ALWAYS include logo analyzer, even when not_detected
            logo_visibility_state = self._get_visibility_state(logo_status, analyzer_type="logo")
            
            # ✅ Extract placementDetails from placement validation
            placement_details = placement_validation.get('placementDetails', []) if placement_validation else []
            
            # ✅ FIX: Get logo status from analyzer (verified | observed | not_detected)
            logo_analyzer_status = logo_analysis.get('analyzerStatus', logo_analysis.get('status', 'not_detected'))
            has_reference_logo = logo_analysis.get('has_reference_logo', False)
            verified_logos = logo_analysis.get('verified_logos', [])
            
            # ✅ FIX: Determine explanation based on status
            logo_explanation = None
            if logo_analyzer_status == 'observed':
                logo_explanation = 'Logo-like regions detected but identity not verified. Reference logo required for validation and placement compliance.'
            elif logo_analyzer_status == 'not_detected':
                logo_explanation = 'No logos detected in image. Analyzer evaluated standard logo zones.'
            elif logo_analyzer_status == 'verified':
                logo_explanation = 'Logo verified and validated against reference logo.'
            
            placement_status = placement_validation.get('status', 'not_applicable') if placement_validation else 'not_applicable'
            placement_reason = None
            if placement_status == 'not_applicable':
                if not has_reference_logo:
                    placement_reason = 'Reference logo not provided - placement validation requires verified logo identity'
                elif not verified_logos:
                    placement_reason = 'Logo identity not verified - placement validation requires verified logo'
                else:
                    placement_reason = placement_validation.get('message', 'Placement validation not applicable')
            
            logo_analysis_transformed = {
                'visibilityState': logo_visibility_state,
                'status': logo_analyzer_status,  # ✅ Use analyzer status: verified | observed | not_detected
                'detections': detections_cleaned if detections_cleaned else [],  # ✅ Always array, never null
                'verified_logos': verified_logos,  # ✅ Expose verified logos
                'placementValidation': {
                    'status': placement_status,
                    'complianceScore': placement_validation.get('compliance_score') if placement_validation else None,  # ✅ null if not applicable
                    'validPlacements': placement_validation.get('validPlacements', 0) if placement_validation else 0,
                    'totalPlacements': placement_validation.get('totalPlacements', len(detections)) if placement_validation else len(detections),
                    'violations': placement_validation.get('violations', []) if placement_validation else [],
                    'placementDetails': placement_details,  # ✅ Always include
                    'reason': placement_reason  # ✅ NEW: Explanation for why validation was skipped
                },
                'message': logo_analysis.get('message') or (
                    'No logo detected in image' if logo_analyzer_status == 'not_detected' else None
                ),
                'explanation': logo_explanation,  # ✅ NEW: Human-readable explanation
                'expectedZones': analysis_options.get('logo_zones', ['top-left', 'top-center']) if analysis_options else ['top-left', 'top-center']
            }
            
            if analysis_mode == "observational_only":
                logo_analysis_transformed['status'] = 'observed'
                logo_analysis_transformed['placementValidation'] = {
                    'status': 'not_applicable',
                    'complianceScore': None,
                    'validPlacements': 0,
                    'totalPlacements': len(detections_cleaned),
                    'violations': [],
                    'placementDetails': []
                }
            
            # FIX: Transform copywriting analysis to match strict schema
            copywriting_analysis = model_results.get('copywriting_analysis', {}) or {}
            copywriting_status = copywriting_analysis.get('status', 'not_applicable')

            issues = copywriting_analysis.get('issues', []) if isinstance(copywriting_analysis.get('issues', []), list) else []
            if issues and copywriting_status == 'clean':
                copywriting_status = 'needs_revision'

            # ✅ FIX: Add explanations for copywriting issues
            copywriting_explanation = None
            grammar_analysis = copywriting_analysis.get('grammar_analysis', {})
            ocr_artifacts = grammar_analysis.get('ocrArtifacts', [])
            if ocr_artifacts:
                copywriting_explanation = f"Found {len(ocr_artifacts)} likely OCR artifacts. These may be OCR noise rather than real spelling errors."
            
            copywriting_analysis_transformed = {
                'status': copywriting_status,
                'issueCount': len(issues),
                'issues': issues,
                'explanation': copywriting_explanation,  # ✅ NEW: Explanation for OCR artifacts
                'ocrArtifacts': ocr_artifacts if ocr_artifacts else []  # ✅ NEW: Expose OCR artifacts separately
            }

            # Legacy format for backward compatibility (golden schema)
            copywriting_golden = copywriting_analysis_transformed
            
            # FIX: Transform color analysis to match golden schema
            color_analysis = model_results.get('color_analysis', {}) or {}
            color_status = color_analysis.get('analyzerStatus', color_analysis.get('status', 'observed_only'))
            if observational_only:
                color_status = 'observed_only'
            brand_validation = color_analysis.get('brand_validation', {}) or {}
            
            # ✅ CRITICAL: Preserve dominant colors - NEVER overwrite to empty array
            dominant_colors = color_analysis.get('dominant_colors', [])
            if not isinstance(dominant_colors, list):
                dominant_colors = []
            
            # ✅ Fallback: Extract from hard signals if colors are missing
            if not dominant_colors:
                hard_signals = model_results.get('hard_signals', {})
                color_signals = hard_signals.get('color_signals', {})
                if color_signals.get('colors'):
                    dominant_colors = color_signals['colors']
                    logger.info(f"[Schema] Using {len(dominant_colors)} colors from hard signals")
            
            # ✅ CRITICAL: If still empty, try to get image from results or extract from hard signals
            if not dominant_colors:
                logger.warning("[Schema] No colors found in analyzer - checking hard signals")
                # Try to get image from results if available (for emergency extraction)
                # Note: Image may not be available in schema transformation, so we rely on hard signals
                if not dominant_colors:
                    logger.warning("[Schema] No colors available - this should not happen if extraction ran correctly")
            
            # Extract hex colors from dominant_colors (can be hex strings or dicts with hex)
            dominant_colors_hex = []
            for color in dominant_colors:
                if isinstance(color, dict):
                    hex_color = color.get('hex', '') or color.get('color', '')
                    if hex_color:
                        dominant_colors_hex.append(hex_color.upper() if hex_color.startswith('#') else f"#{hex_color.upper()}")
                elif isinstance(color, str):
                    dominant_colors_hex.append(color.upper() if color.startswith('#') else f"#{color.upper()}")
            
            # ✅ CRITICAL: If analyzer returned empty, fallback to hard signals
            if not dominant_colors_hex:
                hard_signals = model_results.get('hard_signals', {})
                color_signals = hard_signals.get('color_signals', {})
                hard_colors = color_signals.get('colors', [])
                for color in hard_colors:
                    if isinstance(color, dict):
                        hex_color = color.get('hex') or color.get('color')
                        if hex_color:
                            dominant_colors_hex.append(hex_color.upper() if hex_color.startswith('#') else f"#{hex_color.upper()}")
                if not dominant_colors_hex:
                    logger.error("[Schema] Color analyzer ran but no colors available - using neutral fallback")
                    dominant_colors_hex = ["#000000", "#FFFFFF", "#808080"]

            # ✅ CRITICAL: Ensure minimum 3 colors (min 3, max 5)
            if len(dominant_colors_hex) < 3:
                while len(dominant_colors_hex) < 3:
                    dominant_colors_hex.append(dominant_colors_hex[0])
            
            # ✅ Limit to top 5 colors for UI (min 3, max 5)
            dominant_colors_hex = dominant_colors_hex[:5]
            
            # Format colors to match golden schema: {dominant, status}
            colors_golden = {
                'dominant': dominant_colors_hex[:5],  # Top 5 colors
                'status': 'observed' if color_status == 'observed_only' else 'validated' if color_status == 'validated' else 'observed'
            }
            
            # Also keep detailed format for compatibility
            brand_compliance_score = brand_validation.get('compliance_score') if brand_validation else None
            if observational_only or color_status == 'observed_only' or not color_status or color_status == 'unknown':
                brand_compliance_score = None
                color_status = 'observed_only'
            
            # ✅ ALWAYS include color analyzer, even when observed_only
            color_visibility_state = self._get_visibility_state(color_status, analyzer_type="color")
            extraction_metadata = color_analysis.get('extractionMetadata', {}) or {}
            if isinstance(extraction_metadata, dict) and 'confidence' in extraction_metadata:
                extraction_metadata = {k: v for k, v in extraction_metadata.items() if k != 'confidence'}

            color_analysis_transformed = {
                'visibilityState': color_visibility_state,
                'status': color_status if color_status in ['observed_only', 'validated'] else 'observed_only',
                'dominantColors': dominant_colors_hex,  # ✅ NEVER empty if colors were extracted
                'extractionMetadata': extraction_metadata,
                'observableScore': color_analysis.get('observableScore'),  # ✅ Expose observable score
                'brandValidation': {
                    'status': brand_validation.get('status', 'unknown') if brand_validation else 'unknown',
                    'reason': brand_validation.get('reason', '') if brand_validation else (
                        'Brand palette not provided - color compliance cannot be validated' if color_status == 'observed_only' else ''
                    ),
                    'complianceScore': brand_compliance_score  # null when observed_only
                },
                'message': 'Brand colors observed but no palette provided' if color_status == 'observed_only' else None,
                'explanation': (
                    'Brand palette not provided. Dominant colors were extracted but compliance cannot be validated without brand color configuration.'
                    if color_status == 'observed_only' else None
                )
            }
            
            if analysis_mode == "observational_only":
                color_analysis_transformed['status'] = 'observed'
                color_analysis_transformed['brandValidation'] = {
                    'status': 'not_applicable',
                    'reason': 'Observational only - brand rules not configured',
                    'complianceScore': None
                }
            
            # 🔥 ALWAYS include typography analyzer, even when status=observed
            typography_status = typography_analysis.get('analyzerStatus', typography_analysis.get('status', 'observed'))
            if observational_only:
                typography_status = 'observed'
            if analysis_mode == "observational_only":
                typography_status = 'observed'
            typography_visibility_state = self._get_visibility_state(typography_status, analyzer_type="typography")
            
            # Extract typography observations (always available)
            # 🔥 CRITICAL: Use typographyStyles instead of per-word text_regions
            typography_styles = typography_analysis.get('typographyStyles', [])
            text_regions = []
            font_analysis = []
            typography_validation = {}
            
            # Extract observations from the analyzer output (if available)
            observations_raw = typography_analysis.get('observations', {})
            
            # Calculate observable metrics from observations or fallback to defaults
            text_hierarchy = observations_raw.get('text_hierarchy', {}) if observations_raw else {}
            contrast = observations_raw.get('contrast', {}) if observations_raw else {}
            readability = observations_raw.get('readability', {}) if observations_raw else {}
            
            # Extract scores from nested observations
            text_hierarchy_detected = text_hierarchy.get('detected', False) if isinstance(text_hierarchy, dict) else False
            readability_score = readability.get('score', 0) if isinstance(readability, dict) else 0
            contrast_score = contrast.get('score', 0) if isinstance(contrast, dict) else 0
            
            # ✅ Extract observableScore (computed from hierarchy + contrast + readability)
            observable_score = typography_analysis.get('observableScore')
            
            # Extract limitations
            limitations = typography_analysis.get('limitations', [])
            
            # Count total text regions from styles
            total_text_regions = len(typography_styles)
            
            # ✅ FIX: Build typography analysis with all required fields
            typography_analysis_transformed = {
                'status': typography_status,
                'visibilityState': typography_visibility_state,
                'typographyStyles': typography_styles,  # ✅ NEVER overwrite to empty array
                'observableScore': observable_score,  # ✅ Expose observable score
                'textRegionsCount': total_text_regions,
                'reason': typography_analysis.get('reason') or (
                    'Brand font guidelines not provided - typography observed only' if typography_status == 'observed' else None
                ),
                'limitations': limitations if limitations else (
                    ['Font family identification not reliable from raster images'] if typography_status == 'observed' else []
                )
            }
            
            # ✅ FIX: Add explanation for missing brand config
            if typography_status == 'observed':
                typography_analysis_transformed['message'] = 'Typography observed using OCR regions (no font rules provided)'
                typography_analysis_transformed['explanation'] = 'Brand font guidelines not provided. Only text hierarchy, contrast, and readability were evaluated. Font compliance cannot be validated without brand font configuration.'
            
            # FIX: Build recommendations from analyzer statuses (never empty)
            recommendations = results.get('recommendations', []) or []
            
            # FIX: Add recommendations based on analyzer statuses (not critical failures)
            # STRICT REQUIREMENT: Never emit "Consider adding a logo" if logo text is detected
            # CRITICAL: Only suggest "add logo" if NO logo-like regions exist AND no logo text
            logo_raw_signal = logo_analysis.get('rawSignalPresent', False)
            logo_low_confidence = logo_status == 'low_confidence'
            
            # Check if OCR text contains brand name (logo text detection)
            ocr_result = model_results.get('ocr_result', {})
            ocr_has_text = ocr_result.get('hasText', False)
            ocr_text = ocr_result.get('text', '') if ocr_result else ''
            brand_name = analysis_options.get('brand_name') if analysis_options else None
            logo_text_detected = False
            if ocr_has_text and brand_name and ocr_text:
                logo_text_detected = brand_name.lower() in ocr_text.lower()
            
            if (logo_status == 'not_detected' or not logo_detected) and not logo_raw_signal and not logo_low_confidence:
                # STRICT: Only suggest adding logo if no logo text detected
                if not logo_text_detected:
                    recommendations.append('Consider adding a logo for brand recognition')
                else:
                    # Logo text detected - don't suggest adding logo
                    recommendations.append('Brand name text detected - verify logo placement matches text position')
            elif logo_raw_signal or logo_low_confidence:
                # Logo-like regions exist - suggest verification, not adding
                recommendations.append('Logo-like regions detected - verify logo identity and placement')
            elif logo_text_detected:
                # Logo text detected but no object detection - suggest verification
                recommendations.append('Brand name text detected - verify logo placement matches text position')
            
            # STRICT REQUIREMENT: Never emit "OCR failed to extract text" if hasText = true
            if copywriting_analysis.get('ocr_failure', False) and not ocr_has_text:
                # Only add recommendation if hasText = false
                recommendations.append('OCR failed to extract text - check image quality and OCR configuration')
            elif copywriting_analysis.get('ocr_failure', False) and ocr_has_text:
                # hasText = true but OCR failed - system error
                recommendations.append('OCR detected text but extraction failed - system error, check OCR configuration')
            
            copy_issues = copywriting_analysis.get('issues', []) or []
            if copy_issues:
                recommendations.append("Copy issues detected - review suggested edits")
            
            # FIX: Always include at least one recommendation (never empty)
            if not recommendations:
                if overall_compliance_value and isinstance(overall_compliance_value, (int, float)) and overall_compliance_value >= 85:
                    recommendations.append('Content meets brand guidelines')
                else:
                    recommendations.append('Review analysis results for improvement opportunities')
            
            # FIX: Post-process recommendations (deduplicate, tag sources, remove user-blaming)
            try:
                from .recommendation_postprocessor import RecommendationPostProcessor
                postprocessor = RecommendationPostProcessor()
                analyzer_statuses = postprocessor.extract_analyzer_statuses(results)
                # Create temporary dict for post-processing
                temp_results = {'recommendations': recommendations, 'criticalSignalFailure': critical_signal_failure}
                temp_results = postprocessor.process(temp_results, analyzer_statuses)
                recommendations = temp_results.get('recommendations', recommendations)
            except Exception as e:
                logger.warning(f"[Schema] Recommendation post-processing failed: {e}")
            
            # FIX: Calculate bucket from overall compliance (never null except true critical failure)
            overall_compliance_value = results.get('overall_compliance')
            if analysis_mode == "observational_only":
                overall_compliance_value = None
            
            # FIX: Force overallCompliance = null if criticalSignalFailure = true
            if critical_signal_failure:
                overall_compliance_value = None
                logger.info("[Schema] Forced overallCompliance = null due to criticalSignalFailure")
            
            # FIX: Force overallCompliance = null if criticalSignalFailure = true (post-processor should have done this, but ensure it)
            if critical_signal_failure:
                overall_compliance_value = None
                logger.info("[Schema] Forced overallCompliance = null due to criticalSignalFailure")
            
            # FIX: Calculate bucket (approved/review/reject/unknown)
            if critical_signal_failure:
                bucket = "unknown"
                overall_compliance_value = None  # FIX: Only null if true critical failure (ALL failed AND image load failed)
            elif overall_compliance_value is None or overall_compliance_value == "UNKNOWN":
                bucket = "unknown"
                overall_compliance_value = None
            elif isinstance(overall_compliance_value, (int, float)):
                if overall_compliance_value >= 85:
                    bucket = "approved"  # FIX: Match golden schema: "approved" not "approve"
                elif overall_compliance_value >= 60:
                    bucket = "review"
                else:
                    bucket = "reject"
            else:
                bucket = "unknown"
                overall_compliance_value = 0
            
            # FIX: Build transformed schema matching golden schema exactly
            transformed = {
                'overallCompliance': overall_compliance_value,  # FIX: Only null if true critical failure
                'bucket': bucket,
                'logo': logo_golden,  # FIX: {detected, zone, violations}
                'colors': colors_golden,  # FIX: {dominant, status}
                'criticalSignalFailure': critical_signal_failure,  # FIX: Only true if ALL analyzers failed AND image load failed
                'summary': results.get('summary', ''),
                'recommendations': recommendations,
                # Keep detailed formats for compatibility
                'analysisType': 'image_analysis',
                'analysis_type': 'image_analysis',
                'logoAnalysis': logo_analysis_transformed,
                'copywritingAnalysis': copywriting_analysis_transformed,
                'colorAnalysis': color_analysis_transformed,
                'typographyAnalysis': typography_analysis_transformed,  # 🔥 ALWAYS include typography
            }
            
            # Preserve all additional metadata from original results
            for key in ['analysis_id', 'timestamp', 'source_type', 'input_source']:
                if key in results:
                    # Convert snake_case to camelCase for the camelCase version
                    camel_key = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split('_')))
                    camel_key = camel_key[0].lower() + camel_key[1:] if camel_key else camel_key
                    transformed[camel_key] = results[key]
                    transformed[key] = results[key]  # Keep original too
            
            # Verify all required fields are present
            assert 'logoAnalysis' in transformed, "logoAnalysis missing from transformed schema"
            assert 'status' in transformed['logoAnalysis'], "logoAnalysis.status missing"
            assert 'detections' in transformed['logoAnalysis'], "logoAnalysis.detections missing"
            assert 'copywritingAnalysis' in transformed, "copywritingAnalysis missing from transformed schema"
            assert 'status' in transformed['copywritingAnalysis'], "copywritingAnalysis.status missing"
            assert 'issues' in transformed['copywritingAnalysis'], "copywritingAnalysis.issues missing"
            assert 'colorAnalysis' in transformed, "colorAnalysis missing from transformed schema"
            assert 'status' in transformed['colorAnalysis'], "colorAnalysis.status missing"
            assert 'dominantColors' in transformed['colorAnalysis'], "colorAnalysis.dominantColors missing"
            assert 'brandValidation' in transformed['colorAnalysis'], "colorAnalysis.brandValidation missing"
            
            logger.info(f"[Schema] ✅ Transformed to target schema: analysisType={transformed.get('analysisType')}, "
                       f"criticalSignalFailure={transformed.get('criticalSignalFailure')}, "
                       f"logoAnalysis.status={transformed['logoAnalysis'].get('status')}, "
                       f"logoAnalysis.detections.count={len(transformed['logoAnalysis'].get('detections', []))}, "
                       f"copywritingAnalysis.status={transformed['copywritingAnalysis'].get('status')}, "
                      f"copywritingAnalysis.issues.count={len(transformed['copywritingAnalysis'].get('issues', []))}, "
                       f"colorAnalysis.status={transformed['colorAnalysis'].get('status')}, "
                       f"colorAnalysis.dominantColors.count={len(transformed['colorAnalysis'].get('dominantColors', []))}, "
                       f"colorAnalysis.brandValidation.complianceScore={transformed['colorAnalysis'].get('brandValidation', {}).get('complianceScore')}")
            
            return transformed
            
        except Exception as e:
            logger.error(f"Schema transformation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original results if transformation fails
            return results
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get status of a specific analysis"""
        try:
            if analysis_id in self.analysis_results:
                return {
                    'analysis_id': analysis_id,
                    'status': 'completed',
                    'results': self.analysis_results[analysis_id]
                }
            else:
                return {'error': 'Analysis not found'}
                
        except Exception as e:
            logger.error(f"Failed to get analysis status: {e}")
            return {'error': f'Failed to get analysis status: {str(e)}'}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear analysis results
            self.analysis_results.clear()
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

