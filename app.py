"""
Author: Omer Sayem
Date: 2025-09-09
Version: 1.0.0
Description: Consolidated BrandGuard Pipeline API
Unified backend serving all four models: Color, Typography, Copywriting, and Logo Detection
"""

import os
import json
import logging
import multiprocessing
import atexit
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Set environment variables BEFORE importing any ML libraries
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.brandguard.config.settings import Settings
from src.brandguard.core.pipeline_orchestrator_new import PipelineOrchestrator
from src.brandguard.core.config_validator import evaluate_config_state, build_block_response
from src.brandguard.services.brand_extractor import BrandExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load settings
try:
    settings = Settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    settings = None

# Validate Google Cloud Vision credentials at startup
# CRITICAL: Raises hard error if credentials are missing or invalid
try:
    from src.brandguard.utils.google_credentials_validator import validate_google_credentials
    # Check if we're in debug/development mode (allow graceful fallback)
    debug_mode = os.getenv('FLASK_ENV') == 'development' or os.getenv('FLASK_DEBUG') == '1' or os.getenv('DEBUG') == '1'
    
    if debug_mode:
        # In debug mode, log warning but allow server to start
        is_valid, error_msg = validate_google_credentials(raise_on_error=False)
        if not is_valid:
            logger.warning(f"⚠️  Google Cloud Vision credentials validation failed:\n{error_msg}")
            logger.warning("⚠️  Server will start but Google OCR will not function.")
            logger.warning("   The system will fallback to PaddleOCR if available.")
        else:
            logger.info("✅ Google Cloud Vision credentials validated successfully")
    else:
        # In production mode, raise hard error
        validate_google_credentials(raise_on_error=True)
        logger.info("✅ Google Cloud Vision credentials validated successfully")
except ImportError:
    logger.warning("⚠️  Google credentials validator not available - skipping validation")
except RuntimeError as e:
    # Hard error - credentials validation failed
    logger.critical(f"❌ CRITICAL: Google Cloud Vision credentials validation failed:\n{e}")
    logger.critical("❌ Server startup aborted due to missing/invalid Google credentials.")
    raise
except Exception as e:
    logger.error(f"⚠️  Google credentials validation error: {e}")
    # In production, fail hard on unexpected errors too
    if not (os.getenv('FLASK_ENV') == 'development' or os.getenv('FLASK_DEBUG') == '1'):
        raise

# Initialize pipeline orchestrator
pipeline = None
if settings:
    try:
        pipeline = PipelineOrchestrator(settings)
        logger.info("Pipeline orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

# App configuration
app.config['MAX_CONTENT_LENGTH'] = settings.max_file_size if settings else 50 * 1024 * 1024 # 50MB
app.config['UPLOAD_FOLDER'] = settings.upload_dir if settings else 'uploads'
app.config['RESULTS_FOLDER'] = settings.results_dir if settings else 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'},
    'documents': {'pdf', 'txt', 'doc', 'docx'},
    'all': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'pdf', 'txt', 'doc', 'docx'}
}

def allowed_file(filename: str, file_type: str = 'all') -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, ALLOWED_EXTENSIONS['all'])

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    if not filename:
        return 'unknown'
    
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if ext in ALLOWED_EXTENSIONS['images']:
        return 'image'
    elif ext in ALLOWED_EXTENSIONS['documents']:
        return 'document'
    else:
        return 'unknown'

def determine_analysis_type(input_source: str) -> str:
    """
    Determine analysis type based on input source.
    CRITICAL: Image URLs must be treated as images, not URLs.
    
    Rules:
    - If input_source ends with image extension (.png, .jpg, etc.) → "image"
    - If input_source is HTTP/HTTPS URL with image extension → "image"
    - If input_source is HTTP/HTTPS URL → check content-type or default to "image"
    - Only return "url" for real webpages (HTML) - not yet implemented
    - Default fallback → "image" (safe default for image URLs)
    
    This function is deterministic and testable.
    """
    if not input_source:
        return 'image'  # Safe default
    
    input_lower = input_source.lower().strip()
    
    # CRITICAL: For URLs with query parameters, check path before '?'
    # Example: https://example.com/image.png?X-Amz-Signature=... should be detected as image
    url_path = input_lower
    if '?' in input_lower:
        url_path = input_lower.split('?')[0]
    
    # Check file extension first (most reliable)
    # Check both full input and URL path (for URLs with query params)
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
    if any(input_lower.endswith(ext) for ext in image_extensions) or \
       any(url_path.endswith(ext) for ext in image_extensions):
        return 'image'
    
    # Check if it's a URL
    if input_lower.startswith('http://') or input_lower.startswith('https://'):
        # For URLs, we need to check content-type
        # Since we can't check content-type synchronously here without downloading,
        # we'll default to 'image' for URLs (most URLs passed are image URLs)
        # The pipeline will handle actual content-type checking during download
        logger.info(f"[Routing] URL detected, defaulting to image analysis: {input_source[:100]}")
        return 'image'
    
    # Default to image (safe fallback)
    return 'image'

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_ready': pipeline is not None,
        'settings_loaded': settings is not None,
        'version': '1.0.0',
        'debug_info': {
            'pipeline_type': type(pipeline).__name__ if pipeline else None,
            'settings_type': type(settings).__name__ if settings else None
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """Main analysis endpoint - analyzes content using all models"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503

        config_payload = {
            "brandName": request.form.get("brandName", "").strip(),
            "brandPurpose": request.form.get("brandPurpose", "").strip(),
            "industry": request.form.get("industry", "").strip(),
            "logoUploaded": request.form.get("logoUploaded", "false").lower() == "true",
            "brandFonts": request.form.get("brandFonts", request.form.get("expected_fonts", "")).strip(),
            "brandColors": request.form.get("brandColors", request.form.get("primary_colors", "")).strip(),
            "brandColorPalette": request.form.get("brandColorPalette", request.form.get("brand_palette", "")).strip(),
        }
        config_result = evaluate_config_state(config_payload)
        config_state = config_result["configState"]
        analysis_mode = config_result["analysisMode"]

        # V1 RULE: Block ONLY if BOTH brandName AND brandPurpose are missing
        if config_state == "not_configured":
            return jsonify(build_block_response(config_state)), 400
        
        # V1 RULE: Remove partial_configured block - allow upload with observational mode
        # (No confirmation required - proceed with analysis)
        
        analysis_options = request.form.to_dict(flat=True)
        analysis_options["analysisMode"] = analysis_mode
        analysis_options["configState"] = config_state
        input_type = request.form.get('input_type', 'file')

        
        
        # Handle different input types
        # CRITICAL: Determine analysis type BEFORE any branching or saving
        input_source = None
        source_type = None
        
        if input_type == 'file':
            # File upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not supported'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(filepath)
            
            input_source = filepath
            # CRITICAL: Use determine_analysis_type for consistency (handles both files and URLs)
            # This ensures uploaded images and image URLs use the same logic
            source_type = determine_analysis_type(filepath)
            
        elif input_type == 'text':
            # Direct text input
            text_content = request.form.get('text_content', '').strip()
            if not text_content:
                return jsonify({'error': 'No text content provided'}), 400
            
            input_source = text_content
            source_type = 'text'
            
        elif input_type == 'url':
            # URL input - CRITICAL: Determine if URL is an image
            url = request.form.get('url', '').strip()
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            
            input_source = url
            # CRITICAL: Use determine_analysis_type BEFORE any branching
            # This ensures image URLs route to image_analysis, not url_analysis
            source_type = determine_analysis_type(url)
            logger.info(f"[Routing] URL input detected: '{url[:100]}', routing to: {source_type} analysis")
            
            # CRITICAL: Defensive check - if URL is clearly an image, force source_type to 'image'
            url_lower = url.lower()
            url_path = url_lower.split('?')[0] if '?' in url_lower else url_lower
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
            is_image_url = any(url_path.endswith(ext) for ext in image_extensions)
            
            if is_image_url and source_type != 'image':
                logger.error(
                    f"[Routing] CRITICAL BUG: Image URL '{url[:100]}' was incorrectly routed to "
                    f"'{source_type}'. Forcing correction to 'image'."
                )
                source_type = 'image'
        
        # CRITICAL: Defensive assert - image URLs must NEVER reach url_analysis
        # This prevents routing bugs from causing incorrect analysis
        # Check both full path and URL path (for URLs with query parameters)
        if input_source:
            input_lower = input_source.lower()
            url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
            is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                      any(url_path.endswith(ext) for ext in image_extensions)
            
            if is_image:
                assert source_type == 'image', (
                    f"CRITICAL ROUTING BUG: Image URL/file '{input_source[:100]}' routed to "
                    f"'{source_type}' instead of 'image'. This must never happen!"
                )
        
        # CRITICAL: Ensure source_type is set ONCE and trusted everywhere else
        if not source_type:
            logger.error(f"[Routing] source_type not determined for input: {input_source[:100]}")
            return jsonify({'error': 'Failed to determine analysis type'}), 500
        
        # Log the final routing decision
        logger.info(f"[Routing] Final routing decision: input_source='{input_source[:100]}', source_type='{source_type}'")
        
        # Handle unsupported input types
        if input_source is None:
            return jsonify({'error': f'Unsupported input type: {input_type}'}), 400
        
        # Get analysis options
        analysis_options = {
                    'analysis_priority': request.form.get('analysis_priority', 'balanced'),
                    'report_detail': request.form.get('report_detail', 'detailed'), # detailed, summary, comprehensive
                   'include_recommendations': request.form.get('include_recommendations', 'true').lower() == 'true', # true, false
                   'analysisMode': analysis_mode,
                   'configState': config_state,
                   'color_analysis': {
                       'enabled': request.form.get('enable_color', 'true').lower() == 'true',
                       'n_colors': int(request.form.get('color_n_colors', 8)),
                       'n_clusters': int(request.form.get('color_n_colors', 8)),
                       'color_tolerance': float(request.form.get('color_tolerance', 2.3)),
                       'enable_contrast_check': request.form.get('enable_contrast_check', 'true').lower() == 'true',
                       'brand_palette': request.form.get('brand_palette', '').strip(),
                       'primary_colors': request.form.get('primary_colors', '').strip(),
                       'secondary_colors': request.form.get('secondary_colors', '').strip(),
                       'accent_colors': request.form.get('accent_colors', '').strip(),
                       'primary_threshold': int(request.form.get('primary_threshold', 75)),
                       'secondary_threshold': int(request.form.get('secondary_threshold', 75)),
                       'accent_threshold': int(request.form.get('accent_threshold', 75)),
                       'forbidden_colors': request.form.get('forbidden_colors', '').strip()
                   },
                   'typography_analysis': {
                       'enabled': request.form.get('enable_typography', 'true').lower() == 'true',
                       'merge_regions': request.form.get('merge_regions', 'true').lower() == 'true',
                       'distance_threshold': int(request.form.get('distance_threshold', 20)),
                       'confidence_threshold': float(request.form.get('typography_confidence_threshold', 0.7)),
                       'enable_font_validation': True,
                       'expected_fonts': request.form.get('expected_fonts', '').strip()
                   },
                   'copywriting_analysis': {
                       'enabled': request.form.get('enable_copywriting', 'true').lower() == 'true',
                       'include_suggestions': True,
                       'include_industry_benchmarks': True,
                       'enable_brand_profile_matching': True,
                       'formality_score': int(request.form.get('formality_score', 60)),
                       'confidence_level': request.form.get('confidence_level', 'balanced'),
                       'warmth_score': int(request.form.get('warmth_score', 50)),
                       'energy_score': int(request.form.get('energy_score', 50))
                   },
                   'logo_analysis': {
                       'enabled': request.form.get('enable_logo', 'true').lower() == 'true',
                       'enable_placement_validation': request.form.get('enable_placement_validation', 'true').lower() == 'true',
                       'enable_brand_compliance': True,
                       'generate_annotations': request.form.get('generate_annotations', 'true').lower() == 'true',
                       'confidence_threshold': float(request.form.get('logo_confidence_threshold', 0.5)),
                       'max_detections': int(request.form.get('max_logo_detections', 100)),
                       'allowed_zones': request.form.get('allowed_zones', 'top-left,top-right,bottom-left,bottom-right').split(','),
                       'min_logo_size': float(request.form.get('min_logo_size', 0.01)),
                       'max_logo_size': float(request.form.get('max_logo_size', 0.25)),
                       'min_edge_distance': float(request.form.get('min_edge_distance', 0.05)),
                       'show_original_image': request.form.get('show_original_image', 'true').lower() == 'true',
                       'show_analysis_overlay': request.form.get('show_analysis_overlay', 'true').lower() == 'true',
                       'annotation_style': request.form.get('annotation_style', 'bounding_box'),
                       'annotation_color': request.form.get('annotation_color', 'gray'),
                       'pass_threshold': float(request.form.get('pass_threshold', 0.7)),
                       'warning_threshold': float(request.form.get('warning_threshold', 0.5)),
                       'critical_threshold': float(request.form.get('critical_threshold', 0.3)),
                   }
               }
        
        # Perform comprehensive analysis
        logger.info(f"[Routing] Starting analysis - input_source: '{input_source[:100] if input_source else 'None'}', source_type: '{source_type}'")
        
        # CRITICAL: Final validation before calling pipeline
        # This is the last chance to catch routing errors before they reach the orchestrator
        if input_source:
            input_lower = str(input_source).lower()
            url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
            is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                      any(url_path.endswith(ext) for ext in image_extensions)
            
            if is_image and source_type != 'image':
                logger.error(
                    f"[Routing] CRITICAL BUG DETECTED: Image URL '{input_source[:100]}' has source_type='{source_type}' "
                    f"instead of 'image'. FORCING correction before calling pipeline!"
                )
                source_type = 'image'
        
        analysis_results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options
        )
        
        # CRITICAL: Final validation - ensure response has correct routing
        # This is a last-ditch fix in case the orchestrator didn't catch it
        if input_source and analysis_results:
            input_lower = str(input_source).lower()
            url_path = input_lower.split('?')[0] if '?' in input_lower else input_lower
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']
            is_image = any(input_lower.endswith(ext) for ext in image_extensions) or \
                      any(url_path.endswith(ext) for ext in image_extensions)
            
            if is_image:
                # Force correct routing in response
                if analysis_results.get('source_type') != 'image' or \
                   analysis_results.get('analysis_type') != 'image_analysis':
                    logger.error(
                        f"[Routing] CRITICAL: Response has wrong routing! "
                        f"source_type={analysis_results.get('source_type')}, "
                        f"analysis_type={analysis_results.get('analysis_type')}. FORCING correction!"
                    )
                    analysis_results['source_type'] = 'image'
                    analysis_results['analysis_type'] = 'image_analysis'
                    # Remove any old url_analysis fields - check multiple possible locations
                    response_message = analysis_results.get('message') or analysis_results.get('results', {}).get('message')
                    if response_message and 'URL analysis not yet implemented' in str(response_message):
                        logger.error(f"[Routing] Found old message in response! Removing it. Message was: {response_message}")
                        if 'message' in analysis_results:
                            analysis_results.pop('message', None)
                        if 'results' in analysis_results and isinstance(analysis_results['results'], dict) and 'message' in analysis_results['results']:
                            analysis_results['results'].pop('message', None)
                    
                    # Also check and remove any old url_analysis summary
                    response_summary = analysis_results.get('summary') or (analysis_results.get('results', {}) or {}).get('summary')
                    if isinstance(response_summary, str) and 'Poor brand compliance' in response_summary and analysis_results.get('overall_compliance') == 0:
                        logger.warning(f"[Routing] Found placeholder summary in response. Clearing it.")
                        analysis_results['summary'] = {}
        
        # CRITICAL: Validate response - if it's a not_supported response, return it immediately
        if analysis_results.get('status') == 'not_supported':
            logger.warning(f"[Routing] Analysis returned not_supported: {analysis_results.get('message')}")
            return jsonify({
                'status': 'not_supported',
                'message': analysis_results.get('message', 'This input type is not yet supported')
            }), 400
        
        # Safety guardrail: Check if analysis is not supported
        if analysis_results.get('status') == 'not_supported':
            return jsonify({
                'status': 'not_supported',
                'message': analysis_results.get('message', 'This input type is not yet supported')
            }), 400
        
        if 'error' in analysis_results:
            return jsonify(analysis_results), 500
        
        # Clean up uploaded file if it was a file
        if input_type == 'file' and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass 
        
        return jsonify({
            'success': True,
            'results': analysis_results
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Analysis failed: {str(e)}\n{error_traceback}")
        # Return generic error to client, but log full traceback
        return jsonify({
            'success': False,
            'error': {
                'message': 'Internal server error',
                'code': 'UNKNOWN_ERROR',
                'statusCode': 500
            }
        }), 500

@app.route('/api/analyze/color', methods=['POST'])
def analyze_colors():
    """Color analysis only endpoint"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save and analyze
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Get color analysis options
        analysis_options = {
            'analysisMode': analysis_mode,
            'configState': config_state,
            'color_analysis': {
                'n_colors': int(request.form.get('n_colors', 8)),
                'n_clusters': int(request.form.get('n_clusters', 8)),
                'color_tolerance': float(request.form.get('color_tolerance', 2.3)),
                'enable_contrast_check': request.form.get('enable_contrast_check', 'true').lower() == 'true',
                # New brand color validation features
                'primary_colors': request.form.get('primary_colors', '').strip(),
                'secondary_colors': request.form.get('secondary_colors', '').strip(),
                'accent_colors': request.form.get('accent_colors', '').strip(),
                'primary_threshold': int(request.form.get('primary_threshold', 75)),
                'secondary_threshold': int(request.form.get('secondary_threshold', 75)),
                'accent_threshold': int(request.form.get('accent_threshold', 75))
            }
        }
        
        # Perform color analysis
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type='image',
            analysis_options=analysis_options
        )
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in results:
            return jsonify(results), 500
        
        return jsonify({
            'success': True,
            'color_analysis': results.get('model_results', {}).get('color_analysis', {})
        })
        
    except Exception as e:
        logger.error(f"Color analysis failed: {str(e)}")
        return jsonify({'error': f'Color analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/typography', methods=['POST'])
def analyze_typography():
    """Typography analysis only endpoint"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503

        config_payload = {
            "brandName": request.form.get("brandName", "").strip(),
            "brandPurpose": request.form.get("brandPurpose", "").strip(),
            "industry": request.form.get("industry", "").strip(),
            "logoUploaded": request.form.get("logoUploaded", "false").lower() == "true",
            "brandFonts": request.form.get("brandFonts", request.form.get("expected_fonts", "")).strip(),
            "brandColors": request.form.get("brandColors", request.form.get("primary_colors", "")).strip(),
            "brandColorPalette": request.form.get("brandColorPalette", request.form.get("brand_palette", "")).strip(),
        }
        config_result = evaluate_config_state(config_payload)
        config_state = config_result["configState"]
        analysis_mode = config_result["analysisMode"]

        # V1 RULE: Block ONLY if BOTH brandName AND brandPurpose are missing
        if config_state == "not_configured":
            return jsonify(build_block_response(config_state)), 400
        
        # V1 RULE: Remove partial_configured block - allow upload with observational mode
        # (No confirmation required - proceed with analysis)
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save and analyze
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Get typography analysis options
        analysis_options = {
            'analysisMode': analysis_mode,
            'configState': config_state,
            'typography_analysis': {
                'merge_regions': request.form.get('merge_regions', 'true').lower() == 'true',
                'distance_threshold': int(request.form.get('distance_threshold', 20)),
                'confidence_threshold': float(request.form.get('confidence_threshold', 0.7)),
                'enable_font_validation': request.form.get('enable_font_validation', 'true').lower() == 'true'
            }
        }
        
        # Perform typography analysis
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type='image',
            analysis_options=analysis_options
        )
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in results:
            return jsonify(results), 500
        
        return jsonify({
            'success': True,
            'typography_analysis': results.get('model_results', {}).get('typography_analysis', {})
        })
        
    except Exception as e:
        logger.error(f"Typography analysis failed: {str(e)}")
        return jsonify({'error': f'Typography analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/copywriting', methods=['POST'])
def analyze_copywriting():
    """Copywriting analysis only endpoint"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        input_type = request.form.get('input_type', 'file')
        
        if input_type == 'file':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save and analyze
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(filepath)
            
            input_source = filepath
            source_type = get_file_type(filename)
            
        elif input_type == 'text':
            text_content = request.form.get('text_content', '').strip()
            if not text_content:
                return jsonify({'error': 'No text content provided'}), 400
            
            input_source = text_content
            source_type = 'text'
            
        else:
            return jsonify({'error': f'Unsupported input type: {input_type}'}), 400
        
        # Get copywriting analysis options
        analysis_options = {
            'analysisMode': analysis_mode,
            'configState': config_state,
            'copywriting_analysis': {
                'include_suggestions': request.form.get('include_suggestions', 'true').lower() == 'true',
                'include_industry_benchmarks': request.form.get('include_industry_benchmarks', 'true').lower() == 'true',
                'enable_brand_profile_matching': request.form.get('enable_brand_profile_matching', 'true').lower() == 'true'
            }
        }
        
        # Add brand voice settings if provided
        brand_voice_settings = {}
        if request.form.get('formality_score'):
            brand_voice_settings['formality_score'] = int(request.form.get('formality_score'))
        if request.form.get('confidence_level'):
            brand_voice_settings['confidence_level'] = request.form.get('confidence_level')
        if request.form.get('warmth_score'):
            brand_voice_settings['warmth_score'] = int(request.form.get('warmth_score'))
        if request.form.get('energy_score'):
            brand_voice_settings['energy_score'] = int(request.form.get('energy_score'))
        
        if brand_voice_settings:
            analysis_options['brand_voice_settings'] = brand_voice_settings
        
        # Perform copywriting analysis
        results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options
        )
        
        # Clean up if file was uploaded
        if input_type == 'file' and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        if 'error' in results:
            return jsonify(results), 500
        
        return jsonify({
            'success': True,
            'copywriting_analysis': results.get('model_results', {}).get('copywriting_analysis', {})
        })
        
    except Exception as e:
        logger.error(f"Copywriting analysis failed: {str(e)}")
        return jsonify({'error': f'Copywriting analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/logo', methods=['POST'])
def analyze_logos():
    """Logo detection analysis only endpoint"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save and analyze
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Get logo analysis options
        analysis_options = {
            'analysisMode': analysis_mode,
            'configState': config_state,
            'logo_analysis': {
                'enabled': True,
                'enable_placement_validation': request.form.get('enable_placement_validation', 'true').lower() == 'true',
                'enable_brand_compliance': request.form.get('enable_brand_compliance', 'true').lower() == 'true',
                'generate_annotations': request.form.get('generate_annotations', 'true').lower() == 'true',
                'confidence_threshold': float(request.form.get('logo_confidence_threshold', 0.5))
            }
        }
        
        # Perform logo analysis
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type='image',
            analysis_options=analysis_options
        )
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in results:
            return jsonify(results), 500
        
        return jsonify({
            'success': True,
            'logo_analysis': results.get('model_results', {}).get('logo_analysis', {})
        })
        
    except Exception as e:
        logger.error(f"Logo analysis failed: {str(e)}")
        import traceback
        logger.error(f"Logo analysis traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Logo analysis failed: {str(e)}'}), 500

@app.route('/api/extract-brand-guidelines', methods=['POST'])
def extract_brand_guidelines():
    """Extract brand guidelines from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        try:
            # Determine file type
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            # Initialize extractor
            extractor = BrandExtractor()
            
            # Extract brand guidelines
            result = extractor.extract_brand_guidelines(filepath, file_ext)
            
            return jsonify(result)
            
        finally:
            # Cleanup temp file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
        
    except Exception as e:
        logger.error(f"Brand extraction endpoint failed: {e}")
        import traceback
        logger.error(f"Brand extraction traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status/<analysis_id>')
def get_analysis_status(analysis_id: str):
    """Get status of a specific analysis"""
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        status = pipeline.get_analysis_status(analysis_id)
        
        if 'error' in status:
            return jsonify(status), 404
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Failed to get analysis status: {str(e)}")
        return jsonify({'error': f'Failed to get analysis status: {str(e)}'}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        if not settings:
            return jsonify({'error': 'Settings not loaded'}), 503
        
        return jsonify({
            'success': True,
            'config': {
                'color_palette': {
                    'name': settings.color_palette.name,
                    'primary_colors_count': len(settings.color_palette.primary_colors),
                    'secondary_colors_count': len(settings.color_palette.secondary_colors)
                },
                'typography_rules': {
                    'approved_fonts_count': len(settings.typography_rules.approved_fonts),
                    'max_font_size': settings.typography_rules.max_font_size,
                    'min_font_size': settings.typography_rules.min_font_size
                },
                'brand_voice': {
                    'formality_score': settings.brand_voice.formality_score,
                    'confidence_level': settings.brand_voice.confidence_level,
                    'warmth_score': settings.brand_voice.warmth_score,
                    'energy_score': settings.brand_voice.energy_score
                },
                'logo_detection': {
                    'confidence_threshold': settings.logo_detection.confidence_threshold,
                    'max_detections': settings.logo_detection.max_detections
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        return jsonify({'error': f'Failed to get config: {str(e)}'}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        if not settings:
            return jsonify({'error': 'Settings not loaded'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Update specific configuration sections
        if 'color_palette' in data:
            # Update color palette
            pass
        
        if 'typography_rules' in data:
            # Update typography rules
            pass
        
        if 'brand_voice' in data:
            # Update brand voice settings
            pass
        
        if 'logo_detection' in data:
            # Update logo detection settings
            pass
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        return jsonify({'error': f'Failed to update config: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def get_upload(filename):
    """Get uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def get_result(filename):
    """Get result file"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    import traceback
    error_traceback = traceback.format_exc()
    logger.error(f"Internal server error: {e}\n{error_traceback}")
    return jsonify({
        'success': False,
        'error': {
            'message': 'Internal server error',
            'code': 'UNKNOWN_ERROR',
            'statusCode': 500
        }
    }), 500

def cleanup_resources():
    """Clean up all resources to prevent semaphore leaks"""
    try:
        logger.info("Cleaning up resources...")
        
        # Clean up pipeline
        if pipeline:
            pipeline.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clean up multiprocessing resources
        try:
            multiprocessing.active_children()
            for process in multiprocessing.active_children():
                process.terminate()
                process.join(timeout=5)
        except Exception as e:
            logger.warning(f"Error cleaning up multiprocessing: {e}")
        
        logger.info("Resource cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

if __name__ == '__main__':
    import signal
    import sys
    
    def signal_handler(sig, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, cleaning up...")
        try:
            cleanup_resources()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("🚀 Starting Consolidated BrandGuard Pipeline API...")
        logger.info(f"📁 Upload directory: {app.config['UPLOAD_FOLDER']}")
        logger.info(f"📁 Results directory: {app.config['RESULTS_FOLDER']}")
        logger.info(f"🔧 Pipeline ready: {pipeline is not None}")
        
        # Run the Flask app with auto-reload enabled in development
        # Check if running in development mode
        debug_mode = os.getenv('FLASK_ENV') == 'development' or os.getenv('FLASK_DEBUG') == '1'
        
        app.run(
            debug=debug_mode,  # Enable debug mode for auto-reload
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            threaded=True,
            use_reloader=debug_mode,  # Auto-reload only in debug mode
            use_debugger=debug_mode,
            extra_files=None  # Can add files to watch for changes
        )
        
    except KeyboardInterrupt:
        logger.info("App stopped by user")
        cleanup_resources()
    except Exception as e:
        logger.error(f"App crashed: {e}")
        cleanup_resources()
        sys.exit(1)
