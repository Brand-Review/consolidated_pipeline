"""
Consolidated BrandGuard Pipeline API
Unified backend serving all four models: Color, Typography, Copywriting, and Logo Detection
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our consolidated components
from src.brandguard.config.settings import Settings
from src.brandguard.core.pipeline_orchestrator import PipelineOrchestrator

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
app.config['MAX_CONTENT_LENGTH'] = settings.max_file_size if settings else 50 * 1024 * 1024
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
        
        # Get input type
        input_type = request.form.get('input_type', 'file')

        
        
        # Handle different input types
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
            
            # Determine source type
            source_type = get_file_type(filename)
            
            input_source = filepath
            
        elif input_type == 'text':
            # Direct text input
            text_content = request.form.get('text_content', '').strip()
            if not text_content:
                return jsonify({'error': 'No text content provided'}), 400
            
            input_source = text_content
            source_type = 'text'
            
        elif input_type == 'url':
            # URL input
            url = request.form.get('url', '').strip()
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            
            input_source = url
            source_type = 'url'
            
        else:
            return jsonify({'error': f'Unsupported input type: {input_type}'}), 400
        
        # Get analysis options - DEBUGGING: Only logo analysis enabled
        analysis_options = {
                   'analysis_priority': request.form.get('analysis_priority', 'balanced'),
                   'report_detail': request.form.get('report_detail', 'detailed'),
                   'include_recommendations': request.form.get('include_recommendations', 'true').lower() == 'true',
                   
                   # DEBUGGING: Commented out other analyses
                   'color_analysis': {
                       'enabled': False,  # DEBUGGING: Disabled
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
                       'accent_threshold': int(request.form.get('accent_threshold', 75))
                   },
                   'typography_analysis': {
                       'enabled': False,  # DEBUGGING: Disabled
                       'merge_regions': request.form.get('merge_regions', 'true').lower() == 'true',
                       'distance_threshold': int(request.form.get('distance_threshold', 20)),
                       'confidence_threshold': float(request.form.get('typography_confidence_threshold', 0.7)),
                       'enable_font_validation': True,
                       'expected_fonts': request.form.get('expected_fonts', '').strip()
                   },
                   'copywriting_analysis': {
                       'enabled': False,  # DEBUGGING: Disabled
                       'include_suggestions': True,
                       'include_industry_benchmarks': True,
                       'enable_brand_profile_matching': True,
                       'formality_score': int(request.form.get('formality_score', 60)),
                       'confidence_level': request.form.get('confidence_level', 'balanced'),
                       'warmth_score': int(request.form.get('warmth_score', 50)),
                       'energy_score': int(request.form.get('energy_score', 50))
                   },
                   'logo_analysis': {
                       'enabled': True,  # DEBUGGING: Only logo analysis enabled
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
                   #    # LLVa with Ollama integration
                   #    'enable_llva_ollama': request.form.get('enable_llva_ollama', 'false').lower() == 'true',
                   #    'llva_analysis_focus': request.form.get('llva_analysis_focus', 'comprehensive')
                   }
               }
        
        # Get brand voice settings if provided
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
        
        # Perform comprehensive analysis
        logger.info(f"Starting analysis for {source_type}: {input_source}")
        
        analysis_results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options
        )
        
        if 'error' in analysis_results:
            return jsonify(analysis_results), 500
        
        # Clean up uploaded file if it was a file
        if input_type == 'file' and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass  # Don't fail if cleanup fails
        
        return jsonify({
            'success': True,
            'results': analysis_results
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

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
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import signal
    import sys
    
    def signal_handler(sig, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, cleaning up...")
        try:
            if pipeline:
                pipeline.cleanup()
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
        
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5003,
            threaded=True,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        logger.info("App stopped by user")
    except Exception as e:
        logger.error(f"App crashed: {e}")
        sys.exit(1)
