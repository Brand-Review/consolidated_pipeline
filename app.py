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

# Load .env before anything reads os.environ (including ML libs and Settings)
from dotenv import load_dotenv
load_dotenv()

import yaml

# Set environment variables BEFORE importing any ML libraries
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import threading
import requests as http_requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.brandguard.config.settings import Settings
from src.brandguard.core.pipeline_orchestrator_new import PipelineOrchestrator
from src.brandguard.brand_profile.brand_store import BrandStore
from src.brandguard.brand_profile.pdf_extractor import PDFRuleExtractor
from src.brandguard.brand_profile.text_rag import TextRAG
from src.brandguard.brand_profile.asset_rag import AssetRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Internal webhook secret — must match br-be INTERNAL_WEBHOOK_SECRET
_INTERNAL_SECRET = os.environ.get('INTERNAL_WEBHOOK_SECRET', 'internal-brandguard-secret')

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

# Brand profile services (lazy, shared across requests)
brand_store = BrandStore()
text_rag = TextRAG()
asset_rag = AssetRAG()
pdf_extractor = PDFRuleExtractor()

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

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

_DOCS_DIR = Path(__file__).resolve().parent / 'docs'


@app.route('/api/openapi.json')
def openapi_spec_json():
    """Machine-readable OpenAPI 3.1 document for this Flask API."""
    spec_file = _DOCS_DIR / 'openapi.yaml'
    if not spec_file.is_file():
        return jsonify({'error': 'OpenAPI specification not found'}), 404
    with open(spec_file, encoding='utf-8') as f:
        return jsonify(yaml.safe_load(f))


@app.route('/api/docs')
def api_docs_swagger_ui():
    """Interactive API documentation (Swagger UI)."""
    return render_template('api_docs.html')


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

def _build_analysis_options(form_data: dict) -> dict:
    """Build analysis_options dict from a flat form-data dict."""
    def _bool(key, default='true'):
        return str(form_data.get(key, default)).lower() == 'true'
    def _int(key, default):
        try:
            return int(form_data.get(key, default))
        except (TypeError, ValueError):
            return int(default)
    def _float(key, default):
        try:
            return float(form_data.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    # scoring_weights arrives as a JSON string from br-be case-converter
    scoring_weights_raw = form_data.get('scoring_weights', '')
    try:
        scoring_weights = json.loads(scoring_weights_raw) if scoring_weights_raw else {}
    except Exception:
        scoring_weights = {}

    return {
        'analysis_priority': form_data.get('analysis_priority', 'balanced'),
        'report_detail': form_data.get('report_detail', 'detailed'),
        'include_recommendations': _bool('include_recommendations'),
        'scoring_weights': scoring_weights,
        'pass_threshold': _float('pass_threshold', 0.70),
        'color_analysis': {
            'enabled': _bool('enable_color'),
            'n_colors': _int('color_n_colors', 8),
            'n_clusters': _int('color_n_colors', 8),
            'color_tolerance': _float('color_tolerance', 2.3),
            'enable_contrast_check': _bool('enable_contrast_check'),
            'brand_palette': form_data.get('brand_palette', '').strip(),
            'primary_colors': form_data.get('primary_colors', '').strip(),
            'secondary_colors': form_data.get('secondary_colors', '').strip(),
            'accent_colors': form_data.get('accent_colors', '').strip(),
            'primary_threshold': _int('primary_threshold', 75),
            'secondary_threshold': _int('secondary_threshold', 75),
            'accent_threshold': _int('accent_threshold', 75),
            'forbidden_colors': form_data.get('forbidden_colors', '').strip(),
        },
        'typography_analysis': {
            'enabled': _bool('enable_typography'),
            'merge_regions': _bool('merge_regions'),
            'distance_threshold': _int('distance_threshold', 20),
            'confidence_threshold': _float('typography_confidence_threshold', 0.7),
            'enable_font_validation': True,
            'expected_fonts': form_data.get('expected_fonts', '').strip(),
        },
        'copywriting_analysis': {
            'enabled': _bool('enable_copywriting'),
            'include_suggestions': True,
            'include_industry_benchmarks': True,
            'enable_brand_profile_matching': True,
            'formality_score': _int('formality_score', 60),
            'confidence_level': form_data.get('confidence_level', 'balanced'),
            'warmth_score': _int('warmth_score', 50),
            'energy_score': _int('energy_score', 50),
        },
        'logo_analysis': {
            'enabled': _bool('enable_logo'),
            'enable_placement_validation': _bool('enable_placement_validation'),
            'enable_brand_compliance': True,
            'generate_annotations': _bool('generate_annotations'),
            'confidence_threshold': _float('logo_confidence_threshold', 0.5),
            'max_detections': _int('max_logo_detections', 100),
            'allowed_zones': form_data.get('allowed_zones', 'top-left,top-right,bottom-left,bottom-right').split(','),
            'min_logo_size': _float('min_logo_size', 0.01),
            'max_logo_size': _float('max_logo_size', 0.25),
            'min_edge_distance': _float('min_edge_distance', 0.05),
            'show_original_image': _bool('show_original_image'),
            'show_analysis_overlay': _bool('show_analysis_overlay'),
            'annotation_style': form_data.get('annotation_style', 'bounding_box'),
            'annotation_color': form_data.get('annotation_color', 'gray'),
            'warning_threshold': _float('warning_threshold', 0.5),
            'critical_threshold': _float('critical_threshold', 0.3),
        },
    }


def _run_analysis_background(
    job_id: str,
    input_source: str,
    source_type: str,
    analysis_options: dict,
    brand_id: Optional[str],
    callback_url: Optional[str],
    cleanup_path: Optional[str],
):
    """Thread target: run the pipeline and POST results to callback_url."""
    try:
        logger.info(f"[{job_id}] Starting background analysis (source_type={source_type})")
        analysis_results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options,
            brand_id=brand_id,
        )

        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except Exception:
                pass

        if 'error' in analysis_results:
            payload: Dict[str, Any] = {'job_id': job_id, 'status': 'failed', 'error': analysis_results['error']}
        else:
            payload = {'job_id': job_id, 'status': 'completed', 'results': analysis_results}

    except Exception as e:
        logger.error(f"[{job_id}] Analysis background task failed: {e}")
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except Exception:
                pass
        payload = {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    if callback_url:
        try:
            http_requests.post(
                callback_url,
                json=payload,
                headers={'X-Internal-Secret': _INTERNAL_SECRET},
                timeout=30,
            )
            logger.info(f"[{job_id}] Callback sent to {callback_url}")
        except Exception as cb_err:
            logger.error(f"[{job_id}] Callback POST failed: {cb_err}")


@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """
    Async analysis endpoint.

    Returns {job_id, status: 'processing'} immediately.
    Runs analysis in a background thread, then POSTs results to callback_url
    (br-be /api/internal/analysis/callback/:analysisId).
    """
    import uuid

    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 503

    input_type = request.form.get('input_type', 'file')
    callback_url = request.form.get('callback_url', '').strip() or None
    job_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())

    filepath: Optional[str] = None

    if input_type == 'file':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(filepath)
        source_type = get_file_type(filename)
        input_source = filepath

    elif input_type == 'text':
        text_content = request.form.get('text_content', '').strip()
        if not text_content:
            return jsonify({'error': 'No text content provided'}), 400
        input_source = text_content
        source_type = 'text'

    elif input_type == 'url':
        url = request.form.get('url', '').strip()
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        input_source = url
        source_type = 'url'

    else:
        return jsonify({'error': f'Unsupported input type: {input_type}'}), 400

    form_data = request.form.to_dict()
    analysis_options = _build_analysis_options(form_data)
    brand_id = request.form.get('brand_id', '').strip() or None

    logger.info(f"[{job_id}] Queuing background analysis (source_type={source_type}, brand_id={brand_id})")

    thread = threading.Thread(
        target=_run_analysis_background,
        args=(job_id, input_source, source_type, analysis_options, brand_id, callback_url, filepath),
        daemon=True,
    )
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'processing'}), 202

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

@app.route('/api/brand/onboard', methods=['POST'])
def onboard_brand():
    """
    Brand onboarding endpoint.

    Form fields:
      - brand_name (str, required)
      - guideline_pdf (file, required)
      - approved_images (files[], optional)
      - rejected_images (files[], optional)
      - rejection_reasons (JSON string list matching rejected_images order, optional)

    Returns:
      { brand_id, brand_name, extracted_rules, chunks_indexed, assets_indexed }
    """
    try:
        brand_name = request.form.get('brand_name', '').strip()
        if not brand_name:
            return jsonify({'error': 'brand_name is required'}), 400

        # Optional: caller may supply its own brand_id (e.g. the MongoDB folder _id)
        # so that folder_id == brand_id across both services.
        brand_id_override = request.form.get('brand_id', '').strip() or None

        if 'guideline_pdf' not in request.files:
            return jsonify({'error': 'guideline_pdf is required'}), 400

        pdf_file = request.files['guideline_pdf']
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'guideline_pdf must be a PDF file'}), 400

        # Save PDF temporarily
        pdf_filename = secure_filename(pdf_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{pdf_filename}")
        pdf_file.save(pdf_path)

        try:
            # 1. Extract text chunks + structured rules from PDF
            logger.info(f"Extracting rules from PDF for brand: {brand_name}")
            structured_rules, chunks = pdf_extractor.extract(pdf_path)
        finally:
            try:
                os.remove(pdf_path)
            except Exception:
                pass

        # 2. Create brand profile in MongoDB
        brand_id = brand_store.create(brand_name, brand_id=brand_id_override)
        brand_store.update(brand_id, {
            "rules": structured_rules,
            "qdrant_guideline_collection": text_rag.collection_name(brand_id),
            "qdrant_asset_collection": asset_rag.collection_name(brand_id),
        })

        # 3. Index text chunks in Qdrant
        chunks_indexed = text_rag.index_chunks(brand_id, chunks)
        brand_store.update(brand_id, {"chunk_count": chunks_indexed})

        # 4. Index approved/rejected images
        approved_bytes = []
        for f in request.files.getlist('approved_images'):
            if f and f.filename:
                approved_bytes.append(f.read())

        rejection_reasons_json = request.form.get('rejection_reasons', '[]')
        try:
            all_reasons = json.loads(rejection_reasons_json)
        except Exception:
            all_reasons = []

        rejected_entries = []
        rejected_files = request.files.getlist('rejected_images')
        for i, f in enumerate(rejected_files):
            if f and f.filename:
                reasons = all_reasons[i] if i < len(all_reasons) else []
                if isinstance(reasons, str):
                    reasons = [reasons]
                rejected_entries.append({
                    "image_bytes": f.read(),
                    "rejection_reasons": reasons,
                })

        assets_indexed = asset_rag.index_assets(brand_id, approved_bytes, rejected_entries)
        brand_store.update(brand_id, {"asset_count": assets_indexed})

        logger.info(f"Brand onboarding complete: {brand_id}, chunks={chunks_indexed}, assets={assets_indexed}")

        return jsonify({
            'success': True,
            'brand_id': brand_id,
            'brand_name': brand_name,
            'extracted_rules': structured_rules,
            'chunks_indexed': chunks_indexed,
            'assets_indexed': assets_indexed,
        })

    except Exception as e:
        logger.error(f"Brand onboarding failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Brand onboarding failed: {str(e)}'}), 500


@app.route('/api/brand/<brand_id>', methods=['GET'])
def get_brand(brand_id: str):
    """Return a brand profile by ID."""
    try:
        profile = brand_store.get(brand_id)
        if not profile:
            return jsonify({'error': 'Brand not found'}), 404
        return jsonify({'success': True, 'brand': profile})
    except Exception as e:
        logger.error(f"Failed to get brand {brand_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/brand', methods=['GET'])
def list_brands():
    """List all registered brand profiles."""
    try:
        brands = brand_store.list_brands()
        return jsonify({'success': True, 'brands': brands, 'count': len(brands)})
    except Exception as e:
        logger.error(f"Failed to list brands: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/brand/<brand_id>', methods=['DELETE'])
def delete_brand(brand_id: str):
    """Delete a brand profile and its Qdrant collections."""
    try:
        profile = brand_store.get(brand_id)
        if not profile:
            return jsonify({'error': 'Brand not found'}), 404
        text_rag.delete_brand_collection(brand_id)
        asset_rag.delete_brand_collection(brand_id)
        brand_store.delete(brand_id)
        return jsonify({'success': True, 'brand_id': brand_id})
    except Exception as e:
        logger.error(f"Failed to delete brand {brand_id}: {e}")
        return jsonify({'error': str(e)}), 500


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
        
        # Run the Flask app with multiprocessing disabled
        app.run(
            debug=False,  # Disable debug mode to prevent reloader issues
            host='0.0.0.0',
            port=5003,
            threaded=True,
            use_reloader=False,  # Reloader kills worker mid-request, causing socket hang up
            processes=1  # Single process to avoid multiprocessing conflicts
        )
        
    except KeyboardInterrupt:
        logger.info("App stopped by user")
        cleanup_resources()
    except Exception as e:
        logger.error(f"App crashed: {e}")
        cleanup_resources()
        sys.exit(1)
