"""
Author: Omer Sayem
Date: 2025-09-09
Version: 2.0.0
Description: Consolidated BrandGuard Pipeline API — FastAPI Edition
Unified backend serving all four models: Color, Typography, Copywriting, and Logo Detection
"""

import gc
import json
import logging
import multiprocessing
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

import requests as http_requests
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from werkzeug.utils import secure_filename

from src.brandguard.tasks import run_analysis_task
from src.brandguard.brand_profile.asset_rag import AssetRAG
from src.brandguard.brand_profile.brand_store import BrandStore
from src.brandguard.brand_profile.pdf_extractor import PDFRuleExtractor
from src.brandguard.brand_profile.text_rag import TextRAG
from src.brandguard.config.settings import Settings
from src.brandguard.core.pipeline_orchestrator_new import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal webhook secret — must match br-be INTERNAL_WEBHOOK_SECRET
# ---------------------------------------------------------------------------
_INTERNAL_SECRET = os.environ.get('INTERNAL_WEBHOOK_SECRET', 'internal-brandguard-secret')

# ---------------------------------------------------------------------------
# Settings + pipeline (module-level so they survive across requests)
# ---------------------------------------------------------------------------
settings: Optional[Settings] = None
pipeline: Optional[PipelineOrchestrator] = None

try:
    settings = Settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error("Failed to load settings: %s", e)

if settings:
    try:
        pipeline = PipelineOrchestrator(settings)
        logger.info("Pipeline orchestrator initialized successfully")
    except Exception as e:
        import traceback
        logger.error("Failed to initialize pipeline: %s", e)
        logger.error("Full traceback: %s", traceback.format_exc())

# Brand profile services (lazy, shared across requests)
brand_store = BrandStore()
text_rag = TextRAG()
asset_rag = AssetRAG()
pdf_extractor = PDFRuleExtractor()

# ---------------------------------------------------------------------------
# Runtime directories + constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path(settings.upload_dir if settings else 'uploads')
RESULTS_DIR = Path(settings.results_dir if settings else 'results')
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE: int = settings.max_file_size if settings else 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS: Dict[str, set] = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'},
    'documents': {'pdf', 'txt', 'doc', 'docx'},
    'all': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'pdf', 'txt', 'doc', 'docx'},
}

_DOCS_DIR = Path(__file__).resolve().parent / 'docs'

# ---------------------------------------------------------------------------
# In-memory job store for background analysis tasks
# Key: job_id (UUID string)  Value: {"status": "processing"|"completed"|"failed", ...}
# ---------------------------------------------------------------------------
_job_store: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str, file_type: str = 'all') -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, ALLOWED_EXTENSIONS['all'])


def get_file_type(filename: str) -> str:
    if not filename:
        return 'unknown'
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in ALLOWED_EXTENSIONS['images']:
        return 'image'
    if ext in ALLOWED_EXTENSIONS['documents']:
        return 'document'
    return 'unknown'


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
) -> None:
    """Background task: run the pipeline and POST results to callback_url."""
    _job_store[job_id] = {"status": "processing", "job_id": job_id}

    try:
        logger.info("[%s] Starting background analysis (source_type=%s)", job_id, source_type)
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
        logger.error("[%s] Analysis background task failed: %s", job_id, e)
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except Exception:
                pass
        payload = {'job_id': job_id, 'status': 'failed', 'error': str(e)}

    # Update in-memory store so /api/status/{job_id} reflects the final state
    _job_store[job_id] = payload

    if callback_url:
        try:
            http_requests.post(
                callback_url,
                json=payload,
                headers={'X-Internal-Secret': _INTERNAL_SECRET},
                timeout=30,
            )
            logger.info("[%s] Callback sent to %s", job_id, callback_url)
        except Exception as cb_err:
            logger.error("[%s] Callback POST failed: %s", job_id, cb_err)


def cleanup_resources() -> None:
    """Clean up all resources to prevent semaphore leaks."""
    try:
        logger.info("Cleaning up resources...")
        if pipeline:
            pipeline.cleanup()
        gc.collect()
        try:
            for process in multiprocessing.active_children():
                process.terminate()
                process.join(timeout=5)
        except Exception as e:
            logger.warning("Error cleaning up multiprocessing: %s", e)
        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.error("Error during cleanup: %s", e)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BrandGuard FastAPI starting up…")
    logger.info("Upload directory: %s", UPLOAD_DIR)
    logger.info("Results directory: %s", RESULTS_DIR)
    logger.info("Pipeline ready: %s", pipeline is not None)
    yield
    logger.info("BrandGuard FastAPI shutting down…")
    cleanup_resources()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BrandGuard AI API",
    version="2.0.0",
    description=(
        "Unified HTTP API for color, typography, copywriting, and logo brand-compliance analysis. "
        "POST /api/analyze returns a job_id immediately (HTTP 202) and POSTs results to "
        "callback_url when analysis completes. Single-model endpoints are synchronous."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Rate-limit exceeded → 429 JSON response
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (same as old Flask-CORS default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files + Jinja2 templates
templates = Jinja2Templates(directory="templates")
if Path("templates").exists():
    app.mount("/static", StaticFiles(directory="templates"), name="static")


# ---------------------------------------------------------------------------
# Middleware: request/response logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Middleware: body size guard (FastAPI has no built-in MAX_CONTENT_LENGTH)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        return JSONResponse(
            {"error": f"File too large. Maximum size is {limit_mb}MB."},
            status_code=413,
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        {"error": "Validation error", "detail": exc.errors()},
        status_code=422,
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse({"error": "Endpoint not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse({"error": "Internal server error"}, status_code=500)


# ===========================================================================
# Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# UI / docs
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/openapi.json", include_in_schema=False)
async def openapi_spec_json():
    """Machine-readable OpenAPI 3.1 document (hand-crafted spec from docs/)."""
    spec_file = _DOCS_DIR / 'openapi.yaml'
    if not spec_file.is_file():
        raise HTTPException(status_code=404, detail="OpenAPI specification not found")
    with open(spec_file, encoding='utf-8') as f:
        return JSONResponse(yaml.safe_load(f))


@app.get("/api/docs", response_class=HTMLResponse, include_in_schema=False)
async def api_docs_swagger_ui(request: Request):
    """Interactive API documentation (Swagger UI)."""
    template_path = Path("templates") / "api_docs.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="api_docs.html template not found")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Health"])
@limiter.limit("60/minute")
async def health(request: Request):
    """Liveness and pipeline readiness check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None,
        "settings_loaded": settings is not None,
        "version": "2.0.0",
        "debug_info": {
            "pipeline_type": type(pipeline).__name__ if pipeline else None,
            "settings_type": type(settings).__name__ if settings else None,
        },
    }


# ---------------------------------------------------------------------------
# Full analysis (async — returns job_id, POSTs callback when done)
# ---------------------------------------------------------------------------

@app.post("/api/analyze", tags=["Analysis"], status_code=202)
@limiter.limit("10/minute")
async def analyze_content(
    request: Request,
    # --- file / input ---
    file: Optional[UploadFile] = File(None),
    input_type: str = Form("file"),
    text_content: str = Form(""),
    url: str = Form(""),
    callback_url: str = Form(""),
    brand_id: str = Form(""),
    # --- generic options ---
    analysis_priority: str = Form("balanced"),
    report_detail: str = Form("detailed"),
    include_recommendations: str = Form("true"),
    scoring_weights: str = Form(""),
    pass_threshold: float = Form(0.70),
    # --- color ---
    enable_color: str = Form("true"),
    color_n_colors: int = Form(8),
    color_tolerance: float = Form(2.3),
    enable_contrast_check: str = Form("true"),
    brand_palette: str = Form(""),
    primary_colors: str = Form(""),
    secondary_colors: str = Form(""),
    accent_colors: str = Form(""),
    primary_threshold: int = Form(75),
    secondary_threshold: int = Form(75),
    accent_threshold: int = Form(75),
    forbidden_colors: str = Form(""),
    # --- typography ---
    enable_typography: str = Form("true"),
    merge_regions: str = Form("true"),
    distance_threshold: int = Form(20),
    typography_confidence_threshold: float = Form(0.7),
    expected_fonts: str = Form(""),
    # --- copywriting ---
    enable_copywriting: str = Form("true"),
    formality_score: int = Form(60),
    confidence_level: str = Form("balanced"),
    warmth_score: int = Form(50),
    energy_score: int = Form(50),
    # --- logo ---
    enable_logo: str = Form("true"),
    enable_placement_validation: str = Form("true"),
    generate_annotations: str = Form("true"),
    logo_confidence_threshold: float = Form(0.5),
    max_logo_detections: int = Form(100),
    allowed_zones: str = Form("top-left,top-right,bottom-left,bottom-right"),
    min_logo_size: float = Form(0.01),
    max_logo_size: float = Form(0.25),
    min_edge_distance: float = Form(0.05),
    show_original_image: str = Form("false"),
    show_analysis_overlay: str = Form("false"),
    annotation_style: str = Form("bounding_box"),
    annotation_color: str = Form("gray"),
    warning_threshold: float = Form(0.5),
    critical_threshold: float = Form(0.3),
):
    """
    Async full analysis endpoint.

    Returns `{job_id, status: "processing"}` immediately (HTTP 202).
    Analysis runs in the background; results are POSTed to `callback_url`
    (br-be `/api/internal/analysis/callback/:analysisId`) when complete.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    job_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    # Build flat form dict for the shared helper
    form_data = {
        "analysis_priority": analysis_priority,
        "report_detail": report_detail,
        "include_recommendations": include_recommendations,
        "scoring_weights": scoring_weights,
        "pass_threshold": str(pass_threshold),
        "enable_color": enable_color,
        "color_n_colors": str(color_n_colors),
        "color_tolerance": str(color_tolerance),
        "enable_contrast_check": enable_contrast_check,
        "brand_palette": brand_palette,
        "primary_colors": primary_colors,
        "secondary_colors": secondary_colors,
        "accent_colors": accent_colors,
        "primary_threshold": str(primary_threshold),
        "secondary_threshold": str(secondary_threshold),
        "accent_threshold": str(accent_threshold),
        "forbidden_colors": forbidden_colors,
        "enable_typography": enable_typography,
        "merge_regions": merge_regions,
        "distance_threshold": str(distance_threshold),
        "typography_confidence_threshold": str(typography_confidence_threshold),
        "expected_fonts": expected_fonts,
        "enable_copywriting": enable_copywriting,
        "formality_score": str(formality_score),
        "confidence_level": confidence_level,
        "warmth_score": str(warmth_score),
        "energy_score": str(energy_score),
        "enable_logo": enable_logo,
        "enable_placement_validation": enable_placement_validation,
        "generate_annotations": generate_annotations,
        "logo_confidence_threshold": str(logo_confidence_threshold),
        "max_logo_detections": str(max_logo_detections),
        "allowed_zones": allowed_zones,
        "min_logo_size": str(min_logo_size),
        "max_logo_size": str(max_logo_size),
        "min_edge_distance": str(min_edge_distance),
        "show_original_image": show_original_image,
        "show_analysis_overlay": show_analysis_overlay,
        "annotation_style": annotation_style,
        "annotation_color": annotation_color,
        "warning_threshold": str(warning_threshold),
        "critical_threshold": str(critical_threshold),
    }
    analysis_options = _build_analysis_options(form_data)

    filepath: Optional[str] = None

    if input_type == "file":
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not supported")
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(UPLOAD_DIR / f"{timestamp}_{filename}")
        with open(filepath, "wb") as f_out:
            f_out.write(await file.read())
        input_source = filepath
        source_type = get_file_type(filename)

    elif input_type == "text":
        tc = text_content.strip()
        if not tc:
            raise HTTPException(status_code=400, detail="No text content provided")
        input_source = tc
        source_type = "text"

    elif input_type == "url":
        u = url.strip()
        if not u:
            raise HTTPException(status_code=400, detail="No URL provided")
        input_source = u
        source_type = "url"

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported input type: {input_type}")

    logger.info("[%s] Queuing Celery analysis task (source_type=%s, brand_id=%s)", job_id, source_type, brand_id or None)

    run_analysis_task.delay(
        job_id,
        input_source,
        source_type,
        analysis_options,
        brand_id.strip() or None,
        callback_url.strip() or None,
        filepath,
    )

    return JSONResponse({"job_id": job_id, "status": "processing"}, status_code=202)


# ---------------------------------------------------------------------------
# Single-model endpoints (synchronous)
# ---------------------------------------------------------------------------

def _save_upload(file: UploadFile, content: bytes) -> tuple[str, str]:
    """Save an uploaded file and return (filepath, source_type)."""
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = str(UPLOAD_DIR / f"{timestamp}_{filename}")
    with open(filepath, "wb") as f_out:
        f_out.write(content)
    return filepath, get_file_type(filename)


@app.post("/api/analyze/color", tags=["Analysis"])
@limiter.limit("30/minute")
async def analyze_colors(
    request: Request,
    file: UploadFile = File(...),
    n_colors: int = Form(8),
    n_clusters: int = Form(8),
    color_tolerance: float = Form(2.3),
    enable_contrast_check: str = Form("true"),
    primary_colors: str = Form(""),
    secondary_colors: str = Form(""),
    accent_colors: str = Form(""),
    primary_threshold: int = Form(75),
    secondary_threshold: int = Form(75),
    accent_threshold: int = Form(75),
):
    """Color-only analysis (CIEDE2000 palette matching, WCAG contrast)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not supported")

    content = await file.read()
    filepath, _ = _save_upload(file, content)
    try:
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type="image",
            analysis_options={
                "color_analysis": {
                    "n_colors": n_colors,
                    "n_clusters": n_clusters,
                    "color_tolerance": color_tolerance,
                    "enable_contrast_check": enable_contrast_check.lower() == "true",
                    "primary_colors": primary_colors.strip(),
                    "secondary_colors": secondary_colors.strip(),
                    "accent_colors": accent_colors.strip(),
                    "primary_threshold": primary_threshold,
                    "secondary_threshold": secondary_threshold,
                    "accent_threshold": accent_threshold,
                }
            },
        )
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
    return {"success": True, "color_analysis": results.get("model_results", {}).get("color_analysis", {})}


@app.post("/api/analyze/typography", tags=["Analysis"])
@limiter.limit("30/minute")
async def analyze_typography(
    request: Request,
    file: UploadFile = File(...),
    merge_regions: str = Form("true"),
    distance_threshold: int = Form(20),
    confidence_threshold: float = Form(0.7),
    enable_font_validation: str = Form("true"),
):
    """Typography-only analysis (OCR + font identification)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not supported")

    content = await file.read()
    filepath, _ = _save_upload(file, content)
    try:
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type="image",
            analysis_options={
                "typography_analysis": {
                    "merge_regions": merge_regions.lower() == "true",
                    "distance_threshold": distance_threshold,
                    "confidence_threshold": confidence_threshold,
                    "enable_font_validation": enable_font_validation.lower() == "true",
                }
            },
        )
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
    return {"success": True, "typography_analysis": results.get("model_results", {}).get("typography_analysis", {})}


@app.post("/api/analyze/copywriting", tags=["Analysis"])
@limiter.limit("30/minute")
async def analyze_copywriting(
    request: Request,
    file: Optional[UploadFile] = File(None),
    input_type: str = Form("file"),
    text_content: str = Form(""),
    include_suggestions: str = Form("true"),
    include_industry_benchmarks: str = Form("true"),
    enable_brand_profile_matching: str = Form("true"),
    formality_score: Optional[int] = Form(None),
    confidence_level: Optional[str] = Form(None),
    warmth_score: Optional[int] = Form(None),
    energy_score: Optional[int] = Form(None),
):
    """Copywriting / tone analysis. Use `input_type=text` with `text_content`, or upload a file."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    analysis_options: Dict[str, Any] = {
        "copywriting_analysis": {
            "include_suggestions": include_suggestions.lower() == "true",
            "include_industry_benchmarks": include_industry_benchmarks.lower() == "true",
            "enable_brand_profile_matching": enable_brand_profile_matching.lower() == "true",
        }
    }

    brand_voice: Dict[str, Any] = {}
    if formality_score is not None:
        brand_voice["formality_score"] = formality_score
    if confidence_level:
        brand_voice["confidence_level"] = confidence_level
    if warmth_score is not None:
        brand_voice["warmth_score"] = warmth_score
    if energy_score is not None:
        brand_voice["energy_score"] = energy_score
    if brand_voice:
        analysis_options["brand_voice_settings"] = brand_voice

    filepath: Optional[str] = None

    if input_type == "file":
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not supported")
        content = await file.read()
        filepath, source_type = _save_upload(file, content)
        input_source = filepath
    elif input_type == "text":
        tc = text_content.strip()
        if not tc:
            raise HTTPException(status_code=400, detail="No text content provided")
        input_source = tc
        source_type = "text"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported input type: {input_type}")

    try:
        results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options,
        )
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
    return {"success": True, "copywriting_analysis": results.get("model_results", {}).get("copywriting_analysis", {})}


@app.post("/api/analyze/logo", tags=["Analysis"])
@limiter.limit("30/minute")
async def analyze_logos(
    request: Request,
    file: UploadFile = File(...),
    enable_placement_validation: str = Form("true"),
    enable_brand_compliance: str = Form("true"),
    generate_annotations: str = Form("true"),
    logo_confidence_threshold: float = Form(0.5),
):
    """Logo-only analysis (YOLOv8 + Qwen2.5-VL fallback)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not supported")

    content = await file.read()
    filepath, _ = _save_upload(file, content)
    image_id = f"direct_{uuid.uuid4().hex[:8]}"
    try:
        results = pipeline.analyze_content(
            input_source=filepath,
            source_type="image",
            analysis_options={
                "logo_analysis": {
                    "enabled": True,
                    "enable_placement_validation": enable_placement_validation.lower() == "true",
                    "enable_brand_compliance": enable_brand_compliance.lower() == "true",
                    "generate_annotations": generate_annotations.lower() == "true",
                    "confidence_threshold": logo_confidence_threshold,
                    "image_id": image_id,
                }
            },
        )
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
    return {"success": True, "logo_analysis": results.get("model_results", {}).get("logo_analysis", {})}


# ---------------------------------------------------------------------------
# Brand profile management
# ---------------------------------------------------------------------------

@app.post("/api/brand/onboard", tags=["Brand"])
@limiter.limit("5/minute")
async def onboard_brand(
    request: Request,
    brand_name: str = Form(...),
    brand_id: str = Form(""),
    guideline_pdf: UploadFile = File(...),
    approved_images: List[UploadFile] = File([]),
    rejected_images: List[UploadFile] = File([]),
    rejection_reasons: str = Form("[]"),
):
    """
    Brand onboarding: ingest a PDF guideline + optional approved/rejected images.

    Returns `{brand_id, brand_name, extracted_rules, chunks_indexed, assets_indexed}`.
    """
    if not brand_name.strip():
        raise HTTPException(status_code=400, detail="brand_name is required")
    if not guideline_pdf.filename or not guideline_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="guideline_pdf must be a PDF file")

    # Save PDF temporarily
    pdf_filename = secure_filename(guideline_pdf.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = str(UPLOAD_DIR / f"{timestamp}_{pdf_filename}")
    with open(pdf_path, "wb") as f_out:
        f_out.write(await guideline_pdf.read())

    try:
        logger.info("Extracting rules from PDF for brand: %s", brand_name)
        structured_rules, chunks = pdf_extractor.extract(pdf_path)
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    brand_id_override = brand_id.strip() or None
    bid = brand_store.create(brand_name.strip(), brand_id=brand_id_override)
    brand_store.update(bid, {
        "rules": structured_rules,
        "qdrant_guideline_collection": text_rag.collection_name(bid),
        "qdrant_asset_collection": asset_rag.collection_name(bid),
    })

    chunks_indexed = text_rag.index_chunks(bid, chunks)
    brand_store.update(bid, {"chunk_count": chunks_indexed})

    # Approved images
    approved_bytes = []
    for f in approved_images:
        if f and f.filename:
            approved_bytes.append(await f.read())

    # Rejected images + reasons
    try:
        all_reasons = json.loads(rejection_reasons)
    except Exception:
        all_reasons = []

    rejected_entries = []
    for i, f in enumerate(rejected_images):
        if f and f.filename:
            reasons = all_reasons[i] if i < len(all_reasons) else []
            if isinstance(reasons, str):
                reasons = [reasons]
            rejected_entries.append({"image_bytes": await f.read(), "rejection_reasons": reasons})

    assets_indexed = asset_rag.index_assets(bid, approved_bytes, rejected_entries)
    brand_store.update(bid, {"asset_count": assets_indexed})

    logger.info("Brand onboarding complete: %s, chunks=%d, assets=%d", bid, chunks_indexed, assets_indexed)

    return {
        "success": True,
        "brand_id": bid,
        "brand_name": brand_name.strip(),
        "extracted_rules": structured_rules,
        "chunks_indexed": chunks_indexed,
        "assets_indexed": assets_indexed,
    }


@app.get("/api/brand", tags=["Brand"])
@limiter.limit("60/minute")
async def list_brands(request: Request):
    """List all registered brand profiles."""
    brands = brand_store.list_brands()
    return {"success": True, "brands": brands, "count": len(brands)}


@app.get("/api/brand/{brand_id}", tags=["Brand"])
@limiter.limit("60/minute")
async def get_brand(brand_id: str, request: Request):
    """Return a brand profile by ID."""
    profile = brand_store.get(brand_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Brand not found")
    return {"success": True, "brand": profile}


@app.delete("/api/brand/{brand_id}", tags=["Brand"])
@limiter.limit("10/minute")
async def delete_brand(brand_id: str, request: Request):
    """Delete a brand profile and its Qdrant collections."""
    profile = brand_store.get(brand_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Brand not found")
    text_rag.delete_brand_collection(brand_id)
    asset_rag.delete_brand_collection(brand_id)
    brand_store.delete(brand_id)
    return {"success": True, "brand_id": brand_id}


# ---------------------------------------------------------------------------
# Status + config
# ---------------------------------------------------------------------------

@app.get("/api/status/{analysis_id}", tags=["Analysis"])
@limiter.limit("60/minute")
async def get_analysis_status(analysis_id: str, request: Request):
    """Get status of a specific background analysis job."""
    job = _job_store.get(analysis_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"success": True, "status": job}


@app.get("/api/config", tags=["Config"])
@limiter.limit("60/minute")
async def get_config(request: Request):
    """Get current configuration summary (non-secret values only)."""
    if not settings:
        raise HTTPException(status_code=503, detail="Settings not loaded")
    return {
        "success": True,
        "config": {
            "color_palette": {
                "name": settings.color_palette.name,
                "primary_colors_count": len(settings.color_palette.primary_colors),
                "secondary_colors_count": len(settings.color_palette.secondary_colors),
            },
            "typography_rules": {
                "approved_fonts_count": len(settings.typography_rules.approved_fonts),
                "max_font_size": settings.typography_rules.max_font_size,
                "min_font_size": settings.typography_rules.min_font_size,
            },
            "brand_voice": {
                "formality_score": settings.brand_voice.formality_score,
                "confidence_level": settings.brand_voice.confidence_level,
                "warmth_score": settings.brand_voice.warmth_score,
                "energy_score": settings.brand_voice.energy_score,
            },
            "logo_detection": {
                "confidence_threshold": settings.logo_detection.confidence_threshold,
                "max_detections": settings.logo_detection.max_detections,
            },
        },
    }


@app.post("/api/config", tags=["Config"])
@limiter.limit("10/minute")
async def update_config(request: Request):
    """Update configuration (stub — not yet implemented)."""
    if not settings:
        raise HTTPException(status_code=503, detail="Settings not loaded")
    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail="No configuration data provided")
    # TODO: implement per-section config updates
    return {"success": True, "message": "Configuration updated successfully"}


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------

@app.get("/uploads/{filename}", tags=["Files"])
async def get_upload(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/results/{filename}", tags=["Files"])
async def get_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
