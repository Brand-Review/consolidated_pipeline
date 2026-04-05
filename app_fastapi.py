"""
Consolidated BrandGuard Pipeline API – FastAPI Edition
Unified backend serving all four models: Color, Typography, Copywriting, and Logo Detection
"""

import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import (FastAPI, File, Form, UploadFile, HTTPException,
                     status, Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Re-use existing business code
from src.brandguard.config.settings import Settings
from src.brandguard.core.pipeline_orchestrator import PipelineOrchestrator

# ---------------------- logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------- settings ------------------------
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
        logger.info("Pipeline orchestrator initialized")
    except Exception as e:
        logger.error("Failed to initialize pipeline: %s", e)

# ---------------------- FastAPI app ---------------------
_OPENAPI_TAGS = [
    {"name": "Health", "description": "Liveness and pipeline readiness"},
    {"name": "Analysis", "description": "Full four-model run and single-analyzer endpoints"},
    {"name": "Config", "description": "YAML-backed settings summary (non-secret)"},
    {"name": "Files", "description": "Upload and result asset URLs"},
]

app = FastAPI(
    title="BrandGuard AI API (FastAPI)",
    version="1.0.0",
    description=(
        "Synchronous HTTP API for color, typography, copywriting, and logo analysis. "
        "Unlike the default Flask `app.py`, `POST /api/analyze` here runs inline and returns full results in the response. "
        "For async jobs + `callback_url` integration with br-be, run the Flask app instead. "
        "OpenAPI is generated automatically; use **/docs** (Swagger) or **/redoc** (ReDoc)."
    ),
    openapi_tags=_OPENAPI_TAGS,
    contact={"name": "BrandReview / BrandGuard", "url": "https://app.brandreview.ai"},
    license_info={"name": "Proprietary"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS (same as Flask-CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")
# Ensure runtime directories
UPLOAD_DIR = Path(settings.upload_dir if settings else "uploads")
RESULTS_DIR = Path(settings.results_dir if settings else "results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = settings.max_file_size if settings else 50 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {
    "images": {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"},
    "documents": {"pdf", "txt", "doc", "docx"},
    "all": {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "pdf", "txt", "doc", "docx"},
}

# ---------------------- helper functions ----------------
def allowed_file(filename: str, file_type: str = "all") -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, ALLOWED_EXTENSIONS["all"])

def get_file_type(filename: str) -> str:
    if not filename:
        return "unknown"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ALLOWED_EXTENSIONS["images"]:
        return "image"
    if ext in ALLOWED_EXTENSIONS["documents"]:
        return "document"
    return "unknown"

# ---------------------- routes --------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get(
    "/api/health",
    tags=["Health"],
    summary="Health check",
    response_description="Pipeline and settings readiness flags",
)
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None,
        "settings_loaded": settings is not None,
        "version": "1.0.0",
    }

# ---------------------- main analysis --------------------
@app.post(
    "/api/analyze",
    tags=["Analysis"],
    summary="Full analysis (synchronous)",
    description=(
        "Runs all enabled analyzers and returns `results` in this response. "
        "Send `multipart/form-data`: first file part is the asset when `input_type=file`; "
        "use `text_content` or `url` fields for other input types. Boolean options are form strings `true`/`false`."
    ),
    responses={
        200: {"description": "Analysis finished successfully"},
        400: {"description": "Missing or invalid input"},
        500: {"description": "Analyzer returned an error"},
        503: {"description": "Pipeline not initialized"},
    },
)
async def analyze_content(
    files: List[UploadFile] = File(...),
    input_type: str = Form("file"),
    text_content: str = Form(""),
    url: str = Form(""),
    # Color
    color_n_colors: int = Form(8),
    color_tolerance: float = Form(0.2),
    enable_contrast_check: str = Form("true"),
    enable_color: str = Form("true"),
    brand_palette: str = Form(""),
    # Typography
    distance_threshold: int = Form(20),
    typography_confidence_threshold: float = Form(0.7),
    merge_regions: str = Form("true"),
    enable_typography: str = Form("true"),
    expected_fonts: str = Form(""),
    # Copywriting
    include_suggestions: str = Form("true"),
    include_industry_benchmarks: str = Form("true"),
    enable_copywriting: str = Form("true"),
    formality_score: int = Form(60),
    confidence_level: str = Form("balanced"),
    warmth_score: int = Form(50),
    energy_score: int = Form(50),
    # Logo
    enable_placement_validation: str = Form("true"),
    enable_logo: str = Form("true"),
    generate_annotations: str = Form("true"),
    logo_confidence_threshold: float = Form(0.5),
    max_logo_detections: int = Form(100),
    pass_threshold: float = Form(0.7),
    warning_threshold: float = Form(0.5),
    critical_threshold: float = Form(0.3),
    # Generic flags
    include_recommendations: str = Form("true"),
    analysis_priority: str = Form("balanced"),
    report_detail: str = Form("detailed"),
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Build analysis_options dict exactly like Flask
    analysis_options = {
        "analysis_priority": analysis_priority,
        "report_detail": report_detail,
        "include_recommendations": include_recommendations.lower() == "true",
        "color_analysis": {
            "enabled": enable_color.lower() == "true",
            "n_colors": color_n_colors,
            "n_clusters": color_n_colors,
            "color_tolerance": color_tolerance,
            "enable_contrast_check": enable_contrast_check.lower() == "true",
            "brand_palette": brand_palette,
        },
        "typography_analysis": {
            "enabled": enable_typography.lower() == "true",
            "merge_regions": merge_regions.lower() == "true",
            "distance_threshold": distance_threshold,
            "confidence_threshold": typography_confidence_threshold,
            "enable_font_validation": True,
            "expected_fonts": expected_fonts,
        },
        "copywriting_analysis": {
            "enabled": enable_copywriting.lower() == "true",
            "include_suggestions": include_suggestions.lower() == "true",
            "include_industry_benchmarks": include_industry_benchmarks.lower() == "true",
            "enable_brand_profile_matching": True,
            "formality_score": formality_score,
            "confidence_level": confidence_level,
            "warmth_score": warmth_score,
            "energy_score": energy_score,
        },
        "logo_analysis": {
            "enabled": enable_logo.lower() == "true",
            "enable_placement_validation": enable_placement_validation.lower() == "true",
            "enable_brand_compliance": True,
            "generate_annotations": generate_annotations.lower() == "true",
            "confidence_threshold": logo_confidence_threshold,
            "max_detections": max_logo_detections,
            "pass_threshold": pass_threshold,
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
        },
    }

    # Resolve input_source
    source_type = "unknown"
    input_source = None
    saved_path = None

    if input_type == "file":
        if not files or files[0].filename == "":
            raise HTTPException(status_code=400, detail="No file uploaded")
        uploaded = files[0]
        if not allowed_file(uploaded.filename):
            raise HTTPException(status_code=400, detail="File type not supported")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{uploaded.filename}"
        saved_path = UPLOAD_DIR / filename
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(uploaded.file, buffer)
        input_source = str(saved_path)
        source_type = get_file_type(filename)

    elif input_type == "text":
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content provided")
        input_source = text_content
        source_type = "text"

    elif input_type == "url":
        if not url.strip():
            raise HTTPException(status_code=400, detail="No URL provided")
        input_source = url
        source_type = "url"

    try:
        results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options,
        )
    finally:
        # Always clean up temp file
        if saved_path and saved_path.exists():
            saved_path.unlink(missing_ok=True)

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])

    return {"success": True, "results": results}

# ---------------------- single-model endpoints ----------
# They all follow the same pattern; for brevity we'll show one,
# then give you a reusable helper.

def _single_analysis(
    file: UploadFile,
    model_key: str,
    model_options: Dict[str, Any],
) -> Dict[str, Any]:
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not supported")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    with filepath.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = pipeline.analyze_content(
            input_source=str(filepath),
            source_type="image",
            analysis_options={model_key: model_options},
        )
    finally:
        filepath.unlink(missing_ok=True)

    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])

    return {"success": True, model_key: results["model_results"].get(model_key, {})}

@app.post(
    "/api/analyze/color",
    tags=["Analysis"],
    summary="Color analysis",
    responses={200: {"description": "CIEDE2000 / palette / contrast output"}, 503: {"description": "Pipeline down"}},
)
async def analyze_color(
    file: UploadFile = File(...),
    n_colors: int = Form(8),
    color_tolerance: float = Form(0.2),
    enable_contrast_check: str = Form("true"),
):
    return _single_analysis(
        file,
        "color_analysis",
        {
            "n_colors": n_colors,
            "n_clusters": n_colors,
            "color_tolerance": color_tolerance,
            "enable_contrast_check": enable_contrast_check.lower() == "true",
        },
    )

@app.post(
    "/api/analyze/typography",
    tags=["Analysis"],
    summary="Typography analysis",
    responses={200: {"description": "OCR + font identification output"}, 503: {"description": "Pipeline down"}},
)
async def analyze_typography(
    file: UploadFile = File(...),
    distance_threshold: int = Form(20),
    confidence_threshold: float = Form(0.7),
    merge_regions: str = Form("true"),
):
    return _single_analysis(
        file,
        "typography_analysis",
        {
            "distance_threshold": distance_threshold,
            "confidence_threshold": confidence_threshold,
            "merge_regions": merge_regions.lower() == "true",
            "enable_font_validation": True,
        },
    )

@app.post(
    "/api/analyze/copywriting",
    tags=["Analysis"],
    summary="Copywriting / tone analysis",
    description="`input_type=text` uses `text_content`; otherwise upload `file`.",
    responses={200: {"description": "Tone and brand-voice scoring"}, 503: {"description": "Pipeline down"}},
)
async def analyze_copywriting(
    file: UploadFile = File(None),
    text_content: str = Form(""),
    input_type: str = Form("file"),
    include_suggestions: str = Form("true"),
):
    if input_type == "text":
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content")
        results = pipeline.analyze_content(
            input_source=text_content,
            source_type="text",
            analysis_options={
                "copywriting_analysis": {
                    "include_suggestions": include_suggestions.lower() == "true",
                    "include_industry_benchmarks": True,
                    "enable_brand_profile_matching": True,
                }
            },
        )
    else:
        results = _single_analysis(
            file,
            "copywriting_analysis",
            {
                "include_suggestions": include_suggestions.lower() == "true",
                "include_industry_benchmarks": True,
                "enable_brand_profile_matching": True,
            },
        )
    return {"success": True, "copywriting_analysis": results["model_results"]["copywriting_analysis"]}

@app.post(
    "/api/analyze/logo",
    tags=["Analysis"],
    summary="Logo detection",
    responses={200: {"description": "YOLO / VL detections and placement hints"}, 503: {"description": "Pipeline down"}},
)
async def analyze_logo(
    file: UploadFile = File(...),
    generate_annotations: str = Form("true"),
):
    import uuid as _uuid
    image_id = f"direct_{_uuid.uuid4().hex[:8]}"
    return _single_analysis(
        file,
        "logo_analysis",
        {
            "generate_annotations": generate_annotations.lower() == "true",
            "enable_placement_validation": True,
            "enable_brand_compliance": True,
            "image_id": image_id,
        },
    )

# ---------------------- utilities -----------------------
@app.get(
    "/api/status/{analysis_id}",
    tags=["Analysis"],
    summary="Analysis job status",
    responses={404: {"description": "Unknown analysis id"}, 503: {"description": "Pipeline down"}},
)
async def get_analysis_status(analysis_id: str):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    status = pipeline.get_analysis_status(analysis_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return {"success": True, "status": status}

@app.get(
    "/api/config",
    tags=["Config"],
    summary="Configuration summary",
    responses={503: {"description": "Settings not loaded"}},
)
async def get_config():
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

# Static file serving (unchanged)
@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    return FileResponse(UPLOAD_DIR / filename)

@app.get("/results/{filename}")
async def get_result(filename: str):
    return FileResponse(RESULTS_DIR / filename)

# 413/404/500 handlers are automatically handled by FastAPI,
# but you can override if desired.