"""
Celery tasks for BrandGuard analysis pipeline.

The worker process initialises the PipelineOrchestrator once on startup
(heavy ML models are loaded into memory once, not per-task).
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

from src.brandguard.celery_app import celery_app

logger = logging.getLogger(__name__)

_INTERNAL_SECRET = os.environ.get("INTERNAL_WEBHOOK_SECRET", "internal-brandguard-secret")

# ---------------------------------------------------------------------------
# Lazy pipeline singleton — initialised once per worker process
# ---------------------------------------------------------------------------

_pipeline = None
_settings = None


def _get_pipeline():
    """Return the shared PipelineOrchestrator, initialising it on first call."""
    global _pipeline, _settings
    if _pipeline is None:
        from dotenv import load_dotenv
        load_dotenv()

        from src.brandguard.config.settings import Settings
        from src.brandguard.core.pipeline_orchestrator_new import PipelineOrchestrator

        _settings = Settings()
        _pipeline = PipelineOrchestrator(_settings)
        logger.info("[celery-worker] PipelineOrchestrator initialised")
    return _pipeline


# ---------------------------------------------------------------------------
# Main analysis task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="brandguard.run_analysis",
    acks_late=True,       # ack only after task completes (safe against worker crash)
    max_retries=0,        # fail-fast — no auto-retry
    bind=True,
)
def run_analysis_task(
    self,
    job_id: str,
    input_source: str,
    source_type: str,
    analysis_options: Dict[str, Any],
    brand_id: Optional[str],
    callback_url: Optional[str],
    cleanup_path: Optional[str],
) -> Dict[str, Any]:
    """
    Run the full BrandGuard analysis pipeline and POST results to callback_url.

    Parameters mirror _run_analysis_background() in app.py so the swap is drop-in.
    """
    logger.info("[%s] Celery task started (source_type=%s)", job_id, source_type)

    try:
        pipeline = _get_pipeline()
        analysis_results = pipeline.analyze_content(
            input_source=input_source,
            source_type=source_type,
            analysis_options=analysis_options,
            brand_id=brand_id,
        )

        if "error" in analysis_results:
            payload: Dict[str, Any] = {
                "job_id": job_id,
                "status": "failed",
                "error": analysis_results["error"],
            }
        else:
            payload = {
                "job_id": job_id,
                "status": "completed",
                "results": analysis_results,
            }

    except Exception as exc:
        logger.error("[%s] Celery task raised an exception: %s", job_id, exc, exc_info=True)
        payload = {"job_id": job_id, "status": "failed", "error": str(exc)}

    finally:
        # Clean up the temp upload file regardless of success/failure
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except OSError:
                pass

    # POST results back to br-be
    if callback_url:
        try:
            resp = requests.post(
                callback_url,
                json=payload,
                headers={"X-Internal-Secret": _INTERNAL_SECRET},
                timeout=30,
            )
            logger.info(
                "[%s] Callback sent → %s (HTTP %s)",
                job_id,
                callback_url,
                resp.status_code,
            )
        except Exception as cb_err:
            logger.error("[%s] Callback POST failed: %s", job_id, cb_err)

    return payload
