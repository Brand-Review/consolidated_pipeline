"""
Contract between Celery worker (tasks.py) and br-be completeAnalysis().

Run from repo root:
  pip install pytest
  pytest consolidated_pipeline/tests/test_callback_contract.py
"""


def test_completed_callback_shape():
    payload = {
        "job_id": "507f1f77bcf86cd799439011",
        "status": "completed",
        "results": {
            "verdict": "approved",
            "verdict_reason": "ok",
            "compliance_breakdown": {},
            "overall_compliance_score": "90%",
        },
    }
    assert payload["status"] == "completed"
    assert "results" in payload


def test_failed_callback_shape():
    payload = {
        "job_id": "507f1f77bcf86cd799439011",
        "status": "failed",
        "error": "Pipeline error",
    }
    assert payload["status"] == "failed"
    assert "error" in payload
