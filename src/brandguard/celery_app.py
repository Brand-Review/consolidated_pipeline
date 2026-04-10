"""
Celery application instance for BrandGuard background analysis tasks.

Broker: Redis (CELERY_BROKER_URL env var, default redis://localhost:6379/0)
Backend: Redis (CELERY_RESULT_BACKEND env var, same default)

Start a worker:
    celery -A src.brandguard.celery_app worker --loglevel=info --concurrency=1
"""

import os

from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)

celery_app = Celery(
    "brandguard",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["src.brandguard.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability: task is not ack'd until it finishes (safe against worker crashes)
    task_acks_late=True,
    # Only fetch one task at a time per worker (heavy ML models need all the RAM)
    worker_prefetch_multiplier=1,
    # No automatic retries — surface failures immediately
    task_max_retries=0,
    # Results expire after 1 hour (we don't rely on Celery result backend for status)
    result_expires=3600,
    # Timezone
    timezone="UTC",
    enable_utc=True,
)
