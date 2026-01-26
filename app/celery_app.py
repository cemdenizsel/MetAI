"""
Celery Application Configuration

Configures Celery for async task processing with Redis broker.
"""

import os
import logging
from celery import Celery
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

if REDIS_PASSWORD:
    CELERY_BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
    CELERY_RESULT_BACKEND = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/2"
else:
    CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"
    CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/2"

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", CELERY_BROKER_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_RESULT_BACKEND)

celery_app = Celery(
    "emotion_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks.emotion_tasks"]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,

    # Additional configuration to fix exception serialization
    result_backend_transport_options={'retry_on_timeout': True},
    result_extended=True,

    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "1800")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1500")),

    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,

    result_expires=86400,
    result_persistent=True,

    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    worker_disable_rate_limits=False,

    task_routes={
        "tasks.emotion_tasks.analyze_video_async": {"queue": "emotion_analysis"},
        "tasks.emotion_tasks.batch_analyze_videos": {"queue": "batch_processing"},
        "tasks.emotion_tasks.cleanup_temp_files": {"queue": "maintenance"},
    },

    task_queues=(
        Queue("emotion_analysis", Exchange("emotion_analysis"), routing_key="emotion.analysis"),
        Queue("batch_processing", Exchange("batch_processing"), routing_key="batch.processing"),
        Queue("maintenance", Exchange("maintenance"), routing_key="maintenance"),
    ),

    task_send_sent_event=True,

    beat_schedule={
        "cleanup-temp-files-daily": {
            "task": "tasks.emotion_tasks.cleanup_temp_files",
            "schedule": 86400.0,
            "args": (["/tmp/emotion_*"], 24),
        },
    },

    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "2")),
)

logger.info(f"Celery app configured with broker: {CELERY_BROKER_URL[:30]}...")

# Task error handler
@celery_app.task(bind=True)
def error_handler(self, uuid):
    """Handle task errors."""
    result = celery_app.AsyncResult(uuid)
    logger.error(f"Task {uuid} raised exception: {result.traceback}")


if __name__ == "__main__":
    celery_app.start()

