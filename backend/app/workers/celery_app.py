"""Celery application configuration."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "video_extract_pro",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Results
    result_expires=86400,  # 24 hours

    # Task routing
    task_routes={
        "app.workers.tasks.run_analysis_pipeline": {"queue": "analysis"},
        "app.workers.tasks.transcribe_video": {"queue": "gpu"},
        "app.workers.tasks.diarize_audio": {"queue": "gpu"},
        "app.workers.tasks.detect_scenes": {"queue": "analysis"},
        "app.workers.tasks.analyze_visuals": {"queue": "gpu"},
    },

    # Task time limits
    task_soft_time_limit=3600,  # 1 hour
    task_time_limit=3900,  # 1 hour 5 minutes (hard limit)

    # Worker settings
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (memory management)
)
