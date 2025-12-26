"""
Celery application setup for async task processing
"""
from celery import Celery
from celery.schedules import crontab
from app.config import settings

# Create Celery instance
celery_app = Celery(
    "jarvistrade",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])

# Explicitly import tasks to ensure registration
# This must happen after celery_app is created
import app.tasks  # noqa: E402, F401

# Spec 12: Scheduled tasks
celery_app.conf.beat_schedule = {
    'fetch-eod-daily': {
        'task': 'app.tasks.data_ingestion.fetch_eod_bhavcopy',
        'schedule': crontab(hour=18, minute=0),  # 6 PM UTC daily
    },
    'fetch-intraday-15min': {
        'task': 'app.tasks.data_ingestion.fetch_recent_data',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
    'compute-fresh-features': {  # Phase 1: Real-time features
        'task': 'app.tasks.fresh_features.compute_fresh_features',
        'schedule': 60.0,  # Every 60 seconds (respects rate limits via caching)
    },
    'check-signals': {
        'task': 'app.tasks.execution.check_and_execute_signals',
        'schedule': 60.0,  # Every 60 seconds for ultra-fast peak detection
    },
    'monitor-positions': {
        'task': 'app.tasks.position_monitor.monitor_open_positions',
        'schedule': 30.0,  # Every 30 seconds to catch stop/target/peak exits
    },
    'retrain-weekly': {
        'task': 'app.tasks.model_training.scheduled_retrain',
        'schedule': crontab(hour=2, minute=0, day_of_week=1),  # Monday 2 AM UTC
    },
    'drift-detection-daily': {
        'task': 'app.tasks.monitoring.detect_model_drift',
        'schedule': crontab(hour=3, minute=0),  # 3 AM UTC daily
    },
}
