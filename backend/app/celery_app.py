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

# Configuration - STABILIZED for reliability
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    # Increased limits for training tasks
    task_time_limit=7200,  # 2 hours max per task
    task_soft_time_limit=6000,  # Soft limit 100 min (warning)
    # Worker stability
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=1000,  # Increased from 500
    worker_max_memory_per_child=2000000,  # 2GB max memory
    # Retry settings
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,  # Retry if worker dies
    # FIX 6: Task routing - isolate training from signal generation
    task_routes={
        'app.tasks.model_training.*': {'queue': 'training'},
        'app.tasks.signal_generation.*': {'queue': 'signals'},
        'app.tasks.paper_trading.*': {'queue': 'signals'},
        'app.tasks.position_monitor.*': {'queue': 'signals'},
        'app.tasks.fresh_features.*': {'queue': 'signals'},
    },
    task_default_queue='default',
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])

# Explicitly import tasks to ensure registration
import app.tasks  # noqa: E402, F401

# STABILIZED Beat Schedule - reduced frequency to prevent overload
celery_app.conf.beat_schedule = {
    # Data ingestion - daily only
    'fetch-eod-daily': {
        'task': 'app.tasks.data_ingestion.fetch_eod_bhavcopy',
        'schedule': crontab(hour=18, minute=0),  # 6 PM UTC daily
    },
    
    # Fresh features - every 5 MINUTES (not 60 seconds)
    'compute-fresh-features': {
        'task': 'app.tasks.fresh_features.compute_fresh_features',
        'schedule': 300.0,  # Every 5 minutes
    },
    
    # Signal generation - every 5 MINUTES (not 60 seconds)
    'generate-signals': {
        'task': 'app.tasks.signal_generation.generate_signals',
        'schedule': 300.0,  # Every 5 minutes
    },
    
    # Paper trade execution - every 2 MINUTES (not 60 seconds)
    'execute-paper-trades': {
        'task': 'app.tasks.paper_trading.execute_paper_trades',
        'schedule': 120.0,  # Every 2 minutes
    },
    
    # Paper trade monitoring - every 2 MINUTES (not 30 seconds)
    'monitor-paper-trades': {
        'task': 'app.tasks.paper_trading.monitor_paper_trades',
        'schedule': 120.0,  # Every 2 minutes
    },
    
    # Position monitoring - every 2 MINUTES (not 30 seconds)
    'monitor-positions': {
        'task': 'app.tasks.position_monitor.monitor_open_positions',
        'schedule': 120.0,  # Every 2 minutes
    },
    
    # Weekly retraining
    'retrain-weekly': {
        'task': 'app.tasks.model_training.scheduled_retrain',
        'schedule': crontab(hour=2, minute=0, day_of_week=1),  # Monday 2 AM UTC
    },
    
    # Drift detection - daily
    'drift-detection-daily': {
        'task': 'app.tasks.monitoring.detect_model_drift',
        'schedule': crontab(hour=3, minute=0),  # 3 AM UTC daily
    },
    
    # News sentiment - daily at 6 AM (NEW)
    'fetch-news-sentiment': {
        'task': 'app.tasks.news_sentiment.fetch_and_analyze_news',
        'schedule': crontab(hour=6, minute=0),  # 6 AM UTC daily
    },
}

