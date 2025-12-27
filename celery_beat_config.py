# Celery Beat Configuration
# Add this to backend/app/celery_app.py

from celery.schedules import crontab

# Add to celery_app configuration
celery_app.conf.beat_schedule = {
    # Signal generation - every 60 seconds
    'generate-signals': {
        'task': 'app.tasks.signal_generation.generate_signals',
        'schedule': 60.0,
    },
    
    # Execute paper trades - every 60 seconds
    'execute-paper-trades': {
        'task': 'app.tasks.paper_trading.execute_paper_trades',
        'schedule': 60.0,
    },
    
    # Monitor paper trades - every 30 seconds
    'monitor-paper-trades': {
        'task': 'app.tasks.paper_trading.monitor_paper_trades',
        'schedule': 30.0,
    },
    
    # Fresh features computation - every 60 seconds (if enabled)
    'compute-fresh-features': {
        'task': 'app.tasks.fresh_features.compute_fresh_features',
        'schedule': 60.0,
    },
    
    # Daily scheduled retrain - every day at 6 PM
    'scheduled-retrain': {
        'task': 'app.tasks.model_training.scheduled_retrain',
        'schedule': crontab(hourof day=18, minute=0),
    },
}

celery_app.conf.timezone = 'Asia/Kolkata'
