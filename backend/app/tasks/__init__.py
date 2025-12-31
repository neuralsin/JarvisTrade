# Tasks package
"""
Explicitly import all Celery tasks to ensure proper discovery
"""
# Import all tasks so Celery can discover them
from app.tasks.data_ingestion import *  # noqa
from app.tasks.model_training import *  # noqa
from app.tasks.monitoring import *  # noqa
from app.tasks.execution import *  # noqa
from app.tasks.position_monitor import *  # noqa
from app.tasks.fresh_features import *  # noqa
from app.tasks.news_sentiment import *  # noqa

# ✅ NEW: Import new signal generation and paper trading tasks
from app.tasks.signal_generation import *  # noqa
from app.tasks.paper_trading import *  # noqa

# ✅ V2 DUAL-MODEL TASKS - CRITICAL for V2 training
try:
    from app.tasks.model_training_v2 import *  # noqa
    from app.tasks.signal_generation_v2 import *  # noqa
except ImportError as e:
    import logging
    logging.warning(f"V2 tasks not available: {e}")

__all__ = [
    'train_model',
    'scheduled_retrain',
    'fetch_eod_bhavcopy',
    'fetch_recent_data',
    'fetch_historical_data',
    'check_and_execute_signals',
    'detect_model_drift',
    'fetch_and_analyze_news',
    'update_sentiment_features',
    'compute_fresh_features',  # Phase 1
    'compute_features_for_stock',  # Phase 1
]
