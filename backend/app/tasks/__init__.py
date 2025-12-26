# Tasks package
"""
Explicitly import all Celery tasks to ensure proper discovery
"""
from app.tasks.model_training import train_model, scheduled_retrain
from app.tasks.data_ingestion import fetch_eod_bhavcopy, fetch_recent_data, fetch_historical_data
from app.tasks.execution import check_and_execute_signals
from app.tasks.monitoring import detect_model_drift
from app.tasks.news_sentiment import fetch_and_analyze_news, update_sentiment_features
from app.tasks.fresh_features import compute_fresh_features, compute_features_for_stock  # Phase 1

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
