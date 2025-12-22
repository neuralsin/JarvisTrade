"""
Spec 12: Monitoring tasks - drift detection and model performance tracking
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Model, Feature
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def detect_model_drift(self):
    """
    Spec 12: Detect data drift using KS-test
    """
    db = SessionLocal()
    
    try:
        # Get active model
        active_model = db.query(Model).filter(Model.is_active == True).first()
        if not active_model:
            return {"status": "no_active_model"}
        
        # Get training data distribution (stored in metrics)
        training_metrics = active_model.metrics_json
        
        # Get recent production data (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        recent_features = db.query(Feature).filter(
            Feature.ts_utc >= cutoff_date
        ).limit(1000).all()
        
        if not recent_features:
            return {"status": "no_recent_data"}
        
        # Convert to DataFrame
        df_recent = pd.DataFrame([f.feature_json for f in recent_features])
        
        # Run KS-test on key features
        drift_features = []
        feature_cols = ['rsi_14', 'atr_percent', 'volume_ratio', 'returns_1']
        
        for col in feature_cols:
            if col in df_recent.columns:
                # Get training distribution from model metadata
                training_dist = compare_with_training_distribution(
                    training_metrics, 
                    col, 
                    df_recent[col].dropna().values
                )
                
                if training_dist and training_dist['drift_detected']:
                    drift_features.append({
                        'feature': col,
                        'ks_stat': float(training_dist['ks_stat']),
                        'p_value': float(training_dist['p_value'])
                    })

        
        # Spec 12: If >3 features show drift, trigger retrain
        if len(drift_features) >= 3:
            logger.warning(f"Data drift detected in {len(drift_features)} features")
            from app.tasks.model_training import train_model
            train_model.delay(model_name=f"drift_retrain_{datetime.utcnow().strftime('%Y%m%d')}")
            
            return {
                "status": "drift_detected",
                "drift_features": drift_features,
                "retrain_triggered": True
            }
        
        return {
            "status": "no_drift",
            "drift_features": drift_features
        }
    
    finally:
        db.close()


def compare_with_training_distribution(training_metrics: dict, feature_name: str, production_data):
    """
    Compare production data distribution with training distribution
    
    Args:
        training_metrics: Model metadata containing training distributions
        feature_name: Name of feature to check
        production_data: Recent production data for this feature
    
    Returns:
        Dict with drift detection results
    """
    # Extract training distribution from metrics
    training_dists = training_metrics.get('feature_distributions', {})
    
    if feature_name not in training_dists:
        # No stored training data for this feature
        return None
    
    training_data = training_dists[feature_name]
    
    if len(production_data) < 30:
        # Not enough data for reliable test
        return None
    
    # Perform KS test
    ks_stat, p_value = stats.ks_2samp(training_data, production_data)
    
    # Drift detected if p-value < 0.01 (99% confidence)
    drift_detected = p_value < 0.01
    
    return {
        'drift_detected': drift_detected,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'feature': feature_name
    }

