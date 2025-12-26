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
            # Note: For drift retrain, we should ideally retrain the same stock as active_model
            # For now, use first available stock or make this configurable
            train_model.delay(
                model_name=f"drift_retrain_{datetime.utcnow().strftime('%Y%m%d')}",
                instrument_filter="RELIANCE",  # TODO: Get from active_model.stock_symbol
                model_type="xgboost"
            )
            
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
    Compare production data distribution with training distribution using KS test
    
    Args:
        training_metrics: Model metadata containing training distributions
        feature_name: Name of feature to check
        production_data: Recent production data for this feature (numpy array or list)
    
    Returns:
        Dict with drift detection results or None if validation fails
        {
            'drift_detected': bool,
            'ks_stat': float,
            'p_value': float,
            'feature': str
        }
    
    Raises:
        None - Returns None on any error
    """
    try:
        # Validate inputs
        if not isinstance(training_metrics, dict):
            logger.warning(f"Invalid training_metrics type: {type(training_metrics)}")
            return None
        
        if not isinstance(feature_name, str):
            logger.warning(f"Invalid feature_name type: {type(feature_name)}")
            return None
        
        # Extract training distribution from metrics
        training_dists = training_metrics.get('feature_distributions', {})
        
        if not training_dists:
            logger.debug(f"No training distributions stored in model metadata")
            return None
        
        if feature_name not in training_dists:
            logger.debug(f"No stored training data for feature: {feature_name}")
            return None
        
        training_data = training_dists[feature_name]
        
        # Validate training data
        if not training_data or len(training_data) < 30:
            logger.warning(f"Insufficient training data for {feature_name}: {len(training_data) if training_data else 0} samples")
            return None
        
        # Validate production data
        if production_data is None or len(production_data) < 30:
            logger.debug(f"Insufficient production data for {feature_name}: {len(production_data) if production_data is not None else 0} samples")
            return None
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(training_data, production_data)
        
        # Drift detected if p-value < 0.01 (99% confidence)
        drift_detected = p_value < 0.01
        
        if drift_detected:
            logger.info(f"Drift detected in {feature_name}: KS={ks_stat:.4f}, p={p_value:.4f}")
        else:
            logger.debug(f"No drift in {feature_name}: KS={ks_stat:.4f}, p={p_value:.4f}")
        
        return {
            'drift_detected': drift_detected,
            'ks_stat': ks_stat,
            'p_value': p_value,
            'feature': feature_name
        }
    
    except Exception as e:
        logger.error(f"Error in drift detection for {feature_name}: {str(e)}")
        return None

