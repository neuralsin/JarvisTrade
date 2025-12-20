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
                # Compare with training distribution (placeholder - would need stored training data)
                # For now, check against itself as example
                sample1 = df_recent[col].dropna().values[:len(df_recent)//2]
                sample2 = df_recent[col].dropna().values[len(df_recent)//2:]
                
                if len(sample1) > 30 and len(sample2) > 30:
                    ks_stat, p_value = stats.ks_2samp(sample1, sample2)
                    
                    if p_value < 0.01:
                        drift_features.append({
                            'feature': col,
                            'ks_stat': float(ks_stat),
                            'p_value': float(p_value)
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
