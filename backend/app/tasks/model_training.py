"""
Spec 5: Model training Celery task
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Model, Feature, Instrument, HistoricalCandle
from app.ml.trainer import ModelTrainer
from app.ml.lstm_model import LSTMPredictor
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels
from sqlalchemy import text
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def train_model(
    self, 
    model_name: str, 
    model_type: str = 'xgboost',
    instrument_filter: str = None, 
    hyperparams: dict = None,
    start_date: str = None,
    end_date: str = None
):
    """
    Spec 5: Train ML model with full pipeline
    
    Args:
        model_name: Name for the model
        model_type: 'xgboost', 'lstm', or 'transformer'
        instrument_filter: Optional symbol filter (e.g., 'RELIANCE')
        hyperparams: Optional hyperparameter overrides
        start_date: Training start date (YYYY-MM-DD), defaults to 2 years ago
        end_date: Training end date (YYYY-MM-DD), defaults to today
    """
    db = SessionLocal()
    task_id = self.request.id
    
    try:
        logger.info(f"Starting {model_type} model training: {model_name} (task: {task_id})")
        self.update_state(state='PROGRESS', meta={'status': 'Initializing training pipeline...', 'progress': 5})
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            start_dt = datetime.utcnow() - timedelta(days=730)  # 2 years default
            start_date = start_dt.strftime('%Y-%m-%d')
        
        self.update_state(state='PROGRESS', meta={'status': f'Loading data from {start_date} to {end_date}...', 'progress': 10})
        
        # Load feature data with date filtering
        query = db.query(Feature).join(Instrument)
        
        if instrument_filter:
            query = query.filter(Instrument.symbol == instrument_filter)
        
        # Apply date range
        query = query.filter(
            Feature.ts_utc >= start_date,
            Feature.ts_utc <= end_date
        )
        
        features_data = query.all()
        
        if not features_data:
            # Check if there's any data at all
            total_features = db.query(Feature).count()
            total_instruments = db.query(Instrument).count()
            
            error_msg = f"No training data found for the specified criteria."
            if instrument_filter:
                error_msg += f" Filter: {instrument_filter}."
            error_msg += f" Date range: {start_date} to {end_date}."
            error_msg += f" Total features in DB: {total_features}, Total instruments: {total_instruments}."
            
            if total_features == 0:
                error_msg += " You may need to run data ingestion first."
            
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df_rows = []
        for feat in features_data:
            row = feat.feature_json.copy() if feat.feature_json else {}
            row['target'] = feat.target
            row['ts_utc'] = feat.ts_utc
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        logger.info(f"Loaded {len(df)} feature rows from {start_date} to {end_date}")
        self.update_state(state='PROGRESS', meta={'status': f'Preparing {len(df)} samples for {model_type} training...', 'progress': 25})
        
        # Train based on model type
        if model_type == 'lstm':
            metrics, model_path = _train_lstm(df, model_name, hyperparams, self)
        elif model_type == 'transformer':
            metrics, model_path = _train_transformer(df, model_name, hyperparams, self)
        else:  # xgboost
            metrics, model_path = _train_xgboost(df, model_name, hyperparams, self)
        
        # Save to database
        model_record = Model(
            name=model_name,
            model_type=model_type,
            model_path=model_path,
            trained_at=datetime.utcnow(),
            metrics_json=metrics,
            is_active=False  # Manually activate later
        )
        db.add(model_record)
        db.commit()
        
        logger.info(f"Model training complete: {model_name} (AUC: {metrics.get('auc_roc', 0):.4f})")
        
        return {
            "status": "success",
            "model_id": str(model_record.id),
            "model_name": model_name,
            "model_type": model_type,
            "metrics": metrics,
            "date_range": f"{start_date} to {end_date}",
            "samples": len(df)
        }
    
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


def _train_xgboost(df, model_name, hyperparams, task):
    """Train XGBoost model"""
    task.update_state(state='PROGRESS', meta={'status': 'Splitting data into train/validation/test sets...', 'progress': 35})
    
    trainer = ModelTrainer(hyperparams=hyperparams)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = trainer.prepare_data(df)
    
    task.update_state(state='PROGRESS', meta={'status': f'Training XGBoost on {len(X_train)} samples...', 'progress': 50})
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating model performance...', 'progress': 75})
    metrics = trainer.evaluate(model, X_test, y_test, feature_cols)
    
    task.update_state(state='PROGRESS', meta={'status': 'Saving model to disk...', 'progress': 90})
    model_path = trainer.save_model(model, model_name, metrics)
    
    return metrics, model_path


def _train_lstm(df, model_name, hyperparams, task):
    """Train LSTM model"""
    import os
    
    task.update_state(state='PROGRESS', meta={'status': 'Initializing LSTM neural network...', 'progress': 35})
    
    lstm = LSTMPredictor()
    
    # Prepare sequences
    task.update_state(state='PROGRESS', meta={'status': 'Creating time-series sequences...', 'progress': 45})
    X, y = lstm.prepare_sequences(df)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Train
    task.update_state(state='PROGRESS', meta={'status': f'Training LSTM neural network (50 epochs)...', 'progress': 55})
    history = lstm.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating LSTM performance...', 'progress': 80})
    y_pred = lstm.predict(X_test)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, (y_pred > 0.5).astype(int))),
        'samples_train': len(X_train),
        'samples_val': len(X_val),
        'samples_test': len(X_test)
    }
    
    # Save
    task.update_state(state='PROGRESS', meta={'status': 'Saving LSTM model...', 'progress': 92})
    model_path = f"models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    lstm.save_model(model_path)
    
    return metrics, model_path


def _train_transformer(df, model_name, hyperparams, task):
    """Train Transformer model (placeholder for now)"""
    task.update_state(state='PROGRESS', meta={'status': 'Training Transformer', 'progress': 50})
    
    # For now, use XGBoost as fallback
    # Full transformer implementation would go here
    logger.warning("Transformer not fully implemented, falling back to XGBoost")
    return _train_xgboost(df, model_name, hyperparams, task)


@celery_app.task(bind=True)
def scheduled_retrain(self):
    """
    Spec 12: Weekly retraining task
    """
    # Trigger retraining for all active instruments
    logger.info("Scheduled retrain started")
    
    model_name = f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d')}"
    return train_model.delay(model_name=model_name, model_type='xgboost')
