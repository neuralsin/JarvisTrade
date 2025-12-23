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
import os  # CRITICAL: needed for model saving

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
        
        # Set default date range based on actual data in DB
        if not end_date or not start_date:
            from sqlalchemy import func
            # Query actual date range from features table  
            date_range = db.query(
                func.min(Feature.ts_utc).label('min_date'),
                func.max(Feature.ts_utc).label('max_date')
            ).first()
            
            if date_range and date_range.min_date and date_range.max_date:
                if not start_date:
                    start_date = date_range.min_date.strftime('%Y-%m-%d')
                if not end_date:
                    end_date = date_range.max_date.strftime('%Y-%m-%d')
                logger.info(f"Using actual DB date range: {start_date} to {end_date}")
            else:
                # Fallback if no features exist
                if not end_date:
                    end_date = datetime.utcnow().strftime('%Y-%m-%d')
                if not start_date:
                    start_dt = datetime.utcnow() - timedelta(days=730)
                    start_date = start_dt.strftime('%Y-%m-%d')
                logger.warning(f"No features in DB, using fallback dates: {start_date} to {end_date}")
        
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
        
        # CRITICAL VALIDATION: Ensure we have enough data for training
        MIN_SAMPLES = 100  # Minimum samples needed for reliable training
        if len(df) < MIN_SAMPLES:
            error_msg = f"Insufficient training data: {len(df)} samples (need at least {MIN_SAMPLES}). "
            if instrument_filter:
                error_msg += f"Try training without instrument filter or expanding date range."
            else:
                error_msg += f"Please ingest more historical data."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate target distribution
        if 'target' not in df.columns:
            raise ValueError("No 'target' column found in features. Check labeling logic.")
        
        target_dist = df['target'].value_counts()
        if len(target_dist) < 2:
            raise ValueError(f"Target has only one class ({target_dist.index[0]}). Cannot train. Need both buy/sell signals.")
        
        # Check for reasonable class balance (warn if too imbalanced)
        pos_ratio = (df['target'] == 1).sum() / len(df)
        if pos_ratio < 0.01 or pos_ratio > 0.99:
            logger.warning(f"Severe class imbalance: {pos_ratio*100:.1f}% positive samples. Training may be poor.")
        
        # Validate essential features exist
        essential_features = ['rsi_14', 'atr_percent', 'volume_ratio']
        missing_features = [f for f in essential_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}. Feature engineering may be incomplete.")
        
        logger.info(f"✓ Validation passed: {len(df)} samples, {pos_ratio*100:.1f}% positive class")
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
    
    # Store training distributions for drift detection
    feature_distributions = {}
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    for col in feature_cols[:10]:  # First 10 features
        if col in df.columns:
            feature_distributions[col] = df[col].dropna().values.tolist()[:1000]
    
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, (y_pred > 0.5).astype(int))),
        'samples_train': len(X_train),
        'samples_val': len(X_val),
        'samples_test': len(X_test),
        'feature_distributions': feature_distributions
    }
    
    # Save
    task.update_state(state='PROGRESS', meta={'status': 'Saving LSTM model...', 'progress': 92})
    model_path = f"models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    lstm.save_model(model_path)
    
    return metrics, model_path


def _train_transformer(df, model_name, hyperparams, task):
    """Train Transformer model with multi-head attention"""
    task.update_state(state='PROGRESS', meta={'status': 'Training Transformer', 'progress': 50})
    
    from app.ml.transformer_model import TransformerPredictor
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    logger.info("Starting Transformer model training")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    X = df[feature_cols].values
    y = df['target'].values
    
    # Split data: 70% train, 15% val, 15% test
    total_samples = len(df)
    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Get hyperparameters
    seq_length = hyperparams.get('seq_length', 20)
    d_model = hyperparams.get('d_model', 128)
    num_heads = hyperparams.get('num_heads', 8)
    num_layers = hyperparams.get('num_layers', 3)
    dff = hyperparams.get('dff', 256)
    epochs = hyperparams.get('n_estimators', 100)  # Reuse n_estimators
    batch_size = hyperparams.get('batch_size', 32)
    
    # Initialize transformer
    transformer = TransformerPredictor(
        seq_length=seq_length,
        features=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dff=dff
    )
    
    # Build model
    transformer.build_model()
    logger.info(f"Transformer model built with {transformer.model.count_params():,} parameters")
    
    # Prepare sequences
    task.update_state(state='PROGRESS', meta={'status': 'Preparing sequences', 'progress': 60})
    X_train_seq, y_train_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_train, y_train]), columns=feature_cols + ['target'])
    )
    X_val_seq, y_val_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_val, y_val]), columns=feature_cols + ['target'])
    )
    X_test_seq, y_test_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_test, y_test]), columns=feature_cols + ['target'])
    )
    
    logger.info(f"Sequences prepared - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # Train model
    task.update_state(state='PROGRESS', meta={'status': 'Training Transformer network', 'progress': 70})
    history = transformer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating model', 'progress': 90})
    y_pred_proba = transformer.predict(X_test_seq).flatten()
    y_pred = (y_pred_proba > 0.3).astype(int)  # Use lower threshold like XGBoost
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test_seq, y_pred_proba)
    test_acc = accuracy_score(y_test_seq, y_pred)
    test_precision = precision_score(y_test_seq, y_pred, zero_division=0)
    test_recall = recall_score(y_test_seq, y_pred, zero_division=0)
    
    val_loss = history.history['val_loss'][-1]
    val_auc = history.history['val_auc'][-1]
    
    # Store training distributions for drift detection
    feature_distributions = {}
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    for col in feature_cols[:10]:  # First 10 features
        if col in df.columns:
            feature_distributions[col] = df[col].dropna().values.tolist()[:1000]
    
    metrics = {
        'train_auc': history.history['auc'][-1],
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'val_loss': val_loss,
        'epochs_trained': len(history.history['loss']),
        'total_params': int(transformer.model.count_params()),
        'feature_distributions': feature_distributions
    }
    
    # Calculate class metrics
    true_positives = int(((y_pred == 1) & (y_test_seq == 1)).sum())
    false_positives = int(((y_pred == 1) & (y_test_seq == 0)).sum())
    true_negatives = int(((y_pred == 0) & (y_test_seq == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_test_seq == 1)).sum())
    
    logger.info(f"Transformer Test Metrics - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    logger.info(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    
    # Save model
    model_path = f"/app/models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    transformer.save_model(model_path)
    
    return metrics, model_path


@celery_app.task(bind=True)
def scheduled_retrain(self):
    """
    Spec 12: Weekly retraining task
    """
    # Trigger retraining for all active instruments
    logger.info("Scheduled retrain started")
    
    model_name = f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d')}"
    return train_model.delay(model_name=model_name, model_type='xgboost')
