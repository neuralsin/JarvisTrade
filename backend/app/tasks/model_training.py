"""
Model Training Task - DETERMINISTIC VERSION

Principles:
1. One model = one stock (enforced)
2. Explicit failure - no silent errors
3. Unified model save with format tracking
4. Atomic DB commit (save file, then DB)
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
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import joblib
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - Single source of truth
# FIX 4 & 5: Removed sentiment features, changed EMA200 -> EMA100
# ============================================================================
MODELS_DIR = Path("models")
MIN_TRAINING_SAMPLES = 100
FEATURE_COLUMNS = [
    'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_100',
    'distance_from_ema100', 'rsi_14', 'rsi_slope',
    'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
    # Sentiment features REMOVED - contaminating training with 0.0 bias
]


# ============================================================================
# MODEL SAVE/LOAD UTILITIES
# ============================================================================
def save_model_artifact(model, model_name: str, model_type: str, stock_symbol: str) -> tuple[str, str]:
    """
    Save model artifact to disk with deterministic path.
    
    Returns:
        tuple: (model_path, model_format)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(exist_ok=True)
    
    if model_type == 'xgboost':
        # XGBoost: single joblib file
        model_path = str(MODELS_DIR / f"{stock_symbol}_{model_name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        model_format = 'joblib'
        logger.info(f"✓ Saved XGBoost model to {model_path}")
        
    elif model_type == 'lstm':
        # LSTM: Keras directory
        model_path = str(MODELS_DIR / f"{stock_symbol}_{model_name}_{timestamp}_lstm")
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_path)
        model_format = 'keras'
        logger.info(f"✓ Saved LSTM model to {model_path}")
        
    elif model_type == 'transformer':
        # Transformer: Keras directory
        model_path = str(MODELS_DIR / f"{stock_symbol}_{model_name}_{timestamp}_transformer")
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_path)
        model_format = 'keras'
        logger.info(f"✓ Saved Transformer model to {model_path}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Verify file exists
    if model_type == 'xgboost':
        if not os.path.isfile(model_path):
            raise IOError(f"Model file was not created: {model_path}")
    else:
        if not os.path.isdir(model_path):
            raise IOError(f"Model directory was not created: {model_path}")
    
    return model_path, model_format


def load_model_artifact(model_path: str, model_format: str, model_type: str):
    """
    Load model artifact from disk.
    
    Args:
        model_path: Path to model file/directory
        model_format: 'joblib' or 'keras'
        model_type: 'xgboost', 'lstm', or 'transformer'
    
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if model_format == 'joblib':
        return joblib.load(model_path)
    
    elif model_format == 'keras':
        if model_type == 'lstm':
            from app.ml.lstm_model import LSTMPredictor
            lstm = LSTMPredictor()
            lstm.load_model(model_path)
            return lstm
        elif model_type == 'transformer':
            from app.ml.transformer_model import TransformerPredictor
            transformer = TransformerPredictor()
            transformer.load_model(model_path)
            return transformer
    
    raise ValueError(f"Unknown model format: {model_format}")


# ============================================================================
# DATA FETCHING - Yahoo Finance only
# ============================================================================
def fetch_training_data(
    db,
    stock_symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    task_callback=None
) -> pd.DataFrame:
    """
    Fetch data for training from Yahoo Finance.
    
    Returns DataFrame with OHLCV data or raises exception.
    """
    from app.tasks.data_ingestion import fetch_historical_yahoo
    
    if task_callback:
        task_callback.update_state(state='PROGRESS', meta={
            'status': f'Fetching Yahoo Finance data for {stock_symbol}...',
            'progress': 15
        })
    
    # Ensure instrument exists
    instrument = db.query(Instrument).filter(Instrument.symbol == stock_symbol).first()
    if not instrument:
        logger.info(f"Creating instrument record for {stock_symbol}")
        instrument = Instrument(
            symbol=stock_symbol,
            name=stock_symbol,
            exchange='NSE',
            instrument_type='EQ'
        )
        db.add(instrument)
        db.commit()
        db.refresh(instrument)
    
    # Fetch from Yahoo Finance
    logger.info(f"Fetching {interval} data for {stock_symbol}: {start_date} to {end_date}")
    df = fetch_historical_yahoo(
        symbol=stock_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        exchange='NS'
    )
    
    if df is None or df.empty:
        raise ValueError(
            f"No data available for {stock_symbol} with interval={interval}. "
            f"The symbol may not exist on Yahoo Finance (try adding .NS suffix) "
            f"or the date range may be invalid for this interval."
        )
    
    # Normalize DataFrame
    df = df.reset_index()
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'ts_utc'})
    elif 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ts_utc'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ts_utc'})
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    logger.info(f"✓ Fetched {len(df)} candles for {stock_symbol}")
    return df, instrument


# ============================================================================
# MAIN TRAINING TASK
# ============================================================================
@celery_app.task(bind=True, time_limit=7200, soft_time_limit=6000)
def train_model(
    self, 
    model_name: str,
    instrument_filter: str,
    model_type: str = 'xgboost',
    interval: str = '15m',
    hyperparams: dict = None,
    start_date: str = None,
    end_date: str = None,
    csv_dataset_id: str = None
):
    """
    Train ML model for a SINGLE stock.
    
    DETERMINISTIC BEHAVIOR:
    1. Validate inputs
    2. Fetch/load data
    3. Compute features
    4. Train model
    5. Save model to disk
    6. Save to database (atomic)
    
    Args:
        model_name: Name for the model
        instrument_filter: Stock symbol (REQUIRED) - e.g., 'RELIANCE'
        model_type: 'xgboost', 'lstm', or 'transformer'
        interval: Candle interval ('15m', '1h', '1d', etc.)
        hyperparams: Optional hyperparameter overrides
        start_date: Training start date (YYYY-MM-DD)
        end_date: Training end date (YYYY-MM-DD)
        csv_dataset_id: Optional CSV dataset ID for training from uploaded data
    
    Returns:
        Dict with model_id, metrics, and stock_symbol
    """
    db = SessionLocal()
    task_id = self.request.id
    stock_symbol = instrument_filter.strip().upper() if instrument_filter else None
    hyperparams = hyperparams or {}
    
    try:
        # =====================================================
        # STEP 1: Validate inputs
        # =====================================================
        if not stock_symbol:
            raise ValueError(
                "instrument_filter is REQUIRED. Each model must be trained on exactly one stock. "
                "Example: instrument_filter='RELIANCE'"
            )
        
        logger.info(f"[{task_id}] Starting {model_type} training for {stock_symbol}: {model_name}")
        self.update_state(state='PROGRESS', meta={
            'status': f'Validating inputs for {stock_symbol}...',
            'progress': 5
        })
        
        # Set default dates
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            # Default based on interval
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                days_back = 59  # Yahoo Finance limit for intraday
            else:
                days_back = 365
            start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Date range: {start_date} to {end_date}, interval: {interval}")
        
        # =====================================================
        # STEP 2: Load training data
        # =====================================================
        self.update_state(state='PROGRESS', meta={
            'status': f'Loading data for {stock_symbol}...',
            'progress': 10
        })
        
        if csv_dataset_id:
            # Load from CSV
            df, instrument = _load_csv_data(db, csv_dataset_id, stock_symbol, self)
        else:
            # Fetch from Yahoo Finance
            df, instrument = fetch_training_data(
                db, stock_symbol, interval, start_date, end_date, self
            )
        
        # =====================================================
        # STEP 3: Compute features
        # =====================================================
        self.update_state(state='PROGRESS', meta={
            'status': f'Computing features for {len(df)} candles...',
            'progress': 25
        })
        
        df_features = compute_features(df, str(instrument.id), db)
        
        if df_features.empty:
            raise ValueError(
                f"Feature computation returned empty DataFrame. "
                f"Need at least 200 candles for EMA200 calculation. Got: {len(df)}"
            )
        
        # Generate labels
        df_labeled = generate_labels(df_features)
        
        if df_labeled.empty or 'target' not in df_labeled.columns:
            raise ValueError("Label generation failed - no target column")
        
        # Remove rows with null targets
        df_labeled = df_labeled[df_labeled['target'].notna()].copy()
        
        if len(df_labeled) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {len(df_labeled)} samples "
                f"(need at least {MIN_TRAINING_SAMPLES})"
            )
        
        logger.info(f"✓ Prepared {len(df_labeled)} samples with labels")
        
        # =====================================================
        # STEP 4: Train model
        # =====================================================
        self.update_state(state='PROGRESS', meta={
            'status': f'Training {model_type} model on {len(df_labeled)} samples...',
            'progress': 40
        })
        
        if model_type == 'xgboost':
            model, metrics = _train_xgboost(df_labeled, hyperparams, self)
        elif model_type == 'lstm':
            model, metrics = _train_lstm(df_labeled, hyperparams, self)
        elif model_type == 'transformer':
            model, metrics = _train_transformer(df_labeled, hyperparams, self)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'xgboost', 'lstm', or 'transformer'")
        
        # =====================================================
        # STEP 5: Save model to disk
        # =====================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Saving model to disk...',
            'progress': 90
        })
        
        model_path, model_format = save_model_artifact(model, model_name, model_type, stock_symbol)
        
        # =====================================================
        # STEP 6: Save to database (atomic)
        # =====================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Saving model metadata to database...',
            'progress': 95
        })
        
        # Convert numpy types to native Python
        metrics_clean = _convert_to_native(metrics)
        
        # Check if model should be auto-activated
        should_activate = False
        if settings.AUTO_ACTIVATE_MODELS:
            auc = metrics_clean.get('auc_roc', metrics_clean.get('test_auc', 0))
            accuracy = metrics_clean.get('test_accuracy', metrics_clean.get('accuracy', 0))
            if auc >= settings.MODEL_MIN_AUC and accuracy >= settings.MODEL_MIN_ACCURACY:
                should_activate = True
                logger.info(f"Model meets quality thresholds (AUC: {auc:.4f}) - auto-activating")
        
        # Create model record
        model_record = Model(
            name=model_name,
            model_type=model_type,
            model_path=model_path,
            model_format=model_format,
            stock_symbol=stock_symbol,
            trained_at=datetime.utcnow(),
            metrics_json=metrics_clean,
            is_active=should_activate
        )
        db.add(model_record)
        db.commit()
        db.refresh(model_record)
        
        logger.info(f"✓ Model saved: {model_name} (ID: {model_record.id})")
        
        return {
            "status": "success",
            "model_id": str(model_record.id),
            "model_name": model_name,
            "model_type": model_type,
            "model_format": model_format,
            "stock_symbol": stock_symbol,
            "model_path": model_path,
            "metrics": metrics_clean,
            "samples": len(df_labeled),
            "date_range": f"{start_date} to {end_date}",
            "auto_activated": should_activate
        }
    
    except Exception as e:
        logger.error(f"[{task_id}] Training failed: {str(e)}", exc_info=True)
        db.rollback()
        raise
    
    finally:
        db.close()


# ============================================================================
# TRAINING IMPLEMENTATIONS
# ============================================================================
def _train_xgboost(df: pd.DataFrame, hyperparams: dict, task) -> tuple:
    """Train XGBoost model and return (model, metrics)"""
    task.update_state(state='PROGRESS', meta={
        'status': 'Preparing XGBoost training data...',
        'progress': 45
    })
    
    trainer = ModelTrainer(hyperparams=hyperparams)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = trainer.prepare_data(df)
    
    task.update_state(state='PROGRESS', meta={
        'status': f'Training XGBoost on {len(X_train)} samples...',
        'progress': 55
    })
    
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Evaluating XGBoost performance...',
        'progress': 80
    })
    
    metrics = trainer.evaluate(model, X_test, y_test, feature_cols)
    metrics['samples_train'] = len(X_train)
    metrics['samples_val'] = len(X_val)
    metrics['samples_test'] = len(X_test)
    
    return model, metrics


def _train_lstm(df: pd.DataFrame, hyperparams: dict, task) -> tuple:
    """Train LSTM model and return (model, metrics)"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Initializing LSTM neural network...',
        'progress': 45
    })
    
    lstm = LSTMPredictor()
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Creating time-series sequences...',
        'progress': 50
    })
    
    X, y = lstm.prepare_sequences(df)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    epochs = hyperparams.get('epochs', 50)
    
    task.update_state(state='PROGRESS', meta={
        'status': f'Training LSTM ({epochs} epochs)...',
        'progress': 60
    })
    
    history = lstm.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Evaluating LSTM performance...',
        'progress': 80
    })
    
    y_pred = lstm.predict(X_test)
    
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, (y_pred > 0.5).astype(int))),
        'samples_train': len(X_train),
        'samples_val': len(X_val),
        'samples_test': len(X_test),
        'epochs_trained': len(history.history['loss'])
    }
    
    return lstm, metrics


def _train_transformer(df: pd.DataFrame, hyperparams: dict, task) -> tuple:
    """Train Transformer model and return (model, metrics)"""
    from app.ml.transformer_model import TransformerPredictor
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Initializing Transformer model...',
        'progress': 45
    })
    
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    X = df[feature_cols].values
    y = df['target'].values
    
    # Split data
    train_size = int(0.70 * len(df))
    val_size = int(0.15 * len(df))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    # Hyperparameters
    seq_length = hyperparams.get('seq_length', 20)
    d_model = hyperparams.get('d_model', 128)
    num_heads = hyperparams.get('num_heads', 8)
    num_layers = hyperparams.get('num_layers', 3)
    dff = hyperparams.get('dff', 256)
    epochs = hyperparams.get('epochs', 100)
    batch_size = hyperparams.get('batch_size', 32)
    
    transformer = TransformerPredictor(
        seq_length=seq_length,
        features=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dff=dff
    )
    transformer.build_model()
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Preparing Transformer sequences...',
        'progress': 55
    })
    
    X_train_seq, y_train_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_train, y_train]), columns=feature_cols + ['target'])
    )
    X_val_seq, y_val_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_val, y_val]), columns=feature_cols + ['target'])
    )
    X_test_seq, y_test_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_test, y_test]), columns=feature_cols + ['target'])
    )
    
    task.update_state(state='PROGRESS', meta={
        'status': f'Training Transformer ({epochs} epochs)...',
        'progress': 65
    })
    
    history = transformer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=epochs,
        batch_size=batch_size
    )
    
    task.update_state(state='PROGRESS', meta={
        'status': 'Evaluating Transformer performance...',
        'progress': 85
    })
    
    y_pred_proba = transformer.predict(X_test_seq).flatten()
    y_pred = (y_pred_proba > 0.3).astype(int)
    
    # Calculate AUC - handle multi-class case
    try:
        # Check if binary or multi-class
        unique_classes = np.unique(y_test_seq)
        if len(unique_classes) == 2:
            auc = float(roc_auc_score(y_test_seq, y_pred_proba))
        else:
            # For multi-class, convert to binary (BUY vs not-BUY)
            y_binary = (y_test_seq == 1).astype(int)  # 1 = BUY
            auc = float(roc_auc_score(y_binary, y_pred_proba))
    except Exception as e:
        logger.warning(f"AUC calculation failed: {e}. Using accuracy-based estimate.")
        auc = float(accuracy_score(y_test_seq, y_pred))
    
    metrics = {
        'auc_roc': auc,
        'test_accuracy': float(accuracy_score(y_test_seq, y_pred)),
        'test_precision': float(precision_score(y_test_seq, y_pred, zero_division=0, average='binary' if len(np.unique(y_test_seq)) == 2 else 'weighted')),
        'test_recall': float(recall_score(y_test_seq, y_pred, zero_division=0, average='binary' if len(np.unique(y_test_seq)) == 2 else 'weighted')),
        'samples_train': len(X_train_seq),
        'samples_val': len(X_val_seq),
        'samples_test': len(X_test_seq),
        'epochs_trained': len(history.history['loss']),
        'total_params': int(transformer.model.count_params())
    }
    
    return transformer, metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _load_csv_data(db, csv_dataset_id: str, stock_symbol: str, task) -> tuple:
    """Load training data from CSV dataset"""
    from app.routers.csv_upload import load_csv_for_training
    
    task.update_state(state='PROGRESS', meta={
        'status': f'Loading CSV dataset {csv_dataset_id}...',
        'progress': 12
    })
    
    df = load_csv_for_training(csv_dataset_id, stock_symbol)
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Ensure instrument exists
    instrument = db.query(Instrument).filter(Instrument.symbol == stock_symbol).first()
    if not instrument:
        instrument = Instrument(
            symbol=stock_symbol,
            name=stock_symbol,
            exchange='CSV',
            instrument_type='EQ'
        )
        db.add(instrument)
        db.commit()
        db.refresh(instrument)
    
    return df, instrument


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# ============================================================================
# SCHEDULED RETRAIN
# ============================================================================
@celery_app.task(bind=True)
def scheduled_retrain(self):
    """Weekly retraining task"""
    logger.info("Scheduled retrain started")
    
    model_name = f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d')}"
    stock_symbol = settings.DEFAULT_RETRAIN_STOCK
    
    return train_model.delay(
        model_name=model_name,
        instrument_filter=stock_symbol,
        model_type='xgboost'
    )
