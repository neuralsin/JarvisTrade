"""
Signal Generation Task - DETERMINISTIC VERSION

Generates trading signals (BUY/SELL/HOLD) from active models.

Principles:
1. Use ORM for all database operations (no raw SQL)
2. Handle all model types (XGBoost, LSTM, Transformer)
3. Explicit failure on stale data
4. WebSocket broadcast on signal generation
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import User, Instrument, Model, Signal, Feature
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - FIX 4 & 5: Removed sentiment, changed EMA200 -> EMA100
# ============================================================================
FEATURE_MAX_AGE_SECONDS = 3600  # 1 hour - features older than this are stale
FEATURE_COLUMNS = [
    'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_100',
    'distance_from_ema100', 'rsi_14', 'rsi_slope',
    'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
    # Sentiment features REMOVED - contaminating training with 0.0 bias
]


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_record: Model):
    """
    Load model based on its format.
    
    Args:
        model_record: Model ORM object with model_path and model_format
    
    Returns:
        Loaded model object that has predict/predict_proba methods
    """
    import os
    import joblib
    
    model_path = model_record.model_path
    model_format = model_record.model_format or 'joblib'  # Default for older models
    model_type = model_record.model_type
    
    if not model_path:
        raise ValueError(f"Model {model_record.name} has no model_path")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_format == 'joblib':
        # XGBoost - single file
        return joblib.load(model_path)
    
    elif model_format == 'keras':
        # LSTM or Transformer - directory
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
        else:
            raise ValueError(f"Unknown model type for keras format: {model_type}")
    
    else:
        raise ValueError(f"Unknown model format: {model_format}")


def predict_with_model(model, model_type: str, features: pd.DataFrame) -> tuple:
    """
    Run prediction with any model type.
    
    CRITICAL FIX: Binary models predict trade-worthiness, NOT direction.
    - 1 = TRADE worth taking
    - 0 = NO TRADE
    
    Therefore:
    - High confidence → BUY (enter trade)
    - Low confidence → HOLD (do not enter)
    - SELL is NOT a valid prediction (it's position management)
    
    Args:
        model: Loaded model object
        model_type: 'xgboost', 'lstm', or 'transformer'
        features: DataFrame with feature columns
    
    Returns:
        tuple: (signal_type, confidence)
            signal_type: 'BUY' or 'HOLD' (SELL removed from ML signals)
            confidence: float between 0 and 1
    """
    if model_type == 'xgboost':
        proba = model.predict_proba(features)[0]
        
        if len(proba) == 3:
            # Multi-class: [HOLD, BUY, SELL] - DEPRECATED but kept for compatibility
            predicted_class = int(model.predict(features)[0])
            confidence = float(proba[predicted_class])
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'HOLD'}  # SELL → HOLD (deprecated)
            return signal_map[predicted_class], confidence
        else:
            # Binary: TRADE vs NO_TRADE
            confidence = float(proba[1])
            
            # CRITICAL FIX: Use configurable threshold instead of hardcoded 0.65
            from app.config import settings
            if confidence >= settings.PROB_MIN:
                return 'BUY', confidence
            else:
                return 'HOLD', 1.0 - confidence  # Confidence in HOLD
    
    elif model_type in ['lstm', 'transformer']:
        # Neural networks output probability of TRADE
        X = features.values.reshape(1, -1)
        
        # For sequence models, we need to prepare sequences
        if hasattr(model, 'prepare_sequences'):
            import numpy as np
            seq_length = getattr(model, 'seq_length', 20)
            X = np.tile(X, (seq_length, 1)).reshape(1, seq_length, -1)
        
        proba = float(model.predict(X).flatten()[0])
        
        # CRITICAL FIX: Neural networks also use configurable threshold
        from app.config import settings
        if proba >= settings.PROB_MIN:
            return 'BUY', proba
        else:
            return 'HOLD', 1.0 - proba
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# MAIN TASK
# ============================================================================
@celery_app.task(bind=True)
def generate_signals(self):
    """
    Generate trading signals from all active models.
    
    Process:
    1. Get all users with paper_trading_enabled
    2. For each user, get their selected active models
    3. For each model, check if we have fresh features for its stock
    4. Run prediction and create Signal record (using ORM)
    5. Broadcast signal via WebSocket
    
    Returns:
        Dict with generation stats
    """
    db = SessionLocal()
    
    try:
        stats = {
            'users_checked': 0,
            'models_checked': 0,
            'signals_generated': 0,
            'errors': []
        }
        
        # Get users with paper trading enabled
        users = db.query(User).filter(User.paper_trading_enabled == True).all()
        
        if not users:
            logger.info("No users with paper trading enabled")
            return {"status": "no_users", "stats": stats}
        
        stats['users_checked'] = len(users)
        
        for user in users:
            try:
                _process_user_signals(db, user, stats)
            except Exception as e:
                logger.error(f"Error processing user {user.email}: {str(e)}")
                stats['errors'].append(f"User {user.email}: {str(e)}")
                continue
        
        logger.info(
            f"Signal generation complete - "
            f"Users: {stats['users_checked']}, "
            f"Models: {stats['models_checked']}, "
            f"Signals: {stats['signals_generated']}"
        )
        
        return {"status": "success", "stats": stats}
    
    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}", exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(e)}
    
    finally:
        db.close()


def _process_user_signals(db, user: User, stats: dict):
    """Process signals for a single user"""
    selected_ids = user.selected_model_ids or []
    
    if not selected_ids:
        logger.debug(f"User {user.email} has no models selected")
        return
    
    # Get active models from selection
    models = db.query(Model).filter(
        Model.id.in_(selected_ids),
        Model.is_active == True
    ).all()
    
    if not models:
        logger.debug(f"User {user.email} has no active models in selection")
        return
    
    logger.info(f"Processing {len(models)} models for user {user.email}")
    
    for model_record in models:
        try:
            _process_model_signal(db, user, model_record, stats)
            stats['models_checked'] += 1
        except Exception as e:
            logger.error(f"Error with model {model_record.name}: {str(e)}")
            stats['errors'].append(f"Model {model_record.name}: {str(e)}")
            continue


def _process_model_signal(db, user: User, model_record: Model, stats: dict):
    """Generate signal for a single model"""
    stock_symbol = model_record.stock_symbol
    
    if not stock_symbol:
        logger.warning(f"Model {model_record.name} has no stock_symbol - skipping")
        return
    
    # Get instrument
    instrument = db.query(Instrument).filter(
        Instrument.symbol == stock_symbol
    ).first()
    
    if not instrument:
        logger.warning(f"Instrument {stock_symbol} not found - skipping")
        return
    
    # Get latest feature
    latest_feature = db.query(Feature).filter(
        Feature.instrument_id == instrument.id
    ).order_by(Feature.ts_utc.desc()).first()
    
    if not latest_feature:
        logger.warning(f"No features for {stock_symbol} - skipping")
        return
    
    # Check feature freshness
    feature_age = (datetime.utcnow() - latest_feature.ts_utc.replace(tzinfo=None)).total_seconds()
    
    if feature_age > FEATURE_MAX_AGE_SECONDS:
        logger.error(
            f"Features for {stock_symbol} are stale ({feature_age:.0f}s old, max {FEATURE_MAX_AGE_SECONDS}s) - ABORTING SIGNAL"
        )
        return  # Bug fix: Hard fail on stale data instead of warning
    
    # Check for recent signal (avoid duplicates)
    recent_signal = db.query(Signal).filter(
        Signal.model_id == model_record.id,
        Signal.instrument_id == instrument.id,
        Signal.timestamp >= datetime.utcnow() - timedelta(minutes=5),
        Signal.executed == False
    ).first()
    
    if recent_signal:
        logger.debug(f"Recent unexecuted signal exists for {stock_symbol} - skipping")
        return
    
    # Load model
    try:
        model = load_model(model_record)
    except Exception as e:
        logger.error(f"Failed to load model {model_record.name}: {str(e)}")
        return
    
    # Prepare features
    feature_dict = latest_feature.feature_json or {}
    
    # Build feature DataFrame
    features_data = {col: feature_dict.get(col, 0.0) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([features_data])
    
    # Run prediction
    try:
        signal_type, confidence = predict_with_model(
            model, model_record.model_type, X
        )
    except Exception as e:
        logger.error(f"Prediction failed for {model_record.name}: {str(e)}")
        return
    
    # Only create signal for BUY or SELL
    if signal_type == 'HOLD':
        logger.debug(f"{stock_symbol}: HOLD (confidence: {confidence:.2f}) - no signal created")
        return
    
    # Create signal using ORM
    signal = Signal(
        model_id=model_record.id,
        instrument_id=instrument.id,
        signal_type=signal_type,
        confidence=confidence,
        executed=False
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    
    stats['signals_generated'] += 1
    logger.info(f"✓ Generated {signal_type} signal for {stock_symbol} (conf: {confidence:.2f})")
    
    # Broadcast via WebSocket (async)
    try:
        _broadcast_signal(signal, stock_symbol, model_record.name)
    except Exception as e:
        logger.warning(f"WebSocket broadcast failed: {str(e)}")


def _broadcast_signal(signal: Signal, stock_symbol: str, model_name: str):
    """Broadcast signal to connected WebSocket clients"""
    try:
        from app.websocket_manager import ws_manager
        import asyncio
        
        # Create event loop if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        message = {
            "event": "signal_generated",
            "data": {
                "id": str(signal.id),
                "stock_symbol": stock_symbol,
                "model_name": model_name,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else None
            }
        }
        
        # Broadcast to all connected clients
        if hasattr(ws_manager, 'broadcast'):
            loop.run_until_complete(ws_manager.broadcast(message))
            logger.debug(f"Signal broadcast sent for {stock_symbol}")
    
    except Exception as e:
        logger.debug(f"WebSocket broadcast skipped: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@celery_app.task
def generate_signal_for_model(model_id: str):
    """
    Generate signal for a specific model (for manual triggering).
    
    Args:
        model_id: UUID of the model
    """
    db = SessionLocal()
    
    try:
        model_record = db.query(Model).filter(Model.id == model_id).first()
        
        if not model_record:
            return {"status": "error", "error": "Model not found"}
        
        if not model_record.is_active:
            return {"status": "error", "error": "Model is not active"}
        
        # Get first user with paper trading (for single-model testing)
        user = db.query(User).filter(User.paper_trading_enabled == True).first()
        
        if not user:
            return {"status": "error", "error": "No users with paper trading enabled"}
        
        stats = {'signals_generated': 0, 'errors': []}
        _process_model_signal(db, user, model_record, stats)
        
        return {"status": "success", "stats": stats}
    
    finally:
        db.close()
