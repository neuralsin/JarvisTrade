"""
Issue #2 Fix: Signal Generation Task
Creates and updates signals in the signals table from active models
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import User, Instrument, Model, HistoricalCandle, Feature
from sqlalchemy import text
from datetime import datetime, timedelta
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def generate_signals(self):
    """
    Generate trading signals from all active models
    Runs periodically (every 60 seconds) to check for BUY/SELL signals
    
    Process:
    1. Get all active users with paper_trading_enabled
    2. For each user, get their selected_model_ids
    3. For each model, fetch latest features
    4. Run model.predict() and create Signal record
    """
    db = SessionLocal()
    
    try:
        # Get all users with paper trading enabled
        users = db.query(User).filter(User.paper_trading_enabled == True).all()
        
        if not users:
            logger.info("No users with paper trading enabled")
            return {"status": "no_users"}
        
        total_signals = 0
        
        for user in users:
            selected_ids = user.selected_model_ids or []
            
            if not selected_ids:
                logger.debug(f"User {user.email} has no models selected")
                continue
            
            # Get active models
            models = db.query(Model).filter(
                Model.id.in_(selected_ids),
                Model.is_active == True
            ).all()
            
            logger.info(f"Checking {len(models)} models for user {user.email}")
            
            for model in models:
                try:
                    # Get instrument
                    instrument = db.query(Instrument).filter(
                        Instrument.symbol == model.stock_symbol
                    ).first()
                    
                    if not instrument:
                        logger.warning(f"Instrument {model.stock_symbol} not found")
                        continue
                    
                    # Get latest feature
                    latest_feature = db.query(Feature).filter(
                        Feature.instrument_id == instrument.id
                    ).order_by(Feature.ts_utc.desc()).first()
                    
                    if not latest_feature:
                        logger.warning(f"No features for {instrument.symbol}")
                        continue
                    
                    # Check if feature is fresh (< 2 minutes old)
                    age = (datetime.utcnow() - latest_feature.ts_utc.replace(tzinfo=None)).total_seconds()
                    if age > 120:
                        logger.warning(f"Features too old for {instrument.symbol}: {age}s")
                        continue
                    
                    # Load model
                    try:
                        model_obj = joblib.load(model.artifact_path)
                    except Exception as e:
                        logger.error(f"Failed to load model {model.name}: {e}")
                        continue
                    
                    # Prepare features for prediction
                    feature_dict = latest_feature.feature_json  # ✅ FIXED: Use feature_json not feature_values
                    feature_cols = [
                        'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_200',
                        'distance_from_ema200', 'rsi_14', 'rsi_slope',
                        'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix',
                        'sentiment_1d', 'sentiment_3d', 'sentiment_7d'
                    ]
                    
                    X = pd.DataFrame([feature_dict])[feature_cols].fillna(0)
                    
                    # Predict
                    proba = model_obj.predict_proba(X)[0]
                    
                    # Handle both binary and multi-class
                    if len(proba) == 3:
                        # Multi-class
                        predicted_class = int(model_obj.predict(X)[0])
                        confidence = float(proba[predicted_class])
                        
                        class_to_signal = {0: "HOLD", 1: "BUY", 2: "SELL"}
                        signal_type = class_to_signal[predicted_class]
                        
                    elif len(proba) == 2:
                        # Binary
                        confidence = float(proba[1])
                        signal_type = "BUY" if confidence >= 0.3 else "HOLD"
                    else:
                        logger.error(f"Unexpected proba shape for model {model.name}")
                        continue
                    
                    # Only create signal if BUY or SELL (not HOLD)
                    if signal_type in ["BUY", "SELL"]:
                        # Check if signal already exists (recent)
                        existing = db.execute(text("""
                            SELECT id FROM signals 
                            WHERE model_id = :model_id 
                            AND instrument_id = :instrument_id
                            AND timestamp > NOW() - INTERVAL '5 minutes'
                            AND executed = false
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """), {
                            "model_id": str(model.id),
                            "instrument_id": str(instrument.id)
                        }).fetchone()
                        
                        if existing:
                            logger.debug(f"Recent signal exists for {instrument.symbol}")
                            continue
                        
                        # Create new signal
                        db.execute(text("""
                            INSERT INTO signals (model_id, instrument_id, signal_type, confidence, timestamp, executed)
                            VALUES (:model_id, :instrument_id, :signal_type, :confidence, NOW(), false)
                        """), {
                            "model_id": str(model.id),
                            "instrument_id": str(instrument.id),
                            "signal_type": signal_type,
                            "confidence": confidence
                        })
                        db.commit()
                        
                        total_signals += 1
                        logger.info(f"✓ Generated {signal_type} signal for {instrument.symbol} (conf: {confidence:.2f})")
                
                except Exception as e:
                    logger.error(f"Error processing model {model.name}: {e}")
                    db.rollback()
                    continue
        
        logger.info(f"Signal generation complete: {total_signals} new signals")
        return {
            "status": "success",
            "signals_generated": total_signals
        }
    
    except Exception as e:
        logger.error(f"Signal generation failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(e)}
    
    finally:
        db.close()
