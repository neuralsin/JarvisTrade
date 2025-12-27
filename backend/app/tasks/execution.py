"""
Spec 6 & 12: Execution tasks - check signals and execute trades
Phase 3: Multi-model parallel execution (one model per stock)
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import User, Instrument, Trade, TradeLog, Feature
from app.db.signal_log import SignalLog  # Phase 3: For signal tracking
from app.ml.model_selector import get_active_models_for_user  # Phase 3: Multi-model support
from app.ml.feature_engineer import compute_features, extract_feature_vector
from app.trading.decision_engine import DecisionEngine
from app.trading.paper_simulator import PaperSimulator
from app.trading.risk_manager import RiskManager
from app.utils.mailer import send_trade_execution_email
from datetime import datetime, timedelta
import pandas as pd
import logging
import joblib

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def check_and_execute_signals(self):
    """
    Spec 6: Periodically check for trading signals and execute
    Phase 1: Enhanced with verbose logging
    Phase 3: Multi-model parallel execution (one model per stock)
    
    For each user:
    - Get their selected models (one per stock)
    - For each model, check signals for that model's stock
    - Execute trades if criteria met
    """
    db = SessionLocal()
    
    try:
        logger.info("="*80)
        logger.info("SIGNAL CHECK STARTED (Multi-Model)")
        logger.info("="*80)
        
        # Get users with auto-execute enabled
        users = db.query(User).filter(User.auto_execute == True).all()
        
        if not users:
            logger.warning("\u26a0\ufe0f  NO USERS WITH AUTO_EXECUTE ENABLED")
            logger.info("\ud83d\udcdd Run SQL: UPDATE users SET auto_execute = true WHERE email = 'your@email.com';")
            return {"status": "no_users"}
        
        logger.info(f"\u2713 Checking signals for {len(users)} user(s) with auto_execute enabled")
        
        total_models_checked = 0
        total_signals_generated = 0
        
        for user in users:
            try:
                logger.info(f"\\n--- Processing user: {user.email} ---")
                
                # Phase 3: Get user's selected models
                models = get_active_models_for_user(db, str(user.id))
                
                if not models:
                    logger.warning(f"  \u26a0\ufe0f  No models selected for {user.email}")
                    logger.info(f"  \ud83d\udca1 User needs to select models via API or UI")
                    continue
                
                logger.info(f"  \ud83d\udcca User has {len(models)} model(s) selected")
                
                # Phase 3: Check signals for each model (parallel per-stock)
                for model in models:
                    total_models_checked += 1
                    signals = _check_signals_for_model(db, user, model)
                    total_signals_generated += signals
                    
            except Exception as e:
                logger.error(f"\u274c Error checking signals for user {user.email}: {str(e)}", exc_info=True)
        
        logger.info("="*80)
        logger.info(f"SIGNAL CHECK COMPLETED: {total_models_checked} models, {total_signals_generated} signals")
        logger.info("="*80)
        return {
            "status": "success",
            "models_checked": total_models_checked,
            "signals_generated": total_signals_generated
        }
    
    finally:
        db.close()


def _check_signals_for_model(db, user, model_record):
    """
    Phase 3: Check and execute signals for a single model (stock-specific).
    
    This function is called once per model selected by the user.
    Each model only checks its assigned stock.
    
    Args:
        db: Database session
        user: User object
        model_record: Model database record (with stock_symbol)
    
    Returns:
        Number of signals generated (EXECUTE or REJECT)
    """
    from app.config import settings
    
    stock_symbol = model_record.stock_symbol
    logger.info(f"\\n    \ud83d\udd0d Model: {model_record.name} (Stock: {stock_symbol})")
    
    if not stock_symbol:
        logger.warning(f"      \u26a0\ufe0f  Model {model_record.name} has no stock_symbol - skipping")
        return 0
    
    # Get the instrument for this stock
    instrument = db.query(Instrument).filter(
        Instrument.symbol == stock_symbol
    ).first()
    
    if not instrument:
        logger.warning(f"      \u26a0\ufe0f  Instrument {stock_symbol} not found in database")
        return 0
    
    try:
        decision_engine = DecisionEngine(db)
        paper_sim = PaperSimulator()
        risk_mgr = RiskManager(db)
        
        # Circuit breaker check
        from app.ml.market_data import fetch_india_vix
        try:
            current_vix = fetch_india_vix()
            logger.info(f"      \ud83d\udcc8 VIX: {current_vix:.2f}")
        except Exception as e:
            logger.warning(f"      \u26a0\ufe0f  Could not fetch VIX: {e}, using default 20.0")
            current_vix = 20.0
        
        circuit_check = risk_mgr.check_circuit_breakers(current_vix=current_vix)
        if not circuit_check['safe']:
            logger.warning(f"      \ud83d\udea8 CIRCUIT BREAKERS: {circuit_check['reasons']}")
            return 0
        
        # Get latest features for this stock
        feature_data = _get_latest_features(db, instrument.id)
        if not feature_data:
            logger.warning(f"      \u26a0\ufe0f  No features found for {stock_symbol}")
            return 0
        
        # Check feature freshness
        latest_feature = db.query(Feature).filter(
            Feature.instrument_id == instrument.id
        ).order_by(Feature.ts_utc.desc()).first()
        
        if latest_feature:
            age_seconds = (datetime.utcnow() - latest_feature.ts_utc.replace(tzinfo=None)).total_seconds()
            logger.info(f"      \ud83d\udcc5 Feature age: {age_seconds:.0f}s")
            
            if age_seconds > settings.FEATURE_MAX_AGE_SECONDS:
                logger.warning(f"      \u26a0\ufe0f  Features too old ({age_seconds:.0f}s)")
                return 0
        
        # Get current price
        current_price = feature_data.get('close', 0)
        logger.info(f"      \ud83d\udcb0 Price: \u20b9{current_price:.2f}")
        
        # Load model from disk
        try:
            loaded_model = joblib.load(model_record.model_path)
        except Exception as e:
            logger.error(f"      \u274c Failed to load model: {e}")
            return 0
        
        # Make decision
        decision = decision_engine.decide_and_execute(
            feature_json=feature_data,
            model=loaded_model,
            user_id=str(user.id),
            instrument_id=str(instrument.id),
            mode='paper',
            current_price=current_price
        )
        
        # Phase 3: Log signal to signal_logs table
        signal_log = SignalLog(
            ts_utc=datetime.utcnow(),
            model_id=model_record.id,
            instrument_id=instrument.id,
            user_id=user.id,
            probability=decision.get('prob'),
            action=decision['action'],
            reason=decision.get('reason', 'N/A'),
            filters_passed=decision.get('filters_passed', {}),
            feature_snapshot=feature_data
        )
        db.add(signal_log)
        db.flush()
        
        # Log decision
        action = decision['action']
        prob = decision.get('prob')
        reason = decision.get('reason', 'N/A')
        
        if action == 'EXECUTE':
            logger.info(f"      \u2705 EXECUTE (Prob: {prob:.2%})")
            trade = _execute_trade(db, user, instrument, decision, model_record, paper_sim)
            if trade:
                signal_log.trade_id = trade.id
            db.commit()
            return 1
        else:
            logger.info(f"      \u274c REJECT - {reason} (Prob: {prob:.2% if prob else 'N/A'})")
            db.commit()
            return 1
    
    except Exception as e:
        logger.error(f"      \u274c Error: {str(e)}", exc_info=True)
        return 0


def _get_latest_features(db, instrument_id):
    """Get latest feature vector for an instrument"""
    feature = db.query(Feature).filter(
        Feature.instrument_id == instrument_id
    ).order_by(Feature.ts_utc.desc()).first()
    
    if not feature:
        return None
    
    return feature.feature_json


def _execute_trade(db, user, instrument, decision, model_record, paper_sim):
    """Execute a paper or live trade"""
    mode = 'paper'  # Always paper for now
    
    # Extract order parameters
    order_params = {
        'qty': decision.get('position_size', 0),
        'action': decision.get('signal'),
        'stop': decision.get('stop_price'),
        'target': decision.get('target_price'),
        'probability': decision.get('prob')
    }
    
    if order_params['qty'] == 0:
        logger.warning("Position size is 0, skipping trade")
        return None
    
    # Simulate paper execution
    fill_result = paper_sim.execute_paper_order(
        symbol=instrument.symbol,
        action=order_params['action'],
        qty=order_params['qty'],
        current_price=decision.get('current_price')
    )
    
    # Create trade record
    trade = Trade(
        user_id=user.id,
        instrument_id=instrument.id,
        model_id=model_record.id,
        mode=mode,
        action=order_params['action'],
        quantity=order_params['qty'],
        entry_price=fill_result['filled_price'],
        stop_price=order_params['stop'],
        target_price=order_params['target'],
        status='OPEN',
        entry_ts=datetime.utcnow()
    )
    db.add(trade)
    db.flush()
    
    # Add log
    log = TradeLog(
        trade_id=trade.id,
        log_text=f"\ud83d\udcc8 PAPER TRADE EXECUTED: {instrument.symbol} @ \u20b9{fill_result['filled_price']:.2f} | Qty: {order_params['qty']} | Stop: \u20b9{order_params['stop']:.2f} | Target: \u20b9{order_params['target']:.2f} | Probability: {order_params['probability']:.2%} | Slippage: \u20b9{fill_result['slippage']:.2f} | Commission: \u20b9{fill_result['commission']:.2f}",
        log_level='INFO'
    )
    db.add(log)
    
    logger.info(f"\\n  {'='*70}")
    logger.info(f"  \ud83c\udfaf PAPER TRADE EXECUTED FOR {user.email}")
    logger.info(f"  {'='*70}")
    logger.info(f"     Symbol: {instrument.symbol}")
    logger.info(f"     Entry Price: \u20b9{fill_result['filled_price']:.2f}")
    logger.info(f"     Quantity: {order_params['qty']}")
    logger.info(f"     Stop Loss: \u20b9{order_params['stop']:.2f}")
    logger.info(f"     Target: \u20b9{order_params['target']:.2f}")
    logger.info(f"     Probability: {order_params['probability']:.2%}")
    logger.info(f"     Model: {model_record.name}")
    logger.info(f"     Mode: {mode.upper()}")
    logger.info(f"  {'='*70}\\n")
    
    # Send email notification
    try:
        send_trade_execution_email(user, trade, instrument)
    except Exception as e:
        logger.warning(f"Failed to send email notification: {e}")
    
    return trade
