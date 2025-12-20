"""
Spec 6 & 12: Execution tasks - check signals and execute trades
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import User, Instrument, Trade, TradeLog
from app.ml.model_registry import ModelRegistry
from app.ml.feature_engineer import compute_features, extract_feature_vector
from app.trading.decision_engine import DecisionEngine
from app.trading.paper_simulator import PaperSimulator
from app.trading.kite_client import get_kite_client_for_user
from app.trading.risk_manager import RiskManager
from app.utils.mailer import send_trade_execution_email
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def check_and_execute_signals(self):
    """
    Spec 6: Periodically check for trading signals and execute
    Runs every 15 minutes during market hours
    """
    db = SessionLocal()
    
    try:
        # Get active model
        model, model_record = ModelRegistry.get_active_model(db)
        if not model:
            logger.info("No active model, skipping signal check")
            return {"status": "no_model"}
        
        # Get users with auto-execute enabled
        users = db.query(User).filter(User.auto_execute == True).all()
        
        logger.info(f"Checking signals for {len(users)} users")
        
        for user in users:
            try:
                _check_signals_for_user(db, user, model, model_record)
            except Exception as e:
                logger.error(f"Error checking signals for user {user.email}: {str(e)}")
        
        return {"status": "success"}
    
    finally:
        db.close()


def _check_signals_for_user(db, user, model, model_record):
    """
    Check and execute signals for a single user
    """
    # Get tracked instruments (for now, use a subset)
    instruments = db.query(Instrument).limit(10).all()
    
    decision_engine = DecisionEngine(db)
    paper_sim = PaperSimulator()
    risk_mgr = RiskManager(db)
    
    # Circuit breaker check
    circuit_check = risk_mgr.check_circuit_breakers(current_vix=20.0)  # TODO: Get real VIX
    if not circuit_check['safe']:
        logger.warning(f"Circuit breakers triggered: {circuit_check['reasons']}")
        return
    
    for instrument in instruments:
        try:
            # Get latest features
            feature_data = _get_latest_features(db, instrument.id)
            if not feature_data:
                continue
            
            # Get current price (from latest candle or live quote)
            current_price = feature_data.get('close', 0)
            
            # Make decision
            decision = decision_engine.decide_and_execute(
                feature_json=feature_data,
                model=model,
                user_id=str(user.id),
                instrument_id=str(instrument.id),
                mode='paper',  # Default to paper, check user settings for live
                current_price=current_price
            )
            
            if decision['action'] == 'EXECUTE':
                _execute_trade(db, user, instrument, decision, model_record, paper_sim)
        
        except Exception as e:
            logger.error(f"Error processing {instrument.symbol}: {str(e)}")


def _get_latest_features(db, instrument_id):
    """
    Get latest computed features for an instrument
    """
    from app.db.models import Feature
    
    latest = db.query(Feature).filter(
        Feature.instrument_id == instrument_id
    ).order_by(Feature.ts_utc.desc()).first()
    
    return latest.feature_json if latest else None


def _execute_trade(db, user, instrument, decision, model_record, paper_sim):
    """
    Execute the trade (paper or live)
    """
    order_params = decision['order_params']
    mode = order_params['mode']
    
    # Simulate entry
    fill_result = paper_sim.simulate_entry(order_params)
    
    # Create trade record
    trade = Trade(
        user_id=user.id,
        instrument_id=instrument.id,
        mode=mode,
        entry_ts=datetime.utcnow(),
        entry_price=fill_result['filled_price'],
        qty=order_params['qty'],
        stop_price=order_params['stop'],
        target_price=order_params['target'],
        probability=order_params['probability'],
        model_id=model_record.id,
        status='open',
        reason=order_params['reason']
    )
    db.add(trade)
    db.flush()
    
    # Add log
    log = TradeLog(
        trade_id=trade.id,
        log_text=f"Trade entered: {fill_result['filled_price']} (slippage: {fill_result['slippage']:.2f}, commission: {fill_result['commission']:.2f})",
        log_level='INFO'
    )
    db.add(log)
    db.commit()
    
    logger.info(f"Trade executed for {user.email}: {instrument.symbol} @ {fill_result['filled_price']}")
    
    # Send email notification
    try:
        email_data = {
            'trade_id': str(trade.id),
            'email': user.email,
            'mode': mode,
            'symbol': instrument.symbol,
            'entry_price': fill_result['filled_price'],
            'entry_ts': trade.entry_ts.isoformat(),
            'qty': order_params['qty'],
            'stop': order_params['stop'],
            'target': order_params['target'],
            'prob': order_params['probability'],
            'model_name': model_record.name,
            'reason': order_params['reason']
        }
        send_trade_execution_email(email_data)
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
