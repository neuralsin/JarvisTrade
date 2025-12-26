"""
Phase 3: Per-Model Signal Checking Function

Replaces _check_signals_for_user with _check_signals_for_model
to enable parallel processing of multiple models (one per stock).
"""

# This code should be inserted into execution.py after check_and_execute_signals

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
    from app.ml.model_registry import ModelRegistry
    from datetime import datetime
    import joblib
    
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
            logger.warning(f"      \ud83d\udea8 CIRCUIT BREAKERS TRIGGERED: {circuit_check['reasons']}")
            return 0
        
        # Get latest features for this stock
        feature_data = _get_latest_features(db, instrument.id)
        if not feature_data:
            logger.warning(f"      \u26a0\ufe0f  No features found for {stock_symbol}")
            return 0
        
        # Phase 3: Check feature freshness
        latest_feature = db.query(Feature).filter(
            Feature.instrument_id == instrument.id
        ).order_by(Feature.ts_utc.desc()).first()
        
        if latest_feature:
            age_seconds = (datetime.utcnow() - latest_feature.ts_utc).total_seconds()
            logger.info(f"      \ud83d\udcc5 Feature age: {age_seconds:.0f}s")
            
            if age_seconds > settings.FEATURE_MAX_AGE_SECONDS:
                logger.warning(f"      \u26a0\ufe0f  Features too old ({age_seconds:.0f}s > {settings.FEATURE_MAX_AGE_SECONDS}s)")
                return 0
        
        # Get current price
        current_price = feature_data.get('close', 0)
        logger.info(f"      \ud83d\udcb0 Price: \u20b9{current_price:.2f}")
        
        # Load model from disk
        try:
            loaded_model = joblib.load(model_record.model_path)
        except Exception as e:
            logger.error(f"      \u274c Failed to load model from {model_record.model_path}: {e}")
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
        db.flush()  # Get signal_log.id
        
        # Log decision with details
        action = decision['action']
        prob = decision.get('prob')
        reason = decision.get('reason', 'N/A')
        
        if action == 'EXECUTE':
            logger.info(f"      \u2705 SIGNAL: EXECUTE (Probability: {prob:.2%})")
            
            # Execute trade
            trade = _execute_trade(db, user, instrument, decision, model_record, paper_sim)
            
            # Link signal to trade
            if trade:
                signal_log.trade_id = trade.id
            
            db.commit()
            return 1
        else:
            logger.info(f"      \u274c SIGNAL: REJECTED - {reason} (Probability: {prob:.2% if prob else 'N/A'})")
            db.commit()
            return 1
    
    except Exception as e:
        logger.error(f"      \u274c Error processing {stock_symbol}: {str(e)}", exc_info=True)
        return 0
