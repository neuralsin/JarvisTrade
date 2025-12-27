"""
Position Monitoring Task with Peak Detection
Runs every 30 seconds to check stop/target/peak exits
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Trade, HistoricalCandle
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@celery_app.task
def monitor_open_positions():
    """
    Monitor ALL open positions for exit conditions:
    1. Stop loss hits
    2. Target hits
    3. Peak detection exits
    """
    from app.config import settings
    from app.ml.peak_detector import should_sell_at_peak
    from app.ml.feature_engineer import compute_features
    
    db = SessionLocal()
    
    try:
        open_trades = db.query(Trade).filter(Trade.status == 'open').all()
        
        if not open_trades:
            return {"status": "success", "message": "No open positions", "closed": 0}
        
        logger.info(f"🔍 Monitoring {len(open_trades)} open positions...")
        
        closed_count = 0
        
        for trade in open_trades:
            try:
                # Get latest candle
                latest_candle = db.query(HistoricalCandle).filter(
                    HistoricalCandle.instrument_id == trade.instrument_id
                ).order_by(HistoricalCandle.ts_utc.desc()).first()
                
                if not latest_candle:
                    continue
                
                current_price = float(latest_candle.close)
                entry_price = float(trade.entry_price)
                
                # 1. STOP LOSS CHECK
                if trade.stop_price and current_price <= float(trade.stop_price):
                    _close_position(trade, current_price, "STOP_LOSS_HIT", db)
                    closed_count += 1
                    continue
                
                # 2. TARGET HIT CHECK
                if trade.target_price and current_price >= float(trade.target_price):
                    _close_position(trade, current_price, "TARGET_HIT", db)
                    closed_count += 1
                    continue
                
                # 3. PEAK EXIT CHECK (if enabled)
                if settings.PEAK_EXIT_ENABLED:
                    recent_candles = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == trade.instrument_id
                    ).order_by(HistoricalCandle.ts_utc.desc()).limit(30).all()
                    
                    if len(recent_candles) >= 20:
                        # Convert to DataFrame
                        df = pd.DataFrame([{
                            'ts_utc': c.ts_utc,
                            'open': float(c.open),
                            'high': float(c.high),
                            'low': float(c.low),
                            'close': float(c.close),
                            'volume': int(c.volume)
                        } for c in reversed(recent_candles)])
                        
                        #  Compute features (RSI needed)
                        df = compute_features(df, trade.instrument_id, db)
                        
                        # Check peak
                        peak_decision = should_sell_at_peak(
                            df,
                            entry_price=entry_price,
                            min_profit_pct=settings.PEAK_MIN_PROFIT_PCT,
                            min_peak_score=3
                        )
                        
                        if peak_decision['should_sell']:
                            _close_position(trade, current_price, "PEAK_DETECTED", db)
                            closed_count += 1
                            logger.info(f"🏔️ Peak exit: {trade.instrument.symbol}, "
                                      f"Profit: {peak_decision['profit_pct']*100:.2f}%, "
                                      f"Score: {peak_decision['peak_score']}/5")
            
            except Exception as e:
                logger.error(f"Error monitoring trade {trade.id}: {e}")
                continue
        
        db.commit()
        return {
            "status": "success",
            "positions_checked": len(open_trades),
            "positions_closed": closed_count
        }
    
    except Exception as e:
        logger.error(f"Position monitoring failed: {e}")
        db.rollback()
        return {"status": "error", "message": str(e)}
    
    finally:
        db.close()


def _close_position(trade, exit_price, reason, db):
    """Helper to close a position"""
    pnl = (exit_price - float(trade.entry_price)) * trade.qty
    
    trade.exit_price = exit_price
    trade.exit_ts = datetime.utcnow().replace(tzinfo=None)
    trade.status = 'closed'
    trade.pnl = pnl
    trade.exit_reason = reason
    
    logger.info(f"✅ Closed: {trade.instrument.symbol} @ ₹{exit_price:.2f}, "
                f"P&L: ₹{pnl:.2f}, Reason: {reason}")
