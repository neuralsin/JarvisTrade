"""
Sell Signal Handler - Extends DecisionEngine with sell logic
"""
from app.trading.decision_engine import DecisionEngine
from app.db.models import Trade, HistoricalCandle
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def execute_sell_signal(trade, probability, reason, db):
    """
    Execute sell/exit for a position
    
    Args:
        trade: Trade object to close
        probability: Sell signal strength (0-1)
        reason: Exit reason string
        db: Database session
    
    Returns:
        Dict with sell execution results
    """
    # Get current price
    latest_candle = db.query(HistoricalCandle).filter(
        HistoricalCandle.instrument_id == trade.instrument_id
    ).order_by(HistoricalCandle.ts_utc.desc()).first()
    
    if not latest_candle:
        logger.error(f"No price data for {trade.instrument_id}")
        return {"action": "ERROR", "reason": "NO_PRICE_DATA"}
    
    exit_price = float(latest_candle.close)
    entry_price = float(trade.entry_price)
    pnl = (exit_price - entry_price) * trade.qty
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    
    # Update trade record
    trade.exit_price = exit_price
    trade.exit_ts = datetime.utcnow().replace(tzinfo=None)
    trade.status = 'closed'
    trade.pnl = pnl
    trade.exit_reason = reason
    
    try:
        db.commit()
        logger.info(f"✅ SELL: {trade.instrument.symbol} @ ₹{exit_price:.2f}, "
                   f"P&L: ₹{pnl:.2f} ({pnl_pct:+.2f}%), {reason}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit sell: {e}")
        return {"action": "ERROR", "reason": str(e)}
    
    return {
        "action": "SELL",
        "trade_id": str(trade.id),
        "symbol": trade.instrument.symbol,
        "exit_price": exit_price,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "probability": probability,
        "reason": reason
    }
