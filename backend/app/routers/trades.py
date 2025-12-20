"""
Trades API routes
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User, Trade, Instrument, TradeLog, Model
from app.routers.auth import get_current_user
from typing import List, Optional

router = APIRouter()


@router.get("/")
async def get_trades(
    mode: str = Query("paper", regex="^(paper|live)$"),
    status: Optional[str] = Query(None, regex="^(open|closed)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get trade history with filters
    """
    query = db.query(Trade).join(Instrument).filter(
        Trade.user_id == current_user.id,
        Trade.mode == mode
    )
    
    if status:
        query = query.filter(Trade.status == status)
    
    trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
    
    return {
        "trades": [
            {
                "id": str(trade.id),
                "symbol": trade.instrument.symbol,
                "entry_price": float(trade.entry_price) if trade.entry_price else None,
                "exit_price": float(trade.exit_price) if trade.exit_price else None,
                "entry_ts": trade.entry_ts.isoformat() + "Z" if trade.entry_ts else None,
                "exit_ts": trade.exit_ts.isoformat() + "Z" if trade.exit_ts else None,
                "qty": trade.qty,
                "pnl": float(trade.pnl) if trade.pnl else None,
                "status": trade.status,
                "probability": float(trade.probability) if trade.probability else None,
                "reason": trade.reason
            }
            for trade in trades
        ]
    }


@router.get("/{trade_id}")
async def get_trade_details(
    trade_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 9: Trade detail modal with features used, model version, probability, etc.
    """
    trade = db.query(Trade).join(Instrument).filter(
        Trade.id == trade_id,
        Trade.user_id == current_user.id
    ).first()
    
    if not trade:
        return {"error": "Trade not found"}
    
    # Get logs
    logs = db.query(TradeLog).filter(TradeLog.trade_id == trade_id).order_by(TradeLog.ts).all()
    
    # Get model info
    model_info = None
    if trade.model_id:
        model = db.query(Model).filter(Model.id == trade.model_id).first()
        if model:
            model_info = {
                "name": model.name,
                "trained_at": model.trained_at.isoformat() + "Z" if model.trained_at else None,
                "metrics": model.metrics_json
            }
    
    return {
        "trade_id": str(trade.id),
        "symbol": trade.instrument.symbol,
        "mode": trade.mode,
        "status": trade.status,
        "entry_price": float(trade.entry_price) if trade.entry_price else None,
        "exit_price": float(trade.exit_price) if trade.exit_price else None,
        "entry_ts": trade.entry_ts.isoformat() + "Z" if trade.entry_ts else None,
        "exit_ts": trade.exit_ts.isoformat() + "Z" if trade.exit_ts else None,
        "qty": trade.qty,
        "stop_price": float(trade.stop_price) if trade.stop_price else None,
        "target_price": float(trade.target_price) if trade.target_price else None,
        "pnl": float(trade.pnl) if trade.pnl else None,
        "probability": float(trade.probability) if trade.probability else None,
        "reason": trade.reason,
        "model": model_info,
        "logs": [
            {
                "ts": log.ts.isoformat() + "Z",
                "level": log.log_level,
                "text": log.log_text
            }
            for log in logs
        ]
    }
