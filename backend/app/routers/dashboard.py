"""
Spec 9: Dashboard API routes
Returns equity curve, P&L, probability heatmap, open trades
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.database import get_db
from app.db.models import User, Trade, EquitySnapshot, Instrument
from app.routers.auth import get_current_user
from typing import List, Optional
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/")
async def get_dashboard(
    mode: str = Query("paper", regex="^(paper|live)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 9: Dashboard API returning equity curve, drawdown, probability heatmap, open trades
    """
    # Equity curve
    equity_curve = db.query(EquitySnapshot).filter(
        EquitySnapshot.user_id == current_user.id,
        EquitySnapshot.mode == mode
    ).order_by(EquitySnapshot.ts_utc).all()
    
    equity_data = [
        {"ts": snap.ts_utc.isoformat() + "Z", "value": float(snap.equity_value)}
        for snap in equity_curve
    ]
    
    # Calculate drawdown
    if equity_data:
        peak = equity_data[0]["value"]
        drawdown_data = []
        for point in equity_data:
            if point["value"] > peak:
                peak = point["value"]
            dd = (point["value"] - peak) / peak if peak > 0 else 0
            drawdown_data.append({"ts": point["ts"], "drawdown": dd})
    else:
        drawdown_data = []
    
    # Open trades
    open_trades = db.query(Trade).join(Instrument).filter(
        Trade.user_id == current_user.id,
        Trade.mode == mode,
        Trade.status == 'open'
    ).all()
    
    open_trades_data = [
        {
            "id": str(trade.id),
            "symbol": trade.instrument.symbol,
            "entry_price": float(trade.entry_price) if trade.entry_price else None,
            "entry_ts": trade.entry_ts.isoformat() + "Z" if trade.entry_ts else None,
            "qty": trade.qty,
            "stop": float(trade.stop_price) if trade.stop_price else None,
            "target": float(trade.target_price) if trade.target_price else None,
            "probability": float(trade.probability) if trade.probability else None,
            "current_pnl": None  # TODO: Calculate based on current price
        }
        for trade in open_trades
    ]
    
    # Probability heatmap (last 50 trades)
    recent_trades = db.query(Trade).join(Instrument).filter(
        Trade.user_id == current_user.id,
        Trade.mode == mode
    ).order_by(Trade.created_at.desc()).limit(50).all()
    
    probability_heatmap = [
        {
            "symbol": trade.instrument.symbol,
            "ts": trade.entry_ts.isoformat() + "Z" if trade.entry_ts else trade.created_at.isoformat() + "Z",
            "prob": float(trade.probability) if trade.probability else 0
        }
        for trade in recent_trades
    ]
    
    # Total P&L
    total_pnl = db.query(func.sum(Trade.pnl)).filter(
        Trade.user_id == current_user.id,
        Trade.mode == mode,
        Trade.pnl.isnot(None)
    ).scalar() or 0
    
    return {
        "equity_curve": equity_data,
        "drawdown": drawdown_data,
        "probability_heatmap": probability_heatmap,
        "open_trades": open_trades_data,
        "total_pnl": float(total_pnl),
        "mode": mode
    }


@router.get("/stats")
async def get_stats(
    mode: str = Query("paper", regex="^(paper|live)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get trading statistics
    """
    trades = db.query(Trade).filter(
        Trade.user_id == current_user.id,
        Trade.mode == mode,
        Trade.status == 'closed'
    ).all()
    
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0
        }
    
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    
    return {
        "total_trades": len(trades),
        "win_rate": len(winning_trades) / len(trades) if trades else 0,
        "avg_pnl": sum(t.pnl or 0 for t in trades) / len(trades),
        "total_pnl": sum(t.pnl or 0 for t in trades)
    }
