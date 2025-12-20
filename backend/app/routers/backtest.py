"""
Backtesting API router
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.db.database import get_db
from app.db.models import User, Feature, Instrument
from app.routers.auth import get_current_user
from app.ml.backtesting import BacktestEngine, walk_forward_optimization, monte_carlo_simulation
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class BacktestRequest(BaseModel):
    name: str
    symbol: Optional[str] = None
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    strategy_type: str = 'simple_ma_crossover'  # simple, ml_based
    parameters: Optional[dict] = {}


@router.post("/run")
async def run_backtest(
    request: BacktestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run a backtest with specified parameters
    """
    try:
        # Load historical data
        query = db.query(Feature).join(Instrument)
        
        if request.symbol:
            query = query.filter(Instrument.symbol == request.symbol)
        
        query = query.filter(
            Feature.ts_utc >= request.start_date,
            Feature.ts_utc <= request.end_date
        )
        
        features_data = query.all()
        
        if not features_data:
            raise HTTPException(status_code=404, detail="No data found for backtest period")
        
        # Convert to DataFrame
        df_rows = []
        for feat in features_data:
            row = feat.feature_json.copy() if feat.feature_json else {}
            row['ts_utc'] = feat.ts_utc
            row['symbol'] = request.symbol or 'UNKNOWN'
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        df = df.sort_values('ts_utc')
        
       # Define strategy function
        def simple_strategy(row):
            """Simple MA crossover strategy"""
            # Assuming we have these in features
            if 'ema_9' in row and 'ema_21' in row:
                if row['ema_9'] > row['ema_21'] and row.get('prev_ema_9', 0) <= row.get('prev_ema_21', 0):
                    # Golden cross - BUY
                    stop = row['close'] * 0.97  # 3% stop loss
                    target = row['close'] * 1.05  # 5% target
                    return ('BUY', stop, target)
                elif row['ema_9'] < row['ema_21'] and row.get('prev_ema_9', 0) >= row.get('prev_ema_21', 0):
                    # Death cross - SELL
                    return ('SELL', None, None)
            
            return ('HOLD', None, None)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=request.initial_capital)
        metrics = engine.run_backtest(df, simple_strategy)
        
        # Extract trades
        trades = engine.trades
        equity_curve = engine.equity_curve
        
        logger.info(f"Backtest completed: {request.name}, Return: {metrics['total_return_pct']:.2f}%")
        
        return {
            "status": "success",
            "name": request.name,
            "metrics": metrics,
            "trades": trades[:50],  # Limit to first 50 trades
            "equity_curve": equity_curve[-100:],  # Last 100 points
            "total_trades": len(trades)
        }
    
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk-forward")
async def run_walk_forward(
    request: BacktestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run walk-forward optimization
    """
    try:
        # Load data (same as above)
        query = db.query(Feature).join(Instrument)
        
        if request.symbol:
            query = query.filter(Instrument.symbol == request.symbol)
        
        query = query.filter(
            Feature.ts_utc >= request.start_date,
            Feature.ts_utc <= request.end_date
        )
        
        features_data = query.all()
        
        if not features_data:
            raise HTTPException(status_code=404, detail="No data found")
        
        df_rows = []
        for feat in features_data:
            row = feat.feature_json.copy() if feat.feature_json else {}
            row['ts_utc'] = feat.ts_utc
            row['symbol'] = request.symbol or 'UNKNOWN'
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        df = df.sort_values('ts_utc')
        df = df.set_index('ts_utc')
        
        # Define strategy
        def simple_strategy(row):
            if 'ema_9' in row and 'ema_21' in row:
                if row['ema_9'] > row['ema_21']:
                    return ('BUY', row['close'] * 0.97, row['close'] * 1.05)
                elif row['ema_9'] < row['ema_21']:
                    return ('SELL', None, None)
            return ('HOLD', None, None)
        
        # Run walk-forward
        results = walk_forward_optimization(
            df,
            simple_strategy,
            train_window=252,  # 1 year
            test_window=63,    # 3 months
            step_size=21       # 1 month
        )
        
        return {
            "status": "success",
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Walk-forward failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monte-carlo")
async def run_monte_carlo(
    num_simulations: int = 10000,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run Monte Carlo simulation on user's historical trades
    """
    try:
        from app.db.models import Trade
        
        # Get user's past trades
        trades = db.query(Trade).filter(
            Trade.user_id == current_user.id,
            Trade.status == 'closed'
        ).all()
        
        if not trades:
            raise HTTPException(status_code=404, detail="No closed trades found")
        
        # Convert to list of dicts
        trade_data = [
            {
                'pnl': t.pnl or 0,
                'pnl_pct': ((t.exit_price / t.entry_price - 1) * 100) if t.exit_price and t.entry_price else 0
            }
            for t in trades
        ]
        
        # Run Monte Carlo
        results = monte_carlo_simulation(trade_data, num_simulations)
        
        return {
            "status": "success",
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Monte Carlo failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
