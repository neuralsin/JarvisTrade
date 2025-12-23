"""
Instrument management API - add/remove stocks, validate symbols
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, Instrument
from app.routers.auth import get_current_user
from app.utils.yfinance_wrapper import get_rate_limiter
from typing import Optional
import logging
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()


class ValidateSymbolRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"


class AddInstrumentRequest(BaseModel):
    symbol: str
    name: Optional[str] = None
    exchange: str = "NSE"
    data_period: str = "60d"  # Default 60 days


@router.get("/")
async def get_instruments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all instruments in the system"""
    instruments = db.query(Instrument).order_by(Instrument.symbol).all()
    
    return [
        {
            "id": str(inst.id),
            "symbol": inst.symbol,
            "name": inst.name,
            "exchange": inst.exchange,
            "instrument_type": inst.instrument_type,
            "buy_confidence_threshold": inst.buy_confidence_threshold or 0.3,
            "sell_confidence_threshold": inst.sell_confidence_threshold or 0.5,
            "stop_multiplier": inst.stop_multiplier or 1.5,
            "target_multiplier": inst.target_multiplier or 2.5,
            "max_position_size": inst.max_position_size or 100,
            "is_trading_enabled": inst.is_trading_enabled if inst.is_trading_enabled is not None else True
        }
        for inst in instruments
    ]


@router.post("/validate")
async def validate_symbol(
    request: ValidateSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Validate if stock symbol exists on exchange
    Uses Yahoo Finance with rate limiting to check symbol validity
    """
    # Map exchange codes to Yahoo Finance suffixes
    exchange_suffix_map = {
        "NSE": "NS",  # National Stock Exchange
        "NS": "NS",   # Alias
        "BSE": "BO",  # Bombay Stock Exchange
        "BO": "BO"    # Alias
    }
    
    suffix = exchange_suffix_map.get(request.exchange, request.exchange)
    ticker_symbol = f"{request.symbol}.{suffix}"
    
    try:
        # Use rate-limited wrapper to prevent 429 errors
        rate_limiter = get_rate_limiter()
        
        # Get ticker info
        info = rate_limiter.get_ticker_info(ticker_symbol)
        
        # Check if ticker returned valid data
        if not info or 'symbol' not in info or info.get('symbol') == '':
            return {
                "valid": False,
                "error": "Symbol not found on exchange"
            }
        
        # Additional check - try to get recent data
        hist = rate_limiter.get_ticker_history(ticker_symbol, period="5d")
        if hist.empty:
            return {
                "valid": False,
                "error": "No trading data available for this symbol"
            }
        
        return {
            "valid": True,
            "symbol": request.symbol,
            "name": info.get('longName') or info.get('shortName') or request.symbol,
            "exchange": request.exchange,
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap'),
            "currency": info.get('currency', 'INR')
        }
    
    except Exception as e:
        logger.error(f"Symbol validation failed for {ticker_symbol}: {str(e)}")
        
        # Provide user-friendly error message for rate limiting
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            error_msg = "Rate limit exceeded. Please try again in a few moments."
        
        return {
            "valid": False,
            "error": f"Validation failed: {error_msg}"
        }


@router.post("/add")
async def add_instrument(
    request: AddInstrumentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Add new instrument to the system
    - Validates symbol first
    - Creates instrument record
    - Triggers background data fetch
    """
    # Check if already exists
    existing = db.query(Instrument).filter(
        Instrument.symbol == request.symbol.upper(),
        Instrument.exchange == request.exchange
    ).first()
    
    if existing:
        return {
            "error": f"Instrument {request.symbol} already exists",
            "id": str(existing.id)
        }
    
    # Validate symbol first
    validation_result = await validate_symbol(
        ValidateSymbolRequest(
            symbol=request.symbol.upper(),
            exchange=request.exchange
        ),
        current_user=current_user
    )
    
    if not validation_result.get('valid'):
        raise HTTPException(
            status_code=400,
            detail=validation_result.get('error', 'Invalid symbol')
        )
    
    # Create instrument
    instrument = Instrument(
        symbol=request.symbol.upper(),
        name=request.name or validation_result['name'],
        exchange=request.exchange,
        instrument_type='EQ'  # Equity
    )
    
    db.add(instrument)
    db.commit()
    db.refresh(instrument)
    
    logger.info(f"Added instrument: {instrument.symbol} ({instrument.name})")
    
    # Trigger background data fetch based on period
    try:
        from app.tasks.data_ingestion import fetch_historical_data
        
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Calculate start date and interval based on period
        period_map = {
            '60d': (59, '15m'),   # 60 days, 15min candles
            '1y': (365, '1d'),    # 1 year, daily candles
            '2y': (730, '1d'),    # 2 years, daily candles
            '5y': (1825, '1d')    # 5 years, daily candles
        }
        
        days, interval = period_map.get(request.data_period, (59, '15m'))
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {request.data_period} of data for {instrument.symbol}: {start_date} to {end_date}, interval={interval}")
        
        task = fetch_historical_data.delay(
            symbols=[instrument.symbol],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=request.exchange
        )
        
        logger.info(f"Triggered data fetch for {instrument.symbol}: task {task.id}")
        
        return {
            "success": True,
            "id": str(instrument.id),
            "symbol": instrument.symbol,
            "name": instrument.name,
            "message": f"Added {instrument.symbol}. Fetching {request.data_period} of historical data...",
            "data_fetch_task_id": task.id
        }
    
    except Exception as e:
        logger.warning(f"Failed to trigger data fetch: {str(e)}")
        return {
            "success": True,
            "id": str(instrument.id),
            "symbol": instrument.symbol,
            "name": instrument.name,
            "message": f"Added {instrument.symbol}. Data fetch failed - run manually.",
            "warning": str(e)
        }


@router.delete("/{instrument_id}")
async def remove_instrument(
    instrument_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Remove instrument from system
    Cascades to delete related features, candles, etc.
    """
    instrument = db.query(Instrument).filter(Instrument.id == instrument_id).first()
    
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")
    
    symbol = instrument.symbol
    name = instrument.name
    
    # Delete (cascades to features, candles, etc. due to ON DELETE CASCADE)
    db.delete(instrument)
    db.commit()
    
    logger.info(f"Removed instrument: {symbol} ({name})")
    
    return {
        "success": True,
        "message": f"Removed {symbol} and all associated data"
    }
