"""
Instrument management API - add/remove stocks, validate symbols
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, Instrument
from app.routers.auth import get_current_user
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
            "instrument_type": inst.instrument_type
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
    Uses Yahoo Finance to check symbol validity
    """
    ticker_symbol = f"{request.symbol}.{request.exchange}"
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Check if ticker returned valid data
        if not info or 'symbol' not in info or info.get('symbol') == '':
            return {
                "valid": False,
                "error": "Symbol not found on exchange"
            }
        
        # Additional check - try to get recent data
        hist = ticker.history(period="5d")
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
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}"
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
    
    # Trigger background data fetch for last 60 days (Yahoo limit for 15m)
    try:
        from app.tasks.data_ingestion import fetch_historical_data
        
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=59)).strftime('%Y-%m-%d')
        
        task = fetch_historical_data.delay(
            symbols=[instrument.symbol],
            start_date=start_date,
            end_date=end_date,
            interval='15m',
            exchange=request.exchange
        )
        
        logger.info(f"Triggered data fetch for {instrument.symbol}: task {task.id}")
        
        return {
            "success": True,
            "id": str(instrument.id),
            "symbol": instrument.symbol,
            "name": instrument.name,
            "message": f"Added {instrument.symbol}. Fetching historical data in background...",
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
