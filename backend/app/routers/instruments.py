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
    data_period: str = "60d"


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
async def validate_symbol_endpoint(
    request: ValidateSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Validate stock symbol using Kite Connect API.
    Fast, reliable, no rate limits.
    """
    from app.utils.kite_validator import validate_symbol
    
    # Map NS/BO to NSE/BSE for Kite
    exchange = request.exchange
    if exchange == "NS":
        exchange = "NSE"
    elif exchange == "BO":
        exchange = "BSE"
    
    result = validate_symbol(request.symbol, exchange)
    return result


@router.post("/add")
async def add_instrument(
    request: AddInstrumentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add new instrument to the system"""
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
    validation_result = await validate_symbol_endpoint(
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
        instrument_type='EQ'
    )
    
    db.add(instrument)
    db.commit()
    db.refresh(instrument)
    
    logger.info(f"Added instrument: {instrument.symbol} ({instrument.name})")
    
    # Trigger background data fetch
    try:
        from app.tasks.data_ingestion import fetch_historical_data
        
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        period_map = {
            '60d': (59, '15m'),
            '1y': (365, '1d'),
            '2y': (730, '1d'),
            '5y': (1825, '1d')
        }
        
        days, interval = period_map.get(request.data_period, (59, '15m'))
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        task = fetch_historical_data.delay(
            symbols=[instrument.symbol],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=request.exchange
        )
        
        return {
            "success": True,
            "id": str(instrument.id),
            "symbol": instrument.symbol,
            "name": instrument.name,
            "message": f"Added {instrument.symbol}. Fetching {request.data_period} of data...",
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
    """Remove instrument from system"""
    instrument = db.query(Instrument).filter(Instrument.id == instrument_id).first()
    
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")
    
    symbol = instrument.symbol
    name = instrument.name
    
    db.delete(instrument)
    db.commit()
    
    logger.info(f"Removed instrument: {symbol} ({name})")
    
    return {
        "success": True,
        "message": f"Removed {symbol} and all associated data"
    }
