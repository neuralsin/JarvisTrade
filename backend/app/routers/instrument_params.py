"""
API endpoint for updating instrument trading parameters
"""
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Instrument, User
from app.routers.auth import get_current_user
from uuid import UUID

router = APIRouter()


class UpdateInstrumentParams(BaseModel):
    buy_confidence_threshold: Optional[float] = None
    sell_confidence_threshold: Optional[float] = None
    stop_multiplier: Optional[float] = None
    target_multiplier: Optional[float] = None
    max_position_size: Optional[int] = None
    is_trading_enabled: Optional[bool] = None


@router.put("/{instrument_id}/parameters")
async def update_instrument_parameters(
    instrument_id: str,
    params: UpdateInstrumentParams,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update trading parameters for a specific instrument
    """
    try:
        instrument_uuid = UUID(instrument_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid instrument ID")
    
    instrument = db.query(Instrument).filter(Instrument.id == instrument_uuid).first()
    
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")
    
    # Validate and update
    if params.buy_confidence_threshold is not None:
        if not (0 <= params.buy_confidence_threshold <= 1):
            raise HTTPException(status_code=400, detail="Buy confidence must be 0-1")
        instrument.buy_confidence_threshold = params.buy_confidence_threshold
    
    if params.sell_confidence_threshold is not None:
        if not (0 <= params.sell_confidence_threshold <= 1):
            raise HTTPException(status_code=400, detail="Sell confidence must be 0-1")
        instrument.sell_confidence_threshold = params.sell_confidence_threshold
    
    if params.stop_multiplier is not None:
        if params.stop_multiplier <= 0:
            raise HTTPException(status_code=400, detail="Stop multiplier must be positive")
        instrument.stop_multiplier = params.stop_multiplier
    
    if params.target_multiplier is not None:
        if params.target_multiplier <= 0:
            raise HTTPException(status_code=400, detail="Target multiplier must be positive")
        instrument.target_multiplier = params.target_multiplier
    
    if params.max_position_size is not None:
        if params.max_position_size <= 0:
            raise HTTPException(status_code=400, detail="Max position must be positive")
        instrument.max_position_size = params.max_position_size
    
    if params.is_trading_enabled is not None:
        instrument.is_trading_enabled = params.is_trading_enabled
    
    try:
        db.commit()
        db.refresh(instrument)
        
        return {
            "status": "success",
            "parameters": {
                "buy_confidence_threshold": instrument.buy_confidence_threshold,
                "sell_confidence_threshold": instrument.sell_confidence_threshold,
                "stop_multiplier": instrument.stop_multiplier,
                "target_multiplier": instrument.target_multiplier,
                "max_position_size": instrument.max_position_size,
                "is_trading_enabled": instrument.is_trading_enabled
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
