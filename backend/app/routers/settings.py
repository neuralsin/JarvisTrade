"""
Settings/Parameters management API
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, SystemState
from app.routers.auth import get_current_user
from app.config import settings
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TradingParameters(BaseModel):
    account_capital: float
    risk_per_trade: float
    max_daily_loss: float
    stop_multiplier: float
    target_multiplier: float
    prob_min: float
    prob_strong: float
    max_trades_per_day: int


class UpdateParametersRequest(BaseModel):
    parameters: TradingParameters
    password: str
    mode: str  # 'paper' or 'live'


@router.get("/parameters")
async def get_trading_parameters(
    mode: str = "paper",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current trading parameters
    Parameters can be different for paper vs live trading
    """
    # Try to get user-specific parameters from database
    param_key = f"trading_params_{mode}_{current_user.id}"
    param_record = db.query(SystemState).filter(SystemState.key == param_key).first()
    
    if param_record and param_record.value:
        import json
        return {"mode": mode, "parameters": json.loads(param_record.value)}
    
    # Return default parameters from config
    return {
        "mode": mode,
        "parameters": {
            "account_capital": settings.ACCOUNT_CAPITAL,
            "risk_per_trade": settings.RISK_PER_TRADE,
            "max_daily_loss": settings.MAX_DAILY_LOSS,
            "stop_multiplier": settings.STOP_MULTIPLIER,
            "target_multiplier": settings.TARGET_MULTIPLIER,
            "prob_min": settings.PROB_MIN,
            "prob_strong": settings.PROB_STRONG,
            "max_trades_per_day": settings.MAX_TRADES_PER_DAY
        }
    }


@router.post("/parameters")
async def update_trading_parameters(
    request: UpdateParametersRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update trading parameters (requires password confirmation)
    """
    # Verify password
    if not pwd_context.verify(request.password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    # Validate parameters
    params = request.parameters
    
    if params.risk_per_trade <= 0 or params.risk_per_trade > 0.1:
        raise HTTPException(status_code=400, detail="Risk per trade must be between 0 and 10%")
    
    if params.max_daily_loss <= 0 or params.max_daily_loss > 0.2:
        raise HTTPException(status_code=400, detail="Max daily loss must be between 0 and 20%")
    
    if params.account_capital <= 0:
        raise HTTPException(status_code=400, detail="Account capital must be positive")
    
    if params.prob_min < 0.5 or params.prob_min > 0.95:
        raise HTTPException(status_code=400, detail="Minimum probability must be between 50% and 95%")
    
    # Store parameters
    import json
    param_key = f"trading_params_{request.mode}_{current_user.id}"
    param_record = db.query(SystemState).filter(SystemState.key == param_key).first()
    
    if param_record:
        param_record.value = json.dumps(params.dict())
    else:
        param_record = SystemState(key=param_key, value=json.dumps(params.dict()))
        db.add(param_record)
    
    db.commit()
    
    logger.info(f"Trading parameters updated for user {current_user.email}, mode: {request.mode}")
    
    return {
        "status": "success",
        "message": f"Trading parameters for {request.mode} mode updated successfully",
        "parameters": params.dict()
    }


@router.post("/parameters/reset")
async def reset_to_defaults(
    mode: str,
    password: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Reset parameters to system defaults
    """
    # Verify password
    if not pwd_context.verify(password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    param_key = f"trading_params_{mode}_{current_user.id}"
    param_record = db.query(SystemState).filter(SystemState.key == param_key).first()
    
    if param_record:
        db.delete(param_record)
        db.commit()
    
    logger.info(f"Trading parameters reset to defaults for user {current_user.email}, mode: {mode}")
    
    return {
        "status": "success",
        "message": "Parameters reset to system defaults"
    }
