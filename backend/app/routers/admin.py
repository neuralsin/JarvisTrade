"""
Spec 8 & 11: Admin routes - kill switch, settings, metrics
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, SystemState
from app.routers.auth import get_current_user
from app.config import settings
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Spec 11: Prometheus metrics
trades_executed = Counter('jarvis_trades_executed_total', 'Total trades executed', ['mode'])
trade_pnl_sum = Gauge('jarvis_trade_pnl_sum', 'Sum of trade P&L')
daily_loss = Gauge('jarvis_daily_loss', 'Daily loss per user', ['user_id'])
model_train_duration = Gauge('jarvis_model_train_duration_seconds', 'Model training duration')
kite_ws_status = Gauge('jarvis_kite_ws_status', 'Kite WebSocket status (0=down, 1=up)')
backfill_in_progress = Gauge('jarvis_backfill_in_progress', 'Backfill in progress (0/1)')


class KillSwitchRequest(BaseModel):
    enabled: bool
    password: str


@router.post("/kill-switch")
async def toggle_kill_switch(
    request: KillSwitchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 8: Kill switch to halt all live trading
    Requires password re-entry for security
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Verify password
    if not pwd_context.verify(request.password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    # Update kill switch state
    kill_switch = db.query(SystemState).filter(SystemState.key == "kill_switch").first()
    if kill_switch:
        kill_switch.value = "true" if request.enabled else "false"
    else:
        kill_switch = SystemState(key="kill_switch", value="true" if request.enabled else "false")
        db.add(kill_switch)
    
    db.commit()
    
    logger.warning(f"Kill switch {'ENABLED' if request.enabled else 'DISABLED'} by {current_user.email}")
    
    # TODO: Broadcast to Celery workers to update their cached state
    
    return {
        "status": "success",
        "kill_switch_enabled": request.enabled,
        "message": "All live trading " + ("halted" if request.enabled else "resumed")
    }


@router.get("/kill-switch")
async def get_kill_switch_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current kill switch status
    """
    kill_switch = db.query(SystemState).filter(SystemState.key == "kill_switch").first()
    enabled = kill_switch and kill_switch.value == "true"
    
    return {"kill_switch_enabled": enabled}


@router.post("/auto-execute")
async def toggle_auto_execute(
    enabled: bool,
    password: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 18: Two-step enable for live trading (toggle + password)
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Verify password
    if not pwd_context.verify(password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    # Check if Kite credentials exist
    if enabled and not current_user.kite_api_key_encrypted:
        raise HTTPException(status_code=400, detail="Kite credentials not configured")
    
    current_user.auto_execute = enabled
    db.commit()
    
    logger.warning(f"Auto-execute {'ENABLED' if enabled else 'DISABLED'} for {current_user.email}")
    
    return {
        "status": "success",
        "auto_execute": enabled
    }


@router.get("/metrics")
async def get_metrics():
    """
    Spec 11: Prometheus metrics endpoint
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/system-status")
async def get_system_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get overall system status
    """
    states = db.query(SystemState).all()
    state_dict = {s.key: s.value for s in states}
    
    return {
        "kill_switch": state_dict.get("kill_switch") == "true",
        "kws_status": state_dict.get("kws_status", "DOWN"),
        "market_status": state_dict.get("market_status", "UNKNOWN"),
        "environment": settings.APP_ENV
    }
