"""
Phase 5: Trading Controls API Endpoints

Provides RESTful API for:
- Paper trading toggle
- Model selection/deselection
- Active model management
- Signal log querying
- Manual signal triggering
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from app.db.database import get_db
from app.db.models import User, Model
from app.db.signal_log import SignalLog
from app.ml.model_selector import ModelSelector
from app.routers.auth import get_current_user
from app.tasks.execution import check_and_execute_signals
import logging

router = APIRouter(prefix="/api/trading", tags=["trading"])
logger = logging.getLogger(__name__)


# Request/Response Models
class TradingStatusResponse(BaseModel):
    paper_trading_enabled: bool
    auto_execute: bool
    selected_model_count: int
    selected_stocks: List[str]
    
class PaperTradingToggleRequest(BaseModel):
    enabled: bool

class ModelSelectRequest(BaseModel):
    model_id: str

class AvailableModelsResponse(BaseModel):
    stock_symbol: str
    models: List[dict]

class SignalLogResponse(BaseModel):
    id: str
    timestamp: str
    stock_symbol: str
    model_name: str
    probability: Optional[float]
    action: str
    reason: Optional[str]
    trade_id: Optional[str]


@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current trading status for user.
    
    Returns paper trading state, auto-execute state, and selected models.
    """
    selector = ModelSelector(db)
    selected_models = selector.get_models_for_user(str(current_user.id))
    
    return TradingStatusResponse(
        paper_trading_enabled=current_user.paper_trading_enabled or False,
        auto_execute=current_user.auto_execute or False,
        selected_model_count=len(selected_models),
        selected_stocks=[m.stock_symbol for m in selected_models if m.stock_symbol]
    )


@router.post("/paper/toggle")
async def toggle_paper_trading(
    request: PaperTradingToggleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enable or disable paper trading for user.
    """
    current_user.paper_trading_enabled = request.enabled
    db.commit()
    
    logger.info(f"Paper trading {'enabled' if request.enabled else 'disabled'} for {current_user.email}")
    
    return {
        "success": True,
        "paper_trading_enabled": request.enabled,
        "message": f"Paper trading {'enabled' if request.enabled else 'disabled'}"
    }


@router.get("/models/available", response_model=List[AvailableModelsResponse])
async def get_available_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all available models grouped by stock.
    
    Returns models with their metrics, grouped by stock_symbol.
    """
    selector = ModelSelector(db)
    grouped = selector.get_available_models(str(current_user.id))
    
    # Get user's current selections
    user_selections = current_user.selected_model_ids or []
    
    result = []
    for stock, models in grouped.items():
        model_list = []
        for model in models:
            model_list.append({
                "id": str(model.id),
                "name": model.name,
                "model_type": model.model_type,
                "is_active": model.is_active,
                "is_selected": str(model.id) in user_selections,
                "trained_at": model.trained_at.isoformat() if model.trained_at else None,
                "metrics": {
                    "auc_roc": model.metrics_json.get('auc_roc') if model.metrics_json else None,
                    "accuracy": model.metrics_json.get('test_accuracy') if model.metrics_json else None
                }
            })
        
        result.append(AvailableModelsResponse(
            stock_symbol=stock,
            models=model_list
        ))
    
    return result


@router.get("/models/selected")
async def get_selected_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's currently selected models.
    """
    selector = ModelSelector(db)
    summary = selector.get_selection_summary(str(current_user.id))
    
    return summary


@router.post("/models/select")
async def select_model(
    request: ModelSelectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a model to user's selected models.
    """
    selector = ModelSelector(db)
    success = selector.select_model(str(current_user.id), request.model_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not found or already selected"
        )
    
    return {
        "success": True,
        "message": "Model added to selections",
        "model_id": request.model_id
    }


@router.delete("/models/select/{model_id}")
async def deselect_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove a model from user's selected models.
    """
    selector = ModelSelector(db)
    success = selector.deselect_model(str(current_user.id), model_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not in selections"
        )
    
    return {
        "success": True,
        "message": "Model removed from selections",
        "model_id": model_id
    }


@router.post("/models/select-all")
async def select_all_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Select all active models for user.
    """
    selector = ModelSelector(db)
    count = selector.select_all_for_user(str(current_user.id))
    
    return {
        "success": True,
        "message": f"Selected {count} models",
        "count": count
    }


@router.delete("/models/clear")
async def clear_model_selections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Clear all model selections for user.
    """
    selector = ModelSelector(db)
    success = selector.clear_selections(str(current_user.id))
    
    return {
        "success": success,
        "message": "All model selections cleared"
    }


@router.get("/signals/recent", response_model=List[SignalLogResponse])
async def get_recent_signals(
    limit: int = 100,
    stock_symbol: Optional[str] = None,
    action: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get recent signal logs for user.
    
    Args:
        limit: Maximum number of signals to return (default 100)
        stock_symbol: Optional filter by stock symbol
        action: Optional filter by action (EXECUTE or REJECT)
    """
    query = db.query(SignalLog).filter(
        SignalLog.user_id == current_user.id
    )
    
    # Apply filters
    if stock_symbol:
        from app.db.models import Instrument
        query = query.join(Instrument).filter(Instrument.symbol == stock_symbol)
    
    if action:
        query = query.filter(SignalLog.action == action.upper())
    
    # Order by most recent first
    signals = query.order_by(SignalLog.ts_utc.desc()).limit(limit).all()
    
    # Convert to response model
    result = []
    for signal in signals:
        result.append(SignalLogResponse(
            id=str(signal.id),
            timestamp=signal.ts_utc.isoformat(),
            stock_symbol=signal.instrument.symbol if signal.instrument else "UNKNOWN",
            model_name=signal.model.name if signal.model else "UNKNOWN",
            probability=signal.probability,
            action=signal.action,
            reason=signal.reason,
            trade_id=str(signal.trade_id) if signal.trade_id else None
        ))
    
    return result


@router.post("/check-now")
async def trigger_signal_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Manually trigger a signal check (bypass scheduled interval).
    
    Useful for testing or immediate signal checking.
    """
    try:
        # Fire and forget - don't wait for result
        check_and_execute_signals.apply_async()
        
        return {
            "success": True,
            "message": "Signal check triggered",
            "note": "Check will run asynchronously. Monitor signal logs for results."
        }
    except Exception as e:
        logger.error(f"Failed to trigger signal check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger signal check: {str(e)}"
        )
