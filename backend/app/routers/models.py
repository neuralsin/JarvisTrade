"""
Models API routes - list models, trigger training, get metrics
"""
from fastapi import APIRouter, Depends, Background Tasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, Model
from app.routers.auth import get_current_user
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class TrainModelRequest(BaseModel):
    model_name: str
    instrument_filter: Optional[str] = None
    hyperparams: Optional[Dict[str, Any]] = None


@router.get("/")
async def get_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of trained models
    """
    models = db.query(Model).order_by(Model.trained_at.desc()).all()
    
    return {
        "models": [
            {
                "id": str(model.id),
                "name": model.name,
                "type": model.model_type,
                "trained_at": model.trained_at.isoformat() + "Z" if model.trained_at else None,
                "is_active": model.is_active,
                "metrics": model.metrics_json
            }
            for model in models
        ]
    }


@router.get("/{model_id}")
async def get_model_details(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 9: Model metrics with SHAP feature importance, AUC, precision, etc.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    
    if not model:
        return {"error": "Model not found"}
    
    return {
        "id": str(model.id),
        "name": model.name,
        "type": model.model_type,
        "trained_at": model.trained_at.isoformat() + "Z" if model.trained_at else None,
        "is_active": model.is_active,
        "metrics": model.metrics_json,
        "model_path": model.model_path
    }


@router.post("/train")
async def trigger_training(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 5: Trigger model training task
    """
    from app.tasks.model_training import train_model
    
    # Queue training task
    task = train_model.delay(
        model_name=request.model_name,
        instrument_filter=request.instrument_filter,
        hyperparams=request.hyperparams or {}
    )
    
    logger.info(f"Training task queued: {task.id} for model {request.model_name}")
    
    return {
        "status": "queued",
        "task_id": task.id,
        "model_name": request.model_name
    }


@router.post("/{model_id}/activate")
async def activate_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Spec 12: Activate a model for production use
    """
    # Deactivate all models
    db.query(Model).update({"is_active": False})
    
    # Activate selected model
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        return {"error": "Model not found"}
    
    model.is_active = True
    db.commit()
    
    logger.info(f"Model activated: {model.name} ({model_id})")
    
    return {"status": "success", "model_id": str(model.id), "name": model.name}
