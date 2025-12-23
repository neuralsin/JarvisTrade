"""
Models API routes - list models, trigger training, get metrics
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
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
    model_type: str = 'xgboost'  
    instrument_filter: Optional[str] = None
    hyperparams: Optional[Dict[str, Any]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


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
    from app.celery_app import celery_app
    from celery.result import AsyncResult
    
    # Validate model type
    valid_types = ['xgboost', 'lstm', 'transformer']
    if request.model_type not in valid_types:
        return {"error": f"Invalid model_type. Must be one of: {valid_types}"}
    
    # Check if Celery worker is available
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        if not active_workers:
            logger.warning("No Celery workers available")
            return {
                "error": "Training service unavailable. Please ensure Celery worker is running.",
                "details": "Run: docker-compose up -d celery_worker"
            }
    except Exception as e:
        logger.error(f"Failed to check Celery worker status: {str(e)}")
        # Continue anyway - the task will queue even if we can't check worker status
    
    # Queue training task
    try:
        task = train_model.delay(
            model_name=request.model_name,
            model_type=request.model_type,
            instrument_filter=request.instrument_filter,
            hyperparams=request.hyperparams or {},
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        logger.info(f"Training task queued: {task.id} for model {request.model_name} (type: {request.model_type})")
        
        return {
            "status": "queued",
            "task_id": task.id,
            "model_name": request.model_name,
            "model_type": request.model_type
        }
    except Exception as e:
        logger.error(f"Failed to queue training task: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to queue training task: {str(e)}",
            "details": "Check backend and Celery worker logs for more information"
        }


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get training task status and progress
    """
    from app.celery_app import celery_app
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": result.state,
    }
    
    if result.state == 'PENDING':
        response["progress"] = 0
        response["message"] = "Task is waiting to start..."
    elif result.state == 'PROGRESS':
        info = result.info or {}
        response["progress"] = info.get('progress', 0)
        response["message"] = info.get('status', 'Training in progress...')
    elif result.state == 'SUCCESS':
        response["progress"] = 100
        response["message"] = "Training completed successfully!"
        response["result"] = result.result
    elif result.state == 'FAILURE':
        response["progress"] = 0
        response["message"] = f"Training failed: {str(result.info)}"
        response["error"] = str(result.info)
    
    return response


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


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a trained model
    """
    import os
    import shutil
    from uuid import UUID
    
    try:
        # Validate UUID
        try:
            model_uuid = UUID(model_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid model ID format")
        
        # Get model
        model = db.query(Model).filter(Model.id == model_uuid).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prevent deleting active model
        if model.is_active:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete active model. Please activate another model first."
            )
        
        # Delete model files from disk (if they exist)
        if model.model_path and os.path.exists(model.model_path):
            try:
                shutil.rmtree(model.model_path)
                logger.info(f"Deleted model files from {model.model_path}")
            except Exception as e:
                logger.warning(f"Could not delete model files: {str(e)}")
        
        # Delete from database
        model_name = model.name
        db.delete(model)
        db.commit()
        
        logger.info(f"Model {model_name} (ID: {model_id}) deleted by {current_user.email}")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
