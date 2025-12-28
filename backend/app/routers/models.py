"""
Models API routes - list models, trigger training, get metrics
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, Model, Instrument
from app.routers.auth import get_current_user
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class TrainModelRequest(BaseModel):
    model_name: str
    model_type: str = 'xgboost'  
    instrument_filter: Optional[str] = None
    interval: str = '15m'  # Candle interval: 15m, 1h, 1d, etc.
    hyperparams: Optional[Dict[str, Any]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    csv_dataset_id: Optional[str] = None  # NEW: CSV dataset ID for training


class TrainV2Request(BaseModel):
    """V2 Dual-Model Training Request."""
    stock_symbol: str
    model_name: Optional[str] = None
    interval: str = '15m'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    hyperparams_direction: Optional[Dict[str, Any]] = None
    hyperparams_quality: Optional[Dict[str, Any]] = None


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
                "stock_symbol": model.stock_symbol,  # NEW: Include stock symbol
                "model_format": getattr(model, 'model_format', 'joblib'),  # NEW: Include format
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
        "stock_symbol": model.stock_symbol,  # NEW: Include stock symbol
        "model_format": getattr(model, 'model_format', 'joblib'),  # NEW: Include format
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
    
    # Validate interval
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
    if request.interval not in valid_intervals:
        return {"error": f"Invalid interval. Must be one of: {valid_intervals}"}
    
    # Validate duration limits based on interval (Yahoo Finance restrictions)
    if request.start_date and request.end_date:
        from datetime import datetime
        try:
            start = datetime.strptime(request.start_date, '%Y-%m-%d')
            end = datetime.strptime(request.end_date, '%Y-%m-%d')
            days = (end - start).days
            
            # Intraday intervals limited to 60 days
            intraday_intervals = ['1m', '5m', '15m', '30m', '1h']
            if request.interval in intraday_intervals and days > 60:
                return {
                    "error": f"Interval '{request.interval}' limited to 60 days by Yahoo Finance. Requested: {days} days",
                    "details": f"Please use a daily interval (1d) or reduce date range to max 60 days"
                }
            
            # Daily interval limited to ~2 years
            if request.interval == '1d' and days > 730:
                return {
                    "error": f"Interval '1d' limited to 730 days (2 years) by Yahoo Finance. Requested: {days} days",
                    "details": "Please use weekly/monthly interval or reduce date range"
                }
        except ValueError as e:
            return {"error": f"Invalid date format: {str(e)}"}
    
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
    
    # Validate stock/instrument filter is provided (unless training from CSV)
    if not request.instrument_filter:
        return {
            "error": "instrument_filter is required. Specify which stock to train on (e.g., 'RELIANCE', 'TATAELXSI')"
        }
    
    # Validate instrument exists in database (or will be created from CSV)
    instrument = db.query(Instrument).filter(
        Instrument.symbol == request.instrument_filter.upper()
    ).first()
    
    # Auto-create instrument if it doesn't exist (training task will fetch data)
    if not instrument and not request.csv_dataset_id:
        logger.info(f"Auto-creating instrument: {request.instrument_filter.upper()}")
        instrument = Instrument(
            symbol=request.instrument_filter.upper(),
            name=request.instrument_filter.upper(),
            exchange='NSE',
            instrument_type='EQ'
        )
        db.add(instrument)
        db.commit()
        db.refresh(instrument)
    
    # Queue training task
    try:
        # Trigger training task with ALL required parameters
        task = train_model.delay(
            model_name=request.model_name or f"auto_{request.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_type=request.model_type,
            instrument_filter=request.instrument_filter.upper(),  # ✅ CRITICAL: Pass this!
            interval=request.interval, # Pass interval to task
            start_date=request.start_date,
            end_date=request.end_date,
            hyperparams=request.hyperparams or {},
            csv_dataset_id=request.csv_dataset_id  # NEW: Pass CSV dataset ID if provided
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


@router.post("/train-v2")
async def trigger_training_v2(
    request: TrainV2Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    V2 Dual-Model Training: Train Direction Scout + Quality Gatekeeper models.
    
    Creates 3 models:
    - Model A: Direction Scout (multi-class: Neutral/Long/Short)
    - Model B Long: Quality Gatekeeper for long trades
    - Model B Short: Quality Gatekeeper for short trades
    """
    from app.tasks.model_training_v2 import train_v2_models
    from app.celery_app import celery_app
    
    # Validate stock symbol
    if not request.stock_symbol:
        return {"error": "stock_symbol is required"}
    
    stock_symbol = request.stock_symbol.strip().upper()
    
    # Validate interval
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d']
    if request.interval not in valid_intervals:
        return {"error": f"Invalid interval. Must be one of: {valid_intervals}"}
    
    # Auto-create instrument if needed
    instrument = db.query(Instrument).filter(
        Instrument.symbol == stock_symbol
    ).first()
    
    if not instrument:
        logger.info(f"Auto-creating instrument: {stock_symbol}")
        instrument = Instrument(
            symbol=stock_symbol,
            name=stock_symbol,
            exchange='NSE',
            instrument_type='EQ'
        )
        db.add(instrument)
        db.commit()
    
    # Queue V2 training task
    try:
        task = train_v2_models.delay(
            stock_symbol=stock_symbol,
            model_name=request.model_name,
            interval=request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
            hyperparams_direction=request.hyperparams_direction,
            hyperparams_quality=request.hyperparams_quality
        )
        
        logger.info(f"V2 training task queued: {task.id} for {stock_symbol}")
        
        return {
            "status": "queued",
            "task_id": task.id,
            "stock_symbol": stock_symbol,
            "engine_version": "v2",
            "models_to_train": ["direction", "quality_long", "quality_short"]
        }
    except Exception as e:
        logger.error(f"Failed to queue V2 training task: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to queue V2 training task: {str(e)}",
            "details": "Check backend and Celery worker logs"
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
