"""
Spec 5: Model registry for loading, versioning, and rollback
"""
from sqlalchemy.orm import Session
from app.db.models import Model
from app.ml.trainer import ModelTrainer
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versions and loading
    """
    
    @staticmethod
    def get_active_model(db: Session):
        """
        Get currently active model
        """
        model_record = db.query(Model).filter(Model.is_active == True).first()
        if not model_record:
            logger.warning("No active model found")
            return None, None
        
        trainer = ModelTrainer()
        try:
            model = trainer.load_model(model_record.model_path)
            return model, model_record
        except Exception as e:
            logger.error(f"Failed to load model {model_record.model_path}: {str(e)}")
            return None, None
    
    @staticmethod
    def activate_model(db: Session, model_id: str):
        """
        Spec 12: Activate a specific model version (rollback capability)
        """
        # Deactivate all
        db.query(Model).update({"is_active": False})
        
        # Activate selected
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            model.is_active = True
            db.commit()
            logger.info(f"Activated model: {model.name} ({model_id})")
            return True
        return False
    
    @staticmethod
    def list_models(db: Session, limit: int = 10):
        """
        List recent models
        """
        return db.query(Model).order_by(Model.trained_at.desc()).limit(limit).all()
