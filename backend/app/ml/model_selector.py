"""
Phase 3: Model Selection Service

Manages per-stock model selection and retrieval for multi-model architecture.
Users can select which models (stocks) they want to trade on simultaneously.
"""
from sqlalchemy.orm import Session
from app.db.models import Model, User
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Phase 3: Service for managing user's multi-model selections
    
    Enables users to select multiple models (one per stock) to run in parallel.
    Each model only monitors and trades its assigned stock.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_models_for_user(self, user_id: str) -> List[Model]:
        """
        Get all active models selected by a user.
        
        Args:
            user_id: User UUID (as string)
        
        Returns:
            List of Model objects that are:
            - Selected by the user (in user.selected_model_ids)
            - Currently active (is_active=True)
            - Sorted by stock_symbol for consistent ordering
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.warning(f"User {user_id} not found")
            return []
        
        if not user.paper_trading_enabled:
            logger.info(f"Paper trading disabled for user {user.email}")
            return []
        
        # Get selected model IDs (empty list if None)
        selected_ids = user.selected_model_ids or []
        
        if not selected_ids:
            logger.info(f"No models selected for user {user.email}")
            return []
        
        # Query models that are both selected AND active
        models = self.db.query(Model).filter(
            Model.id.in_(selected_ids),
            Model.is_active == True
        ).order_by(Model.stock_symbol).all()
        
        logger.info(
            f"Retrieved {len(models)} active model(s) for user {user.email}: "
            f"{[f'{m.stock_symbol}({m.name})' for m in models]}"
        )
        
        return models
    
    def get_model_for_stock(self, stock_symbol: str, user_id:Optional[str] = None) -> Optional[Model]:
        """
        Get the active model for a specific stock.
        
        Args:
            stock_symbol: Stock symbol (e.g., 'RELIANCE')
            user_id: Optional user ID to filter by user's selections
        
        Returns:
            Model object if found and active, None otherwise
        """
        query = self.db.query(Model).filter(
            Model.stock_symbol == stock_symbol.upper(),
            Model.is_active == True
        )
        
        # If user_id provided, filter by user's selections
        if user_id:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user and user.selected_model_ids:
                query = query.filter(Model.id.in_(user.selected_model_ids))
        
        model = query.first()
        
        if model:
            logger.info(f"Found active model for {stock_symbol}: {model.name}")
        else:
            logger.warning(f"No active model found for {stock_symbol}")
        
        return model
    
    def select_model(self, user_id: str, model_id: str) -> bool:
        """
        Add a model to user's selected models.
        
        Args:
            user_id: User UUID (as string)
            model_id: Model UUID (as string)
        
        Returns:
            True if added successfully, False if already selected or error
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        # Check if model exists and is active
        model = self.db.query(Model).filter(
            Model.id == model_id,
            Model.is_active == True
        ).first()
        
        if not model:
            logger.error(f"Model {model_id} not found or not active")
            return False
        
        # Initialize selected_model_ids if None
        if user.selected_model_ids is None:
            user.selected_model_ids = []
        
        # Check if already selected
        if model_id in user.selected_model_ids:
            logger.info(f"Model {model.name} already selected for user {user.email}")
            return True
        
        # Add to selections
        user.selected_model_ids = user.selected_model_ids + [model_id]
        self.db.commit()
        
        logger.info(f"✓ Model {model.name} ({model.stock_symbol}) added to selections for {user.email}")
        return True
    
    def deselect_model(self, user_id: str, model_id: str) -> bool:
        """
        Remove a model from user's selected models.
        
        Args:
            user_id: User UUID (as string)
            model_id: Model UUID (as string)
        
        Returns:
            True if removed successfully, False if not selected or error
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        if not user.selected_model_ids or model_id not in user.selected_model_ids:
            logger.warning(f"Model {model_id} not in user's selections")
            return False
        
        # Remove from selections
        user.selected_model_ids = [mid for mid in user.selected_model_ids if mid != model_id]
        self.db.commit()
        
        logger.info(f"✓ Model {model_id} removed from selections for {user.email}")
        return True
    
    def select_all_for_user(self, user_id: str) -> int:
        """
        Select all active models for a user.
        
        Args:
            user_id: User UUID (as string)
        
        Returns:
            Number of models selected
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"User {user_id} not found")
            return 0
        
        # Get all active models
        active_models = self.db.query(Model).filter(Model.is_active == True).all()
        
        # Set all active model IDs
        user.selected_model_ids = [str(m.id) for m in active_models]
        self.db.commit()
        
        logger.info(f"✓ Selected all {len(active_models)} active models for {user.email}")
        return len(active_models)
    
    def clear_selections(self, user_id: str) -> bool:
        """
        Clear all model selections for a user.
        
        Args:
            user_id: User UUID (as string)
        
        Returns:
            True if cleared successfully
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        user.selected_model_ids = []
        self.db.commit()
        
        logger.info(f"✓ Cleared all model selections for {user.email}")
        return True
    
    def get_available_models(self, user_id: Optional[str] = None) -> Dict[str, List[Model]]:
        """
        Get all available models grouped by stock symbol.
        
        Args:
            user_id: Optional user ID to mark which are selected
        
        Returns:
            Dict mapping stock_symbol -> list of models for that stock
        """
        # Get all active models
        models = self.db.query(Model).filter(Model.is_active == True).all()
        
        # Group by stock
        grouped = {}
        for model in models:
            stock = model.stock_symbol or "UNKNOWN"
            if stock not in grouped:
                grouped[stock] = []
            grouped[stock].append(model)
        
        logger.info(f"Found {len(models)} active models across {len(grouped)} stocks")
        return grouped
    
    def get_selection_summary(self, user_id: str) -> Dict:
        """
        Get summary of user's model selections.
        
        Args:
            user_id: User UUID (as string)
        
        Returns:
            Dict with selection stats and model details
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return {"error": "User not found"}
        
        selected_models = self.get_models_for_user(user_id)
        available_models = self.db.query(Model).filter(Model.is_active == True).count()
        
        return {
            "user_email": user.email,
            "paper_trading_enabled": user.paper_trading_enabled,
            "selected_count": len(selected_models),
            "available_count": available_models,
            "selected_stocks": [m.stock_symbol for m in selected_models],
            "selected_models": [
                {
                    "id": str(m.id),
                    "name": m.name,
                    "stock": m.stock_symbol,
                    "type": m.model_type,
                    "auc": m.metrics_json.get('auc_roc', 'N/A') if m.metrics_json else 'N/A'
                }
                for m in selected_models
            ]
        }


# Global helper functions for backward compatibility
def get_active_models_for_user(db: Session, user_id: str) -> List[Model]:
    """
    Helper function: Get active models for a user.
    
    This is the primary function used by execution.py
    """
    selector = ModelSelector(db)
    return selector.get_models_for_user(user_id)


def get_model_for_stock(db: Session, stock_symbol: str) -> Optional[Model]:
    """
    Helper function: Get active model for a specific stock.
    """
    selector = ModelSelector(db)
    return selector.get_model_for_stock(stock_symbol)
