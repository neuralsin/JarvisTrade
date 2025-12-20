"""
Spec 5: Model training pipeline with XGBoost
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import shap
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Spec 5: XGBoost training pipeline with SHAP feature importance
    """
    
    def __init__(self, hyperparams: Optional[Dict] = None):
        """
        Initialize trainer with hyperparameters
        """
        from app.config import settings
        
        self.hyperparams = hyperparams or {
            "n_estimators": settings.XGBOOST_N_ESTIMATORS,
            "max_depth": settings.XGBOOST_MAX_DEPTH,
            "learning_rate": settings.XGBOOST_LEARNING_RATE,
            "subsample": settings.XGBOOST_SUBSAMPLE,
            "colsample_bytree": settings.XGBOOST_COLSAMPLE_BYTREE,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        }
    
    def prepare_data(self, df: pd.DataFrame, train_end: str = "2022-12-31", 
                     val_end: str = "2023-12-31") -> Tuple:
        """
        Spec 5: Create train/val/test splits
        train = 2015-2022, val = 2023, test = 2024+
        """
        df = df.copy()
        df['ts_utc'] = pd.to_datetime(df['ts_utc'])
        
        # Feature columns
        feature_cols = [
            'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_200',
            'distance_from_ema200', 'rsi_14', 'rsi_slope',
            'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
        ]
        
        # Remove rows with null targets
        df = df[df['target'].notna()].copy()
        
        # Split by date
        train_df = df[df['ts_utc'] <= train_end]
        val_df = df[(df['ts_utc'] > train_end) & (df['ts_utc'] <= val_end)]
        test_df = df[df['ts_utc'] > val_end]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['target'].astype(int)
        
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df['target'].astype(int)
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['target'].astype(int)
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model with early stopping
        """
        model = XGBClassifier(**self.hyperparams)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        logger.info(f"Model trained. Best iteration: {model.best_iteration}")
        return model
    
    def evaluate(self, model, X, y, feature_cols) -> Dict:
        """
        Spec 5: Compute metrics including AUC, precision@k, F1, SHAP
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Basic metrics
        auc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        
        # Precision at top 10% probability
        threshold = np.quantile(y_pred_proba, 0.9)
        high_prob_mask = y_pred_proba >= threshold
        if high_prob_mask.sum() > 0:
            precision_at_k = precision_score(y[high_prob_mask], y_pred[high_prob_mask])
        else:
            precision_at_k = 0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # SHAP feature importance
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:min(1000, len(X))])  # Sample for speed
            
            # Mean absolute SHAP values
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False).head(15)
            
            top_features = feature_importance.to_dict('records')
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            top_features = []
        
        metrics = {
            "auc_roc": float(auc),
            "precision_at_k": float(precision_at_k),
            "f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "top_features": top_features
        }
        
        return metrics
    
    def save_model(self, model, model_name: str, metrics: Dict) -> str:
        """
        Save model artifact
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{model_name}_{timestamp}.pkl"
        
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load saved model
        """
        return joblib.load(model_path)
