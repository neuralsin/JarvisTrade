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
    
    
    def prepare_data(self, df: pd.DataFrame, train_end: str = None, 
                     val_end: str = None) -> Tuple:
        """
        Create train/val/test splits
        If train_end/val_end not provided, uses 70%-15%-15% split automatically
        """
        df = df.copy()
        df['ts_utc'] = pd.to_datetime(df['ts_utc'])
        df = df.sort_values('ts_utc').reset_index(drop=True)
        
        # Feature columns (must match feature_engineer.py output)
        feature_cols = [
            'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_200',
            'distance_from_ema200', 'rsi_14', 'rsi_slope',
            'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix',
            'sentiment_1d', 'sentiment_3d', 'sentiment_7d'
        ]
        
        # Remove rows with null targets
        df = df[df['target'].notna()].copy()
        
        if len(df) == 0:
            raise ValueError("No data after filtering null targets")
        
        # Dynamic date-based splits if not provided
        if train_end is None or val_end is None:
            # Use 70% train, 15% val, 15% test
            n = len(df)
            train_idx = int(n * 0.70)
            val_idx = int(n * 0.85)
            
            train_df = df.iloc[:train_idx]
            val_df = df.iloc[train_idx:val_idx]
            test_df = df.iloc[val_idx:]
            
            logger.info(f"Using automatic 70-15-15 split")
            logger.info(f"Date ranges: Train={train_df['ts_utc'].min()} to {train_df['ts_utc'].max()}, "
                       f"Val={val_df['ts_utc'].min()} to {val_df['ts_utc'].max()}, "
                       f"Test={test_df['ts_utc'].min()} to {test_df['ts_utc'].max()}")
        else:
            # Use provided dates
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
        
        if len(X_train) == 0:
            raise ValueError(f"Empty training set! Check date range or data availability.")
        if len(X_val) == 0:
            logger.warning("Empty validation set - will use train set for early stopping")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model for MULTI-CLASS classification (HOLD/BUY/SELL)
        """
        # Check number of unique classes
        n_classes = len(np.unique(y_train))
        logger.info(f"Training for {n_classes} classes: {sorted(np.unique(y_train))}")
        
        # Configure for multi-class if we have 3 classes (HOLD/BUY/SELL)
        if n_classes == 3:
            logger.info("Detected 3-class problem (HOLD/BUY/SELL). Using multi:softprob objective.")
            self.hyperparams['objective'] = 'multi:softprob'
            self.hyperparams['num_class'] = 3
            self.hyperparams['eval_metric'] = 'mlogloss'
            # Remove scale_pos_weight - not applicable for multi-class
            self.hyperparams.pop('scale_pos_weight', None)
            
            # Log class distribution
            for cls in [0, 1, 2]:
                count = (y_train == cls).sum()
                label = ['HOLD', 'BUY', 'SELL'][cls]
                logger.info(f"  Class {cls} ({label}): {count} samples ({count/len(y_train)*100:.1f}%)")
        
        elif n_classes == 2:
            # Binary classification (backward compatibility)
            logger.info("Detected 2-class problem. Using binary:logistic objective.")
            self.hyperparams['objective'] = 'binary:logistic'
            self.hyperparams['eval_metric'] = 'logloss'
            
            # Handle class imbalance for binary
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                scale_pos_weight = neg_count / pos_count
                logger.info(f"Class imbalance: {neg_count} negative, {pos_count} positive")
                logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f}")
                self.hyperparams['scale_pos_weight'] = scale_pos_weight
        else:
            raise ValueError(f"Unsupported number of classes: {n_classes}. Expected 2 or 3.")
        
        model = XGBClassifier(**self.hyperparams)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        logger.info(f"✓ Model trained. Best iteration: {model.best_iteration}")
        return model
    
    def evaluate(self, model, X, y, feature_cols) -> Dict:
        """
        Compute metrics for both binary and multi-class models
        """
        n_classes = len(np.unique(y))
        
        if n_classes == 3:
            # Multi-class: predict probabilities for each class
            y_pred_proba = model.predict_proba(X)  # Shape: (n_samples, 3)
            y_pred = model.predict(X)  # Direct class predictions
            
            # For multi-class AUC, use one-vs-rest
            try:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_auc_score
                y_bin = label_binarize(y, classes=[0, 1, 2])
                auc = roc_auc_score(y_bin, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:  # ✅ FIXED: Proper exception handling
                logger.warning(f"Multi-class AUC calculation failed: {str(e)}. Using accuracy instead")
                auc = (y_pred == y).mean()
            
            # Multi-class F1
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Confusion matrix for multi-class
            cm = confusion_matrix(y, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Precision for BUY class (most important)
            buy_precision = precision_score(y, y_pred, labels=[1], average='micro', zero_division=0)
            
            metrics = {
                "auc_roc": float(auc),
                "f1": float(f1),
                "accuracy": float((y_pred == y).mean()),
                "buy_precision": float(buy_precision),
                "confusion_matrix": cm.tolist(),
                "class_distribution": {
                    "HOLD": int((y_pred == 0).sum()),
                    "BUY": int((y_pred == 1).sum()),
                    "SELL": int((y_pred == 2).sum())
                }
            }
        
        else:
            # Binary classification (original logic)
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= 0.3).astype(int)
            
            auc = roc_auc_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            
            # Precision at top 10%
            threshold = np.quantile(y_pred_proba, 0.9)
            high_prob_mask = y_pred_proba >= threshold
            precision_at_k = precision_score(y[high_prob_mask], y_pred[high_prob_mask]) if high_prob_mask.sum() > 0 else 0
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            metrics = {
                "auc_roc": float(auc),
                "precision_at_k": float(precision_at_k),
                "f1": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }
        
        # SHAP feature importance (works for both)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:min(1000, len(X))])
            
            if n_classes == 3:
                # For multi-class, average across classes
                shap_avg = np.abs(shap_values).mean(axis=(0, 2)) if len(np.array(shap_values).shape) == 3 else np.abs(shap_values).mean(axis=0)
            else:
                shap_avg = np.abs(shap_values).mean(axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': shap_avg
            }).sort_values('importance', ascending=False).head(15)
            
            metrics['top_features'] = feature_importance.to_dict('records')
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            metrics['top_features'] = []
        
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
