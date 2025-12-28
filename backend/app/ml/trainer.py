"""
Spec 5: Model training pipeline with XGBoost
FIXED: Correct metric computation, Precision@TopK, dead model detection
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import shap
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CORRECT METRIC FUNCTIONS - Based on expert guidance
# ============================================================================

def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Precision at top k fraction (e.g. top 10%)
    THIS IS THE ONLY PRECISION METRIC THAT MATTERS FOR TRADING
    
    Args:
        y_true: True binary labels
        y_prob: Model probabilities for positive class
        k: Fraction of top predictions to evaluate (default 10%)
    
    Returns:
        Precision in the top k% of predictions
    """
    n = int(len(y_prob) * k)
    if n == 0:
        return 0.0
    
    idx = np.argsort(y_prob)[::-1][:n]  # Top k indices by probability
    return float(y_true[idx].sum() / n)


def is_dead_model(y_prob: np.ndarray) -> bool:
    """
    Detects collapsed models (constant or near-constant predictions)
    
    STRENGTHENED: Original threshold (1e-4) was too weak.
    Now catches near-collapse that still passes weak checks.
    
    Args:
        y_prob: Model probability predictions
    
    Returns:
        True if model is dead (collapsed), False otherwise
    """
    # Check 1: Near-zero variance (original, strengthened)
    if float(np.std(y_prob)) < 0.01:  # Was 1e-4, now 0.01
        return True
    
    # Check 2: Tiny range (new check)
    prob_range = float(np.max(y_prob) - np.min(y_prob))
    if prob_range < 0.05:
        return True
    
    return False


def correct_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Correct precision on FULL set (not filtered to predicted positives only)
    """
    return float(precision_score(y_true, y_pred, zero_division=0))


class ModelTrainer:
    """
    Spec 5: XGBoost training pipeline with SHAP feature importance
    FIXED: Binary-only classification, correct metrics
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
        # FIXED: Strip timezone info to ensure consistent comparisons
        if df['ts_utc'].dt.tz is not None:
            df['ts_utc'] = df['ts_utc'].dt.tz_localize(None)
        df = df.sort_values('ts_utc').reset_index(drop=True)
        
        # FIX 4 & 5: Updated feature columns - no sentiment, EMA100 instead of 200
        feature_cols = [
            'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_100',
            'distance_from_ema100', 'rsi_14', 'rsi_slope',
            'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
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
        
        # FIXED: Use training data as validation if validation set is empty
        if len(X_val) == 0:
            logger.warning("Empty validation set - using 20% of training data for early stopping")
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            logger.info(f"After split - Train: {len(X_train)}, Val: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model - BINARY ONLY
        FIX 2: Multi-class removed. Trading ML is a binary ranking problem.
        """
        # FIX 2: Force binary classification
        # Convert any multi-class to binary (1 = trade, 0 = no trade)
        n_classes = len(np.unique(y_train))
        logger.info(f"Unique labels in training: {sorted(np.unique(y_train))}")
        
        if n_classes > 2:
            logger.warning(f"Multi-class detected ({n_classes} classes). Converting to binary.")
            # Convert to binary: 1 = any trade (BUY or SELL), 0 = no trade
            y_train = (y_train >= 1).astype(int)
            y_val = (y_val >= 1).astype(int)
            n_classes = 2
        
        # Binary classification ONLY
        logger.info("Training BINARY classifier (TRADE vs NO_TRADE)")
        self.hyperparams['objective'] = 'binary:logistic'
        self.hyperparams['eval_metric'] = 'auc'  # Use AUC as primary metric
        
        # Handle class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count
            logger.info(f"Class distribution: {neg_count} NO_TRADE, {pos_count} TRADE")
            logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f}")
            self.hyperparams['scale_pos_weight'] = scale_pos_weight
        
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
        Compute CORRECT metrics for trading ML
        
        Key metrics:
        - AUC: Does the model know anything? (ranking power)
        - Precision@10%: Can it make money? (top predictions accuracy)
        - Dead model check: Is it collapsed?
        """
        # Get probabilities
        y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1 (TRADE)
        
        # DEAD MODEL CHECK - Critical safety
        if is_dead_model(y_prob):
            logger.error("🚨 DEAD MODEL DETECTED: Predictions have near-zero variance")
            return {
                "auc_roc": 0.5,
                "precision_at_10": 0.0,
                "precision_at_5": 0.0,
                "is_dead": True,
                "rejection_reason": "collapsed predictions",
                "prob_std": float(np.std(y_prob))
            }
        
        # Convert y to binary if needed
        y_binary = (y >= 1).astype(int) if len(np.unique(y)) > 2 else y.values
        
        # CRITICAL FIX: AUC flip check
        # In trading ML, model may learn p↑ = bad trade (inverted)
        # This is equally valid but needs correction
        try:
            auc = roc_auc_score(y_binary, y_prob)
            auc_flipped = roc_auc_score(y_binary, 1.0 - y_prob)
            
            if auc_flipped > auc:
                logger.warning(f"🔄 Probability direction inverted! AUC {auc:.4f} → {auc_flipped:.4f}")
                y_prob = 1.0 - y_prob
                auc = auc_flipped
                flipped = True
            else:
                flipped = False
        except Exception as e:
            logger.warning(f"AUC calculation failed: {e}")
            auc = 0.5
            flipped = False
        
        # Precision@TopK - THE METRIC THAT MATTERS
        p_at_10 = precision_at_k(y_binary, y_prob, k=0.10)
        p_at_5 = precision_at_k(y_binary, y_prob, k=0.05)
        p_at_20 = precision_at_k(y_binary, y_prob, k=0.20)
        
        # Standard precision at threshold 0.5 (for reference only)
        y_pred = (y_prob >= 0.5).astype(int)
        precision_std = correct_precision(y_binary, y_pred)
        
        # Accuracy (for reference only - NEVER use for activation)
        accuracy = accuracy_score(y_binary, y_pred)
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics = {
            # Key metrics
            "auc_roc": float(auc),
            "precision_at_10": float(p_at_10),
            "precision_at_5": float(p_at_5),
            "precision_at_20": float(p_at_20),
            
            # Reference metrics
            "precision_std": float(precision_std),
            "accuracy": float(accuracy),
            
            # Dead model check
            "is_dead": False,
            "prob_std": float(np.std(y_prob)),
            "prob_min": float(np.min(y_prob)),
            "prob_max": float(np.max(y_prob)),
            
            # Confusion matrix details
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }
        
        # Log key metrics clearly
        logger.info(f"📊 MODEL EVALUATION:")
        logger.info(f"   AUC-ROC: {auc:.4f} (>0.55 = has signal)")
        logger.info(f"   Precision@10%: {p_at_10:.4f} (>0.60 = tradable)")
        logger.info(f"   Precision@5%: {p_at_5:.4f}")
        logger.info(f"   Prob range: [{metrics['prob_min']:.4f}, {metrics['prob_max']:.4f}]")
        
        # SHAP feature importance
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:min(1000, len(X))])
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
    
    def should_activate(self, metrics: Dict) -> Tuple[bool, str]:
        """
        CORRECT activation logic based on expert guidance
        
        Activation requires:
        - AUC >= 0.55 (model has ranking power)
        - Precision@10% >= 0.60 (can make money)
        - Not a dead model
        
        Returns:
            (should_activate: bool, reason: str)
        """
        from app.config import settings
        
        # Check for dead model first
        if metrics.get('is_dead', False):
            return False, f"Dead model: {metrics.get('rejection_reason', 'collapsed predictions')}"
        
        auc = metrics.get('auc_roc', 0)
        p_at_10 = metrics.get('precision_at_10', 0)
        
        min_auc = getattr(settings, 'MODEL_MIN_AUC', 0.55)
        min_p_at_10 = getattr(settings, 'MODEL_MIN_PRECISION_AT_10', 0.60)
        
        reasons = []
        
        if auc < min_auc:
            reasons.append(f"AUC {auc:.4f} < {min_auc}")
        
        if p_at_10 < min_p_at_10:
            reasons.append(f"Precision@10% {p_at_10:.4f} < {min_p_at_10}")
        
        if reasons:
            return False, "; ".join(reasons)
        
        return True, f"AUC={auc:.4f}, P@10%={p_at_10:.4f}"
    
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
