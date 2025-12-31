"""
Walk-Forward Validation - V3 Fix

Prevent overfitting with rolling out-of-sample testing.

Instead of single train/test split, use multiple periods:
1. Train on period 1, test on period 2
2. Train on periods 1-2, test on period 3
3. etc.

Model must pass ALL periods to be activated.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation"""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    
    # Results
    metrics: Optional[Dict] = None
    passed: Optional[bool] = None


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result"""
    folds: List[WalkForwardFold]
    all_passed: bool
    avg_auc: float
    avg_precision_at_10: float
    min_auc: float
    std_auc: float
    rejection_reason: Optional[str] = None


class WalkForwardValidator:
    """
    Walk-forward validation to detect overfitting.
    
    Key principle: A model that only works on one period is overfit.
    """
    
    def __init__(
        self,
        n_folds: int = 3,
        train_ratio: float = 0.7,
        min_train_samples: int = 500,
        min_test_samples: int = 100,
        gap_periods: int = 1  # Gap between train and test to prevent leakage
    ):
        """
        Initialize validator.
        
        Args:
            n_folds: Number of rolling folds
            train_ratio: Ratio of data for training in each fold
            min_train_samples: Minimum samples for training
            min_test_samples: Minimum samples for testing
            gap_periods: Gap periods between train and test
        """
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.gap_periods = gap_periods
    
    def create_folds(
        self,
        df: pd.DataFrame,
        time_col: str = 'ts_utc'
    ) -> List[WalkForwardFold]:
        """
        Create walk-forward folds from data.
        
        Args:
            df: DataFrame with time column
            time_col: Name of timestamp column
            
        Returns:
            List of WalkForwardFold objects
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        n = len(df)
        
        # Calculate fold sizes
        total_periods = self.n_folds + 1  # n_folds train-test pairs
        period_size = n // total_periods
        
        folds = []
        
        for i in range(self.n_folds):
            # Expanding window: train on all previous data
            train_end_idx = (i + 1) * period_size
            test_start_idx = train_end_idx + self.gap_periods
            test_end_idx = (i + 2) * period_size
            
            # Ensure we have enough data
            if test_end_idx > n:
                test_end_idx = n
            if test_start_idx >= test_end_idx:
                continue
            
            train_data = df.iloc[:train_end_idx]
            test_data = df.iloc[test_start_idx:test_end_idx]
            
            if len(train_data) < self.min_train_samples:
                logger.warning(f"Fold {i}: insufficient train samples ({len(train_data)})")
                continue
            if len(test_data) < self.min_test_samples:
                logger.warning(f"Fold {i}: insufficient test samples ({len(test_data)})")
                continue
            
            fold = WalkForwardFold(
                fold_idx=i,
                train_start=train_data[time_col].iloc[0],
                train_end=train_data[time_col].iloc[-1],
                test_start=test_data[time_col].iloc[0],
                test_end=test_data[time_col].iloc[-1],
                train_samples=len(train_data),
                test_samples=len(test_data)
            )
            folds.append(fold)
        
        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds
    
    def validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        train_fn: Callable,
        eval_fn: Callable,
        time_col: str = 'ts_utc',
        min_auc: float = 0.52,
        min_precision_at_10: float = 0.40
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            df: Complete dataset
            feature_cols: Feature column names
            target_col: Target column name
            train_fn: Function(X_train, y_train) -> model
            eval_fn: Function(model, X_test, y_test) -> metrics dict
            time_col: Timestamp column
            min_auc: Minimum AUC per fold
            min_precision_at_10: Minimum precision@10% per fold
            
        Returns:
            WalkForwardResult with all fold results
        """
        folds = self.create_folds(df, time_col)
        
        if len(folds) < 2:
            return WalkForwardResult(
                folds=[],
                all_passed=False,
                avg_auc=0,
                avg_precision_at_10=0,
                min_auc=0,
                std_auc=0,
                rejection_reason="INSUFFICIENT_FOLDS"
            )
        
        df = df.sort_values(time_col).reset_index(drop=True)
        
        aucs = []
        precisions = []
        
        for fold in folds:
            logger.info(f"Fold {fold.fold_idx}: "
                       f"train {fold.train_start} to {fold.train_end} ({fold.train_samples}), "
                       f"test {fold.test_start} to {fold.test_end} ({fold.test_samples})")
            
            # Get train/test data
            train_mask = (df[time_col] >= fold.train_start) & (df[time_col] <= fold.train_end)
            test_mask = (df[time_col] >= fold.test_start) & (df[time_col] <= fold.test_end)
            
            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, target_col].values
            X_test = df.loc[test_mask, feature_cols].values
            y_test = df.loc[test_mask, target_col].values
            
            try:
                # Train model
                model = train_fn(X_train, y_train)
                
                # Evaluate
                metrics = eval_fn(model, X_test, y_test)
                fold.metrics = metrics
                
                auc = metrics.get('auc_roc', metrics.get('auc', 0))
                p_at_10 = metrics.get('precision_at_10', metrics.get('precision_at_10_long', 0))
                
                aucs.append(auc)
                precisions.append(p_at_10)
                
                # Check thresholds
                fold.passed = auc >= min_auc and p_at_10 >= min_precision_at_10
                
                logger.info(f"Fold {fold.fold_idx} results: AUC={auc:.4f}, P@10={p_at_10:.4f}, "
                           f"passed={fold.passed}")
                
            except Exception as e:
                logger.error(f"Fold {fold.fold_idx} failed: {e}")
                fold.passed = False
                fold.metrics = {'error': str(e)}
        
        # Aggregate results
        all_passed = all(f.passed for f in folds if f.passed is not None)
        
        if not aucs:
            return WalkForwardResult(
                folds=folds,
                all_passed=False,
                avg_auc=0,
                avg_precision_at_10=0,
                min_auc=0,
                std_auc=0,
                rejection_reason="ALL_FOLDS_FAILED"
            )
        
        result = WalkForwardResult(
            folds=folds,
            all_passed=all_passed,
            avg_auc=float(np.mean(aucs)),
            avg_precision_at_10=float(np.mean(precisions)) if precisions else 0,
            min_auc=float(np.min(aucs)),
            std_auc=float(np.std(aucs))
        )
        
        # Check for inconsistency (high variance = unstable model)
        if result.std_auc > 0.1:
            result.rejection_reason = f"HIGH_VARIANCE (std={result.std_auc:.3f})"
            result.all_passed = False
        
        # Check if any fold failed
        if not all_passed:
            failed_folds = [f.fold_idx for f in folds if f.passed is False]
            result.rejection_reason = f"FAILED_FOLDS: {failed_folds}"
        
        logger.info(f"Walk-forward validation: all_passed={result.all_passed}, "
                   f"avg_auc={result.avg_auc:.4f}, std={result.std_auc:.4f}")
        
        return result


def quick_walk_forward_check(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_class,
    n_folds: int = 3
) -> Tuple[bool, str]:
    """
    Quick walk-forward check for model stability.
    
    Returns:
        (passed, reason)
    """
    from sklearn.metrics import roc_auc_score
    
    validator = WalkForwardValidator(n_folds=n_folds)
    
    def train_fn(X, y):
        model = model_class()
        model.fit(X, y)
        return model
    
    def eval_fn(model, X, y):
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
        auc = roc_auc_score(y, y_prob)
        
        # Simple precision@10%
        top_n = max(1, int(len(y) * 0.1))
        top_indices = np.argsort(y_prob)[-top_n:]
        p_at_10 = np.mean(y[top_indices])
        
        return {'auc_roc': auc, 'precision_at_10': p_at_10}
    
    result = validator.validate(
        df, feature_cols, target_col, train_fn, eval_fn
    )
    
    if result.all_passed:
        return True, f"PASSED (avg_auc={result.avg_auc:.3f})"
    else:
        return False, result.rejection_reason or "FAILED"
