"""
Probability Calibration Module - V3 Fix

XGBoost probabilities are NOT calibrated by default.
A predicted 0.7 does NOT mean 70% win chance.

This module provides:
- Platt scaling (sigmoid calibration)
- Isotonic regression calibration
- Per-regime calibration
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Optional, Dict, Tuple, Literal
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate XGBoost/ML model probabilities to true probabilities.
    
    After calibration, a 0.7 probability should mean ~70% actual win rate.
    """
    
    def __init__(
        self,
        method: Literal['platt', 'isotonic'] = 'isotonic'
    ):
        """
        Initialize calibrator.
        
        Args:
            method: 'platt' for sigmoid/Platt scaling, 'isotonic' for isotonic regression
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.calibration_stats = {}
    
    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> 'ProbabilityCalibrator':
        """
        Fit the calibrator on validation data.
        
        Args:
            y_true: True labels (0/1)
            y_prob: Predicted probabilities (0-1)
            
        Returns:
            self for chaining
        """
        # Validate inputs
        if len(y_true) < 100:
            logger.warning(f"⚠️ Low sample size for calibration: {len(y_true)}. "
                          "Calibration may be unreliable.")
        
        # Store original stats
        self.calibration_stats['n_samples'] = len(y_true)
        self.calibration_stats['original_mean_prob'] = float(np.mean(y_prob))
        self.calibration_stats['actual_positive_rate'] = float(np.mean(y_true))
        
        if self.method == 'platt':
            # Platt scaling: fit logistic regression on probabilities
            self.calibrator = LogisticRegression(solver='lbfgs')
            # Reshape for sklearn
            X = y_prob.reshape(-1, 1)
            self.calibrator.fit(X, y_true)
        
        elif self.method == 'isotonic':
            # Isotonic regression: monotonic function fitting
            self.calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds='clip'
            )
            self.calibrator.fit(y_prob, y_true)
        
        self.is_fitted = True
        
        # Calculate calibrated stats
        calibrated = self.calibrate(y_prob)
        self.calibration_stats['calibrated_mean_prob'] = float(np.mean(calibrated))
        
        # Log calibration shift
        shift = self.calibration_stats['calibrated_mean_prob'] - self.calibration_stats['original_mean_prob']
        logger.info(f"✅ Calibration complete ({self.method}): "
                   f"prob shift = {shift:+.3f}, "
                   f"actual positive rate = {self.calibration_stats['actual_positive_rate']:.3f}")
        
        return self
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.
        
        Args:
            y_prob: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("⚠️ Calibrator not fitted, returning raw probabilities")
            return y_prob
        
        if self.method == 'platt':
            X = y_prob.reshape(-1, 1)
            return self.calibrator.predict_proba(X)[:, 1]
        
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_prob)
        
        return y_prob
    
    def save(self, path: str) -> None:
        """Save calibrator to file"""
        data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'calibration_stats': self.calibration_stats
        }
        joblib.dump(data, path)
        logger.info(f"💾 Saved calibrator to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        """Load calibrator from file"""
        data = joblib.load(path)
        instance = cls(method=data['method'])
        instance.calibrator = data['calibrator']
        instance.is_fitted = data['is_fitted']
        instance.calibration_stats = data['calibration_stats']
        logger.info(f"📂 Loaded calibrator from {path}")
        return instance


class RegimeAwareCalibrator:
    """
    Calibrate probabilities separately for each regime.
    
    Different market regimes may have different probability distributions.
    """
    
    def __init__(self, method: Literal['platt', 'isotonic'] = 'isotonic'):
        self.method = method
        self.calibrators: Dict[str, ProbabilityCalibrator] = {}
        self.default_calibrator: Optional[ProbabilityCalibrator] = None
    
    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        regimes: np.ndarray
    ) -> 'RegimeAwareCalibrator':
        """
        Fit calibrators for each regime.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            regimes: Regime labels for each sample
        """
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            mask = regimes == regime
            if np.sum(mask) < 50:
                logger.warning(f"⚠️ Regime '{regime}' has only {np.sum(mask)} samples, skipping")
                continue
            
            calibrator = ProbabilityCalibrator(method=self.method)
            calibrator.fit(y_true[mask], y_prob[mask])
            self.calibrators[str(regime)] = calibrator
        
        # Fit a default calibrator on all data
        self.default_calibrator = ProbabilityCalibrator(method=self.method)
        self.default_calibrator.fit(y_true, y_prob)
        
        logger.info(f"✅ Fitted {len(self.calibrators)} regime-specific calibrators")
        
        return self
    
    def calibrate(
        self,
        y_prob: np.ndarray,
        regimes: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate probabilities using regime-specific calibrators.
        """
        result = np.zeros_like(y_prob)
        
        for i, (prob, regime) in enumerate(zip(y_prob, regimes)):
            regime_str = str(regime)
            if regime_str in self.calibrators:
                result[i] = self.calibrators[regime_str].calibrate(np.array([prob]))[0]
            elif self.default_calibrator:
                result[i] = self.default_calibrator.calibrate(np.array([prob]))[0]
            else:
                result[i] = prob
        
        return result
    
    def save(self, path: str) -> None:
        """Save all calibrators"""
        data = {
            'method': self.method,
            'calibrators': self.calibrators,
            'default_calibrator': self.default_calibrator
        }
        joblib.dump(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'RegimeAwareCalibrator':
        """Load calibrators"""
        data = joblib.load(path)
        instance = cls(method=data['method'])
        instance.calibrators = data['calibrators']
        instance.default_calibrator = data['default_calibrator']
        return instance


def calibrate_model_probabilities(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: Literal['platt', 'isotonic'] = 'isotonic'
) -> Tuple[ProbabilityCalibrator, Dict]:
    """
    Convenience function to calibrate a trained model.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation labels
        method: Calibration method
        
    Returns:
        (calibrator, stats_dict)
    """
    # Get raw probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_val)
        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]  # Binary: take positive class
    else:
        raise ValueError("Model must have predict_proba method")
    
    # Fit calibrator
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_val, y_prob)
    
    return calibrator, calibrator.calibration_stats


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Generate data for a reliability diagram (calibration curve).
    
    Returns:
        Dict with bin_means, fraction_positives, and counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    fraction_positives = []
    counts = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(y_prob[mask]))
            fraction_positives.append(np.mean(y_true[mask]))
            counts.append(int(np.sum(mask)))
        else:
            bin_means.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            fraction_positives.append(0)
            counts.append(0)
    
    # Calculate calibration error
    ece = 0  # Expected Calibration Error
    total = sum(counts)
    for mean, frac, count in zip(bin_means, fraction_positives, counts):
        if count > 0:
            ece += (count / total) * abs(frac - mean)
    
    return {
        'bin_means': bin_means,
        'fraction_positives': fraction_positives,
        'counts': counts,
        'expected_calibration_error': ece
    }
