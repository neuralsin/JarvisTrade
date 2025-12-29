"""
V2 Kill Conditions - Model Rejection Logic

Centralized logic for determining when a model should NOT be activated.
Models must FAIL LOUDLY with clear reasons.

Kill Conditions:
1. AUC < 0.52 (random model)
2. Precision@10% < 0.40 (worse than random)
3. Samples < 200 (insufficient data)
4. Prob variance < 0.01 (collapsed/dead model)
5. Prediction all same class (broken model)
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# KILL THRESHOLDS - Not negotiable
# =============================================================================

KILL_THRESHOLDS = {
    'min_auc': 0.52,                    # Below this = random
    'min_precision_at_10': 0.40,        # Below this = worse than random
    'min_samples': 200,                 # Below this = insufficient data
    'min_prob_variance': 0.01,          # Below this = dead model
    'max_class_imbalance': 0.95,        # Above this = broken labels
    'min_classes_predicted': 2,         # Must predict at least 2 classes
    
    # Warning thresholds (not kills, but logged)
    'warn_auc': 0.55,
    'warn_precision_at_10': 0.50,
    'warn_samples': 500,
}


def should_reject_model(
    metrics: Dict,
    n_samples: int,
    model_type: str = 'xgboost'
) -> Tuple[bool, str, List[str]]:
    """
    Determine if a model should be rejected.
    
    Args:
        metrics: Dict with auc_roc, precision_at_10, prob_std, etc.
        n_samples: Number of training samples
        model_type: 'xgboost', 'transformer', or 'binary'
    
    Returns:
        Tuple of (should_reject, reason, warnings)
    """
    warnings = []
    kill_reasons = []
    
    # ==========================================================================
    # KILL CONDITION 1: Insufficient samples
    # ==========================================================================
    if n_samples < KILL_THRESHOLDS['min_samples']:
        kill_reasons.append(f"Insufficient samples: {n_samples} < {KILL_THRESHOLDS['min_samples']}")
    elif n_samples < KILL_THRESHOLDS['warn_samples']:
        warnings.append(f"Low samples: {n_samples} (recommended >= {KILL_THRESHOLDS['warn_samples']})")
    
    # ==========================================================================
    # KILL CONDITION 2: Dead model (collapsed predictions)
    # ==========================================================================
    prob_std = metrics.get('prob_std', 0)
    if prob_std < KILL_THRESHOLDS['min_prob_variance']:
        kill_reasons.append(f"Dead model: prob_std={prob_std:.6f} < {KILL_THRESHOLDS['min_prob_variance']}")
    
    is_dead = metrics.get('is_dead', False)
    if is_dead:
        kill_reasons.append(f"Model flagged as dead: {metrics.get('rejection_reason', 'unknown')}")
    
    # ==========================================================================
    # KILL CONDITION 3: AUC too low (random model)
    # ==========================================================================
    auc = metrics.get('auc_roc', 0.5)
    if auc == 0.5:
        # Also check for direction model specific AUC fields
        auc_long = metrics.get('auc_long', 0.5)
        auc_short = metrics.get('auc_short', 0.5)
        auc = max(auc_long, auc_short)
    
    if auc < KILL_THRESHOLDS['min_auc']:
        kill_reasons.append(f"Random model: AUC={auc:.4f} < {KILL_THRESHOLDS['min_auc']}")
    elif auc < KILL_THRESHOLDS['warn_auc']:
        warnings.append(f"Weak AUC: {auc:.4f} (recommended >= {KILL_THRESHOLDS['warn_auc']})")
    
    # ==========================================================================
    # KILL CONDITION 4: Precision@10% too low
    # ==========================================================================
    p_at_10 = metrics.get('precision_at_10', metrics.get('precision_at_10_long', 0))
    if p_at_10 == 0:
        p_at_10 = metrics.get('precision_at_10_short', 0)
    
    if p_at_10 < KILL_THRESHOLDS['min_precision_at_10']:
        kill_reasons.append(f"Low precision: P@10%={p_at_10:.4f} < {KILL_THRESHOLDS['min_precision_at_10']}")
    elif p_at_10 < KILL_THRESHOLDS['warn_precision_at_10']:
        warnings.append(f"Marginal precision: P@10%={p_at_10:.4f} (recommended >= {KILL_THRESHOLDS['warn_precision_at_10']})")
    
    # ==========================================================================
    # KILL CONDITION 5: Model rejected by inversion check
    # ==========================================================================
    model_state = metrics.get('model_state', None)
    if model_state is not None:
        from app.ml.model_inverter import ModelState
        if model_state == ModelState.REJECT:
            kill_reasons.append(f"Model state REJECTED by inversion check")
    
    # ==========================================================================
    # Transformer-specific checks
    # ==========================================================================
    if model_type == 'transformer':
        if n_samples < 10000:
            kill_reasons.append(f"Transformer requires >= 10000 samples, got {n_samples}")
    
    # Build final result
    should_reject = len(kill_reasons) > 0
    
    if should_reject:
        reason = " | ".join(kill_reasons)
        logger.error(f"🚫 MODEL REJECTED: {reason}")
    else:
        reason = "Model passed all checks"
        if warnings:
            logger.warning(f"⚠️ Model warnings: {' | '.join(warnings)}")
    
    return should_reject, reason, warnings


def get_activation_status(
    metrics: Dict,
    n_samples: int,
    model_type: str = 'xgboost'
) -> Dict:
    """
    Get detailed activation status for a model.
    
    Returns:
        Dict with is_activated, reason, warnings, and individual check results
    """
    should_reject, reason, warnings = should_reject_model(metrics, n_samples, model_type)
    
    # Extract individual metrics for detailed display
    auc = metrics.get('auc_roc', metrics.get('auc_long', 0.5))
    p_at_10 = metrics.get('precision_at_10', metrics.get('precision_at_10_long', 0))
    prob_std = metrics.get('prob_std', 0)
    
    return {
        'is_activated': not should_reject,
        'reason': reason,
        'warnings': warnings,
        'checks': {
            'samples': {
                'value': n_samples,
                'threshold': KILL_THRESHOLDS['min_samples'],
                'passed': n_samples >= KILL_THRESHOLDS['min_samples']
            },
            'auc': {
                'value': auc,
                'threshold': KILL_THRESHOLDS['min_auc'],
                'passed': auc >= KILL_THRESHOLDS['min_auc']
            },
            'precision_at_10': {
                'value': p_at_10,
                'threshold': KILL_THRESHOLDS['min_precision_at_10'],
                'passed': p_at_10 >= KILL_THRESHOLDS['min_precision_at_10']
            },
            'prob_variance': {
                'value': prob_std,
                'threshold': KILL_THRESHOLDS['min_prob_variance'],
                'passed': prob_std >= KILL_THRESHOLDS['min_prob_variance']
            }
        }
    }


def log_kill_conditions():
    """Log current kill condition thresholds"""
    logger.info("📋 Kill Condition Thresholds:")
    for key, value in KILL_THRESHOLDS.items():
        if key.startswith('min_'):
            logger.info(f"   {key}: >= {value}")
        elif key.startswith('max_'):
            logger.info(f"   {key}: <= {value}")
        elif key.startswith('warn_'):
            logger.info(f"   {key}: >= {value} (warning only)")


def validate_labels_distribution(
    y: np.ndarray,
    min_class_ratio: float = 0.05
) -> Tuple[bool, str]:
    """
    Validate that labels have reasonable distribution.
    
    Returns:
        Tuple of (is_valid, reason)
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    if len(unique) < 2:
        return False, f"Only one class present: {unique[0]}"
    
    ratios = counts / total
    
    for cls, ratio in zip(unique, ratios):
        if ratio < min_class_ratio:
            return False, f"Class {cls} has only {ratio*100:.1f}% of samples (minimum: {min_class_ratio*100:.0f}%)"
        if ratio > KILL_THRESHOLDS['max_class_imbalance']:
            return False, f"Class {cls} dominates with {ratio*100:.1f}% (maximum: {KILL_THRESHOLDS['max_class_imbalance']*100:.0f}%)"
    
    return True, "Label distribution is acceptable"
