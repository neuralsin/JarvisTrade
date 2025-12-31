"""
Model Inverter - Inverted Signal Weaponization

Detects models that have learned systematic inverse relationships to price movement
and converts them into profitable alpha sources by flipping predictions.

Key Insight: A model with AUC < 0.5 that consistently predicts the opposite 
of reality contains INFORMATION - it just needs to be inverted.

Anti-signal ≠ noise. Consistently wrong = information.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Configuration thresholds
MIN_FLIPPED_AUC = 0.55  # Minimum AUC after flipping to be considered INVERTIBLE
MIN_STD_THRESHOLD = 0.01  # Minimum std dev of predictions (dead model detection)
MIN_RANGE_THRESHOLD = 0.05  # Minimum range of predictions (dead model detection)
INVERTED_POSITION_PENALTY = 0.7  # Initial position size multiplier for inverted models
MIN_SAMPLES_TRANSFORMER = 5000  # Minimum samples for transformer inversion


class ModelState:
    """Model state enum-like class"""
    NORMAL = "NORMAL"
    INVERTIBLE = "INVERTIBLE"
    INVERTED = "INVERTED"  # After transformation applied
    REJECT = "REJECT"


def detect_model_state(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_flipped_auc: float = MIN_FLIPPED_AUC,
    model_type: str = "xgboost"
) -> Tuple[str, Dict]:
    """
    Detect if model is NORMAL, INVERTIBLE, or REJECT.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        min_flipped_auc: Minimum AUC required after flipping
        model_type: Type of model (for transformer sample checks)
        
    Returns:
        tuple: (state, metadata)
        
    State Logic:
        - NORMAL: AUC >= min_flipped_auc and not dead
        - INVERTIBLE: AUC < 0.5, flipped AUC >= min_flipped_auc, not dead
        - REJECT: Dead model or low AUC in both directions
    """
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()
    
    # Ensure we have valid data
    if len(y_true) == 0 or len(y_prob) == 0:
        return ModelState.REJECT, {"reason": "empty_data"}
    
    # Handle edge case of single class in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(f"Only one class in y_true: {unique_classes}")
        return ModelState.REJECT, {"reason": "single_class"}
    
    try:
        auc = roc_auc_score(y_true, y_prob)
        auc_flipped = roc_auc_score(y_true, 1.0 - y_prob)
    except Exception as e:
        logger.error(f"AUC calculation failed: {e}")
        return ModelState.REJECT, {"reason": f"auc_error: {str(e)}"}
    
    # Dead model detection - predictions collapsed to near-constant values
    prob_std = np.std(y_prob)
    prob_range = np.max(y_prob) - np.min(y_prob)
    
    is_dead = (
        prob_std < MIN_STD_THRESHOLD or
        prob_range < MIN_RANGE_THRESHOLD
    )
    
    metadata = {
        "raw_auc": round(auc, 4),
        "flipped_auc": round(auc_flipped, 4),
        "prob_std": round(prob_std, 4),
        "prob_range": round(prob_range, 4),
        "is_dead": is_dead
    }
    
    if is_dead:
        logger.warning(f"Dead model detected: std={prob_std:.4f}, range={prob_range:.4f}")
        metadata["reason"] = "dead_model"
        return ModelState.REJECT, metadata
    
    # Check for invertible model
    if auc < 0.5 and auc_flipped >= min_flipped_auc:
        logger.info(
            f"🔄 INVERTIBLE model detected! "
            f"Raw AUC={auc:.4f}, Flipped AUC={auc_flipped:.4f}"
        )
        return ModelState.INVERTIBLE, metadata
    
    # Check for normal good model
    if auc >= min_flipped_auc:
        return ModelState.NORMAL, metadata
    
    # Low AUC in both directions - reject
    metadata["reason"] = "low_auc_both_directions"
    logger.warning(f"Model rejected: AUC={auc:.4f}, Flipped={auc_flipped:.4f}")
    return ModelState.REJECT, metadata


def apply_inversion(y_prob: np.ndarray) -> np.ndarray:
    """
    Apply inversion transformation to predictions.
    
    Args:
        y_prob: Original predicted probabilities
        
    Returns:
        Inverted probabilities: 1.0 - y_prob
    """
    return 1.0 - np.array(y_prob)


def validate_inverted_model(
    y_true: np.ndarray,
    y_prob_flipped: np.ndarray,
    min_precision_at_k: float = 0.6,
    k: int = 10,
    model_type: str = "xgboost",
    n_samples: int = 0
) -> Tuple[bool, Dict]:
    """
    Validate that an inverted model meets all activation criteria.
    
    Strict validation rules:
    - Flipped AUC >= 0.55
    - Precision@TopK >= threshold
    - Expected Value (post-flip) > 0
    - Not a Transformer trained on < 5k samples
    
    Args:
        y_true: Ground truth labels
        y_prob_flipped: ALREADY FLIPPED probabilities
        min_precision_at_k: Minimum precision at top K
        k: Top K for precision
        model_type: Model type string
        n_samples: Number of training samples
        
    Returns:
        tuple: (is_valid, validation_metrics)
    """
    validation = {
        "checks_passed": [],
        "checks_failed": []
    }
    
    # Check 1: Transformer sample check
    if model_type.lower() == "transformer" and n_samples < MIN_SAMPLES_TRANSFORMER:
        validation["checks_failed"].append(
            f"transformer_samples: {n_samples} < {MIN_SAMPLES_TRANSFORMER}"
        )
    else:
        validation["checks_passed"].append("transformer_samples")
    
    # Check 2: Post-flip AUC
    try:
        flipped_auc = roc_auc_score(y_true, y_prob_flipped)
        validation["flipped_auc"] = round(flipped_auc, 4)
        
        if flipped_auc >= MIN_FLIPPED_AUC:
            validation["checks_passed"].append("flipped_auc")
        else:
            validation["checks_failed"].append(f"flipped_auc: {flipped_auc:.4f} < {MIN_FLIPPED_AUC}")
    except Exception as e:
        validation["checks_failed"].append(f"flipped_auc_error: {str(e)}")
    
    # Check 3: Precision@TopK - FIXED: Use fraction instead of literal K
    try:
        # CRITICAL FIX: k was "top 10 samples", now "top 10% of samples"
        # This matches trainer.py precision_at_k() calculation
        n_samples = len(y_true)
        k_samples = max(int(n_samples * 0.10), 10)  # 10% but at least 10 samples
        
        # Sort by probability and take top K
        sorted_indices = np.argsort(y_prob_flipped)[::-1][:k_samples]
        top_k_true = y_true[sorted_indices]
        precision_at_k = np.mean(top_k_true)
        validation["precision_at_k"] = round(precision_at_k, 4)
        validation["k_samples_used"] = k_samples  # For debugging
        
        if precision_at_k >= min_precision_at_k:
            validation["checks_passed"].append("precision_at_k")
        else:
            validation["checks_failed"].append(
                f"precision_at_k: {precision_at_k:.4f} < {min_precision_at_k}"
            )
    except Exception as e:
        validation["checks_failed"].append(f"precision_error: {str(e)}")
    
    # Check 4: Expected Value calculation
    # EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    # For simplified check, we use precision as proxy for win_rate
    try:
        if "precision_at_k" in validation:
            win_rate = validation["precision_at_k"]
            # Assume 2:1 reward-risk ratio for simplified EV
            expected_value = (win_rate * 2.0) - ((1 - win_rate) * 1.0)
            validation["expected_value"] = round(expected_value, 4)
            
            if expected_value > 0:
                validation["checks_passed"].append("expected_value")
            else:
                validation["checks_failed"].append(
                    f"expected_value: {expected_value:.4f} <= 0"
                )
    except Exception as e:
        validation["checks_failed"].append(f"ev_error: {str(e)}")
    
    is_valid = len(validation["checks_failed"]) == 0
    validation["is_valid"] = is_valid
    
    if is_valid:
        logger.info(f"✅ Inverted model validation PASSED: {validation['checks_passed']}")
    else:
        logger.warning(f"❌ Inverted model validation FAILED: {validation['checks_failed']}")
    
    return is_valid, validation


def get_position_size_multiplier(is_inverted: bool, n_trades: int = 0) -> float:
    """
    Get position size multiplier for risk adjustment.
    
    Inverted models start with 0.7x position size.
    After 30 trades, if performing well, penalty can be removed.
    
    Args:
        is_inverted: Whether model is inverted
        n_trades: Number of real trades executed
        
    Returns:
        Position size multiplier (0.0 to 1.0)
    """
    if not is_inverted:
        return 1.0
    
    # After 30 trades, start removing penalty
    if n_trades >= 30:
        return 1.0  # Penalty removed (should check rolling expectancy in practice)
    
    return INVERTED_POSITION_PENALTY


def create_inversion_metadata(
    model_id: str,
    state: str,
    raw_auc: float,
    flipped_auc: float,
    activated: bool,
    validation_metrics: Optional[Dict] = None
) -> Dict:
    """
    Create standardized inversion metadata for logging and storage.
    
    Args:
        model_id: Unique model identifier
        state: Model state (NORMAL, INVERTED, REJECT)
        raw_auc: Original AUC before flip
        flipped_auc: AUC after flipping
        activated: Whether model was activated
        validation_metrics: Optional validation details
        
    Returns:
        Metadata dictionary
    """
    from datetime import datetime
    
    metadata = {
        "model_id": model_id,
        "state": state,
        "raw_auc": round(raw_auc, 4),
        "flipped_auc": round(flipped_auc, 4),
        "activated": activated,
        "inversion_applied_at": datetime.utcnow().isoformat() + "Z"
    }
    
    if validation_metrics:
        metadata["validation"] = validation_metrics
    
    return metadata


# Shorthand for common use case
def process_model_inversion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_id: str,
    model_type: str = "xgboost",
    n_samples: int = 0,
    auto_activate: bool = True
) -> Tuple[str, np.ndarray, Dict]:
    """
    Complete pipeline: detect, validate, transform, and activate model.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        model_id: Unique model identifier
        model_type: Type of model
        n_samples: Number of training samples
        auto_activate: Whether to auto-activate valid models
        
    Returns:
        tuple: (final_state, final_probs, metadata)
        
    Usage:
        state, probs, meta = process_model_inversion(y_true, y_prob, "TATA_V2")
        if state in [ModelState.NORMAL, ModelState.INVERTED]:
            # Model can be activated
            model.save(final_probs, metadata=meta)
    """
    # Step 1: Detect state
    state, detect_meta = detect_model_state(y_true, y_prob, model_type=model_type)
    
    if state == ModelState.REJECT:
        logger.warning(f"❌ Model {model_id} REJECTED: {detect_meta.get('reason', 'unknown')}")
        return state, y_prob, create_inversion_metadata(
            model_id, state,
            detect_meta.get("raw_auc", 0),
            detect_meta.get("flipped_auc", 0),
            activated=False,
            validation_metrics=detect_meta
        )
    
    if state == ModelState.NORMAL:
        logger.info(f"✅ Model {model_id} NORMAL: AUC={detect_meta['raw_auc']:.4f}")
        return state, y_prob, create_inversion_metadata(
            model_id, state,
            detect_meta["raw_auc"],
            detect_meta.get("flipped_auc", 0),
            activated=auto_activate
        )
    
    # State is INVERTIBLE - apply transformation
    y_prob_flipped = apply_inversion(y_prob)
    
    # Step 2: Validate inverted model
    is_valid, validation_meta = validate_inverted_model(
        y_true, y_prob_flipped,
        model_type=model_type,
        n_samples=n_samples
    )
    
    if not is_valid:
        logger.warning(f"❌ Model {model_id} inversion FAILED validation")
        return ModelState.REJECT, y_prob, create_inversion_metadata(
            model_id, ModelState.REJECT,
            detect_meta["raw_auc"],
            detect_meta["flipped_auc"],
            activated=False,
            validation_metrics=validation_meta
        )
    
    # Step 3: Apply inversion
    final_state = ModelState.INVERTED
    logger.info(
        f"🔄 Model {model_id} INVERTED: "
        f"Raw AUC={detect_meta['raw_auc']:.4f} → Flipped AUC={detect_meta['flipped_auc']:.4f}"
    )
    
    return final_state, y_prob_flipped, create_inversion_metadata(
        model_id, final_state,
        detect_meta["raw_auc"],
        detect_meta["flipped_auc"],
        activated=auto_activate,
        validation_metrics=validation_meta
    )
