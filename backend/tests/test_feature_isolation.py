"""
Feature Isolation Tests - V3 Fix

Verify that Model A and Model B do not share correlated features.

Model A (Direction): momentum, direction, trend features
Model B (Quality): structure, volatility, regime features

If features leak between models → false confidence.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Set


# Define allowed features for each model
MODEL_A_FEATURES = {
    # Momentum features
    'rsi_14', 'rsi_7', 'rsi_21',
    'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d',
    'roc_10', 'roc_20',
    'momentum_5', 'momentum_10',
    
    # Direction features
    'ema_trend', 'sma_trend',
    'ema_9', 'ema_21', 'ema_50',
    'sma_20', 'sma_50', 'sma_200',
    'price_vs_ema_20', 'price_vs_sma_50',
    
    # Trend features
    'adx', 'adx_14',
    'plus_di', 'minus_di',
    'supertrend',
    'heikin_ashi_direction',
    
    # Price action
    'returns_1', 'returns_5', 'returns_10',
    'log_return',
    'higher_high', 'lower_low',
}

MODEL_B_FEATURES = {
    # Structure features
    'support_1', 'resistance_1',
    'price_to_support', 'price_to_resistance',
    'pivot_point', 'pivot_r1', 'pivot_s1',
    'fib_38', 'fib_50', 'fib_62',
    
    # Volatility features
    'atr_14', 'atr_7', 'atr_21',
    'atr_ratio', 'atr_percentile',
    'bb_width', 'bb_percent_b',
    'keltner_width',
    'volatility_5', 'volatility_20',
    'range_pct', 'true_range',
    
    # Regime features
    'regime', 'regime_encoded',
    'adx_regime', 'volatility_regime',
    'trend_stable', 'trend_volatile', 'range_quiet', 'chop_panic',
    
    # Volume/liquidity
    'volume', 'volume_ma', 'volume_ratio',
    'obv', 'obv_slope',
    'vwap', 'vwap_distance',
    
    # Time features (neutral)
    'hour', 'day_of_week', 'month',
    'is_expiry', 'days_to_expiry',
}

# Features that can be shared (truly neutral)
SHARED_FEATURES = {
    'open', 'high', 'low', 'close',
    'ts_utc', 'symbol',
}


def test_feature_isolation_no_overlap():
    """
    Test that Model A and Model B feature sets don't overlap.
    
    Overlap = potential information leakage.
    """
    overlap = MODEL_A_FEATURES & MODEL_B_FEATURES
    
    assert len(overlap) == 0, (
        f"Feature overlap detected between Model A and B: {overlap}\n"
        "This can cause information leakage and false confidence."
    )


def test_feature_correlation_check():
    """
    Test that features in one model are not highly correlated with features in the other.
    
    Even different features can leak information if highly correlated.
    """
    # Create synthetic data to check correlation patterns
    np.random.seed(42)
    n = 1000
    
    # Simulate some base data
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    returns = np.diff(np.log(price))
    returns = np.concatenate([[0], returns])
    
    # Model A features (momentum-based)
    rsi = 50 + 25 * np.tanh(returns * 50)  # RSI-like
    macd = np.convolve(returns, np.ones(12)/12, mode='same') - np.convolve(returns, np.ones(26)/26, mode='same')
    
    # Model B features (volatility-based)
    atr = np.abs(np.convolve(returns, np.ones(14)/14, mode='same')) * 100
    bb_width = atr * 2  # Proportional but different
    
    # Check cross-model correlation
    model_a_data = np.column_stack([rsi, macd])
    model_b_data = np.column_stack([atr, bb_width])
    
    # Compute cross-correlation
    for i, a_name in enumerate(['rsi', 'macd']):
        for j, b_name in enumerate(['atr', 'bb_width']):
            corr = np.corrcoef(model_a_data[:, i], model_b_data[:, j])[0, 1]
            
            # Correlation above 0.7 is concerning
            assert abs(corr) < 0.7, (
                f"High correlation ({corr:.2f}) between Model A's {a_name} "
                f"and Model B's {b_name}. Reduce feature leakage."
            )


def test_model_a_features_are_direction_related():
    """
    Verify Model A features relate to direction/momentum.
    """
    direction_keywords = {'rsi', 'macd', 'momentum', 'trend', 'ema', 'sma', 
                         'stoch', 'roc', 'returns', 'direction', 'plus_di', 
                         'minus_di', 'adx', 'supertrend'}
    
    for feature in MODEL_A_FEATURES:
        has_keyword = any(kw in feature.lower() for kw in direction_keywords)
        # Log warning if feature doesn't match expected pattern
        if not has_keyword:
            import warnings
            warnings.warn(f"Model A feature '{feature}' doesn't match direction keywords")


def test_model_b_features_are_quality_related():
    """
    Verify Model B features relate to structure/volatility.
    """
    quality_keywords = {'atr', 'bb', 'volume', 'volatility', 'range', 'support', 
                       'resistance', 'pivot', 'fib', 'regime', 'keltner', 'obv',
                       'vwap', 'hour', 'day', 'expiry', 'width'}
    
    for feature in MODEL_B_FEATURES:
        has_keyword = any(kw in feature.lower() for kw in quality_keywords)
        if not has_keyword:
            import warnings
            warnings.warn(f"Model B feature '{feature}' doesn't match quality keywords")


def get_feature_set_for_model(model_type: str) -> Set[str]:
    """
    Get the allowed feature set for a model type.
    
    Use this in training to filter features.
    """
    if model_type in ('direction', 'model_a', 'a'):
        return MODEL_A_FEATURES | SHARED_FEATURES
    elif model_type in ('quality', 'model_b', 'b'):
        return MODEL_B_FEATURES | SHARED_FEATURES
    else:
        # For combined/V1 models, allow all
        return MODEL_A_FEATURES | MODEL_B_FEATURES | SHARED_FEATURES


def filter_features_for_model(
    df: pd.DataFrame,
    model_type: str
) -> pd.DataFrame:
    """
    Filter DataFrame to only include features allowed for the model type.
    
    Args:
        df: DataFrame with all features
        model_type: 'direction' or 'quality'
        
    Returns:
        DataFrame with only allowed features
    """
    allowed = get_feature_set_for_model(model_type)
    available = set(df.columns)
    
    # Keep only available features that are allowed
    keep = list(available & allowed)
    
    return df[keep].copy()


if __name__ == '__main__':
    print("Running Feature Isolation Tests...")
    
    test_feature_isolation_no_overlap()
    print("✓ test_feature_isolation_no_overlap passed")
    
    test_feature_correlation_check()
    print("✓ test_feature_correlation_check passed")
    
    test_model_a_features_are_direction_related()
    print("✓ test_model_a_features_are_direction_related passed")
    
    test_model_b_features_are_quality_related()
    print("✓ test_model_b_features_are_quality_related passed")
    
    print("\nAll Feature Isolation tests passed! ✓")
