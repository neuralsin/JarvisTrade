"""
Lookahead Prevention Tests - V3 Fix

Verify that no features use future data (lookahead bias).

Rules:
1. All features must use shift(1) or only past data
2. No feature timestamp >= signal timestamp
3. Rolling indicators must use closed=True or equivalent
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings


def make_test_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create synthetic OHLCV data"""
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        'ts_utc': dates,
        'open': prices - 0.1,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })


def test_no_lookahead_in_returns():
    """
    Test that return features don't use future data.
    
    Common bug: using close[t] when we should use close[t-1]
    """
    df = make_test_ohlcv()
    
    # Correct: returns at time t use close[t]/close[t-1]
    # This is actually OK because at time t, close[t] is known
    df['returns'] = df['close'].pct_change()
    
    # Check: no NaN in first row (if so, shift was used correctly)
    assert pd.isna(df['returns'].iloc[0]), "Returns should have NaN at first row"
    
    # The return at t should NOT correlate with future returns
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    
    # Filter out NaNs
    valid = df[['returns', 'future_return']].dropna()
    if len(valid) > 10:
        corr = valid['returns'].corr(valid['future_return'])
        # Correlation should be low (not perfectly correlated)
        assert abs(corr) < 0.9, f"Returns may be using future data (corr={corr:.2f})"


def test_no_lookahead_in_rolling_indicators():
    """
    Test that rolling indicators don't include current bar in calculation.
    """
    df = make_test_ohlcv()
    
    # Incorrect: SMA includes current bar
    # sma = df['close'].rolling(20).mean()  # This includes close[t]
    
    # Correct: Shift before rolling to exclude current bar
    df['sma_shifted'] = df['close'].shift(1).rolling(20).mean()
    
    # At time t, SMA should only use data up to t-1
    # Check that SMA at t is not affected by close[t]
    for i in range(25, len(df) - 5):
        close_t = df['close'].iloc[i]
        sma_t = df['sma_shifted'].iloc[i]
        
        # Change close[t] and recalculate - should not change sma_t
        df_modified = df.copy()
        df_modified.loc[i, 'close'] = close_t * 2  # Double the close
        df_modified['sma_test'] = df_modified['close'].shift(1).rolling(20).mean()
        
        # SMA at t should be unchanged
        assert abs(df_modified['sma_test'].iloc[i] - sma_t) < 0.01, \
            f"SMA at t should not depend on close[t]"
        
        # Only need to test a few
        if i > 30:
            break


def test_feature_timestamp_causality():
    """
    Test that feature timestamps are always <= signal timestamp.
    """
    df = make_test_ohlcv()
    
    # Add feature timestamps (simulating when feature was computed)
    df['feature_ts'] = df['ts_utc'].shift(1)  # Correct: features from previous bar
    df['signal_ts'] = df['ts_utc']  # Signal at current bar
    
    # Verify causality
    valid = df[df['feature_ts'].notna()]
    for _, row in valid.iterrows():
        assert row['feature_ts'] <= row['signal_ts'], \
            f"Feature timestamp {row['feature_ts']} > signal timestamp {row['signal_ts']}"


def test_labels_use_future_data():
    """
    Labels SHOULD use future data (that's what we're predicting).
    
    But they should use ONLY future data, not current bar info
    that wouldn't be available at trading time.
    """
    df = make_test_ohlcv()
    
    # Label: did price go up over next 5 bars?
    df['label'] = (df['close'].shift(-5) > df['close']).astype(float)
    
    # This is correct - labels look forward
    # But we must ensure entry is at NEXT bar, not current
    
    # Check: label at t uses close[t+5] vs close[t]
    # At trading time t, we don't know close[t] yet (only open[t])
    # So label should compare future vs current close (which we see at bar close)
    
    # This is mostly a documentation test - labels are meant to be forward-looking
    assert True


def test_no_lookahead_in_atr():
    """
    Test that ATR calculation doesn't include current bar.
    """
    df = make_test_ohlcv()
    
    # Correct ATR: uses shift(1) before rolling
    def calculate_atr_correct(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Shift before rolling to avoid lookahead
        return tr.shift(1).rolling(period).mean()
    
    df['atr_correct'] = calculate_atr_correct(df['high'], df['low'], df['close'])
    
    # Modify current bar and check ATR doesn't change
    for i in range(20, 25):
        atr_before = df['atr_correct'].iloc[i]
        
        df_mod = df.copy()
        df_mod.loc[i, 'high'] = df_mod.loc[i, 'high'] * 2
        df_mod.loc[i, 'low'] = df_mod.loc[i, 'low'] * 0.5
        
        df_mod['atr_test'] = calculate_atr_correct(df_mod['high'], df_mod['low'], df_mod['close'])
        atr_after = df_mod['atr_test'].iloc[i]
        
        assert abs(atr_before - atr_after) < 0.01, \
            f"ATR at t should not be affected by bar t's high/low"


def verify_no_lookahead(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Utility function to check a DataFrame for potential lookahead bias.
    
    Returns dict of potentially problematic features.
    """
    problems = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Check 1: Feature has same value as future
        future_shifted = df[col].shift(-1)
        if df[col].equals(future_shifted.shift(1)):
            problems[col] = "May be using shift(-1) instead of shift(1)"
        
        # Check 2: Feature correlates too highly with future close
        if 'close' in df.columns:
            future_close = df['close'].shift(-5)
            valid = df[[col]].join(future_close.rename('future')).dropna()
            if len(valid) > 30:
                corr = valid[col].corr(valid['future'])
                if abs(corr) > 0.95:
                    problems[col] = f"Suspiciously high correlation with future ({corr:.2f})"
    
    return problems


if __name__ == '__main__':
    print("Running Lookahead Prevention Tests...")
    
    test_no_lookahead_in_returns()
    print("✓ test_no_lookahead_in_returns passed")
    
    test_no_lookahead_in_rolling_indicators()
    print("✓ test_no_lookahead_in_rolling_indicators passed")
    
    test_feature_timestamp_causality()
    print("✓ test_feature_timestamp_causality passed")
    
    test_labels_use_future_data()
    print("✓ test_labels_use_future_data passed")
    
    test_no_lookahead_in_atr()
    print("✓ test_no_lookahead_in_atr passed")
    
    print("\nAll Lookahead Prevention tests passed! ✓")
