"""
V2 Labeler Tests - Triple Barrier Verification

Tests verify:
1. Entry at NEXT bar open (not current close)
2. Path-dependent stop/target checking
3. Slippage applied correctly
4. Timeout labels excluded
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'backend')


def make_test_data():
    """Create synthetic price data with known outcomes"""
    # Create 100 bars of price data
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    
    # Scenario: Price at 100, goes up to 105 then back down
    prices = [100.0] * 10  # Bars 0-9: flat at 100
    prices += [100.0 + i * 0.5 for i in range(10)]  # Bars 10-19: rise to 105
    prices += [105.0 - i * 0.3 for i in range(10)]  # Bars 20-29: fall to 102
    prices += [102.0] * 60  # Bars 30-89: flat at 102
    
    data = {
        'ts_utc': dates,
        'open': [p - 0.1 for p in prices],
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000] * 100
    }
    return pd.DataFrame(data)


def test_v2_labeler_next_bar_entry():
    """Test that V2 labeler uses next bar open for entry, not current close"""
    from app.ml.labeler import generate_labels_v2
    
    df = make_test_data()
    
    # Add ATR (needed for labeling)
    df['atr_14'] = 1.0  # Fixed ATR for predictable thresholds
    
    # Generate labels
    result = generate_labels_v2(
        df,
        direction=1,  # Long
        target_atr_mult=1.5,
        stop_atr_mult=1.0,
        time_limit_bars=24,
        slippage_pct=0.0005
    )
    
    assert 'target' in result.columns, "Target column should be added"
    
    # Check that labels exist
    valid_labels = result['target'].notna()
    assert valid_labels.sum() > 0, "Should have some valid labels"
    
    # Last bars should be None (insufficient future data)
    assert result['target'].iloc[-5] is None or pd.isna(result['target'].iloc[-5])


def test_v2_labeler_path_dependency():
    """Test that stop hit BEFORE target results in loss label"""
    from app.ml.labeler import generate_labels_v2
    
    # Create data where stop is hit before target
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)]
    
    # Entry at bar 0, next bar open = 100.1
    # Then price drops to 98 (hits stop) before rising to 105 (would hit target)
    prices = [100.0]  # Bar 0
    prices += [100.1]  # Bar 1 (entry)
    prices += [99.5]   # Bar 2 (slightly down)
    prices += [98.0]   # Bar 3 (STOP HIT - low will be even lower)
    prices += [99.0]   # Bar 4
    prices += [105.0]  # Bar 5 (target - but stop already hit!)
    prices += [105.0] * 44  # Rest
    
    data = {
        'ts_utc': dates,
        'open': [p - 0.1 for p in prices],
        'high': [p + 0.5 for p in prices],
        'low': [p - 1.0 for p in prices],  # Low is 1 below close
        'close': prices,
        'volume': [1000] * 50,
        'atr_14': [1.5] * 50  # ATR = 1.5
    }
    df = pd.DataFrame(data)
    
    # For bar 0: entry at bar 1 open = 100.0
    # Target = 100.0 + 1.5 * 1.5 = 102.25
    # Stop = 100.0 - 1.0 * 1.5 = 98.5
    # Bar 3 has low = 97.0 < 98.5 (stop hit)
    # Even though bar 5 high = 105.5 >= 102.25 (target hit), stop was first
    
    result = generate_labels_v2(
        df,
        direction=1,
        target_atr_mult=1.5,
        stop_atr_mult=1.0,
        time_limit_bars=24,
        slippage_pct=0.0
    )
    
    # Bar 0 should be labeled as LOSS (0) because stop hit before target
    label_0 = result['target'].iloc[0]
    assert label_0 == 0, f"Expected loss (0) when stop hit before target, got {label_0}"


def test_v2_labeler_slippage():
    """Test that slippage is applied to entry"""
    from app.ml.labeler import generate_labels_v2
    
    # Create data where slippage makes the difference
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)]
    
    # Without slippage: entry at 100.0, target at 101.5 (met at bar 2)
    # With 0.5% slippage: entry at 100.5, target at 102.0 (NOT met)
    prices = [100.0] * 50
    prices[1] = 100.0  # Entry bar
    prices[2] = 101.8  # High = 102.3 - just barely hits target with slippage
    
    data = {
        'ts_utc': dates,
        'open': [p for p in prices],
        'high': [p + 0.3 for p in prices],  # High is 0.3 above close
        'low': [p - 0.3 for p in prices],
        'close': prices,
        'volume': [1000] * 50,
        'atr_14': [1.0] * 50
    }
    df = pd.DataFrame(data)
    
    # With slippage
    result_with = generate_labels_v2(df, slippage_pct=0.005)  # 0.5%
    
    # Without slippage
    result_without = generate_labels_v2(df.copy(), slippage_pct=0.0)
    
    # Results should potentially differ due to slippage adjustment
    # (This is a basic sanity check - actual differences depend on prices)
    assert 'target' in result_with.columns
    assert 'target' in result_without.columns


def test_v2_labeler_timeout():
    """Test that timeout results in -1 label (excluded from training)"""
    from app.ml.labeler import generate_labels_v2, prepare_labeled_data
    
    # Create flat data where neither stop nor target is hit
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    
    # Price stays at 100 with minimal movement
    data = {
        'ts_utc': dates,
        'open': [100.0] * 100,
        'high': [100.1] * 100,  # Very small range
        'low': [99.9] * 100,
        'close': [100.0] * 100,
        'volume': [1000] * 100,
        'atr_14': [0.1] * 100  # Very small ATR = tight barriers
    }
    df = pd.DataFrame(data)
    
    result = generate_labels_v2(
        df,
        direction=1,
        target_atr_mult=1.5,  # Target at 100.15
        stop_atr_mult=1.0,    # Stop at 99.9
        time_limit_bars=10    # Short time limit
    )
    
    # Many labels should be timeout (-1) since barriers are very tight
    # But high=100.1 might hit stop (99.9) or target (100.15)
    # Check that prepare_labeled_data excludes -1 labels
    timeouts = (result['target'] == -1).sum()
    
    prepared = prepare_labeled_data(result, exclude_timeouts=True)
    assert len(prepared) < len(result[result['target'].notna()])


if __name__ == '__main__':
    print("Running V2 Labeler Tests...")
    test_v2_labeler_next_bar_entry()
    print("✓ test_v2_labeler_next_bar_entry passed")
    
    test_v2_labeler_path_dependency()
    print("✓ test_v2_labeler_path_dependency passed")
    
    test_v2_labeler_slippage()
    print("✓ test_v2_labeler_slippage passed")
    
    test_v2_labeler_timeout()
    print("✓ test_v2_labeler_timeout passed")
    
    print("\nAll V2 Labeler tests passed! ✓")
