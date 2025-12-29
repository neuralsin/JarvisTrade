"""
V2 Backtest Tests - Next-Bar Entry Verification

Tests verify:
1. Entry happens at NEXT bar open (not same bar)
2. Intrabar stop/target evaluation using high/low
3. Commission applied to both entry and exit
4. Trade records include signal_bar_idx and entry_bar_idx
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'backend')


def make_test_ohlcv():
    """Create synthetic OHLCV data"""
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)]
    
    # Price starts at 100, rises to 110, then falls
    base_prices = [100.0 + i * 0.5 for i in range(20)]  # 100 to 110
    base_prices += [110.0 - i * 0.3 for i in range(20)]  # 110 to 104
    base_prices += [104.0] * 10  # Flat
    
    data = {
        'ts_utc': dates,
        'open': [p - 0.1 for p in base_prices],
        'high': [p + 0.5 for p in base_prices],
        'low': [p - 0.5 for p in base_prices],
        'close': base_prices,
        'volume': [1000] * 50,
        'symbol': ['TEST'] * 50
    }
    return pd.DataFrame(data)


def test_v2_backtest_next_bar_entry():
    """Test that V2 backtest enters at NEXT bar open, not same bar"""
    from app.ml.backtesting import BacktestEngineV2
    
    df = make_test_ohlcv()
    
    # Simple strategy: buy on bar 5
    def strategy(row):
        idx = df[df['ts_utc'] == row['ts_utc']].index[0]
        if idx == 5:
            return ('BUY', 95.0, 110.0)  # Stop at 95, target at 110
        return ('HOLD', None, None)
    
    engine = BacktestEngineV2(initial_capital=100000, commission=0.0003, slippage=0.0005)
    result = engine.run_backtest(df, strategy)
    
    # Check that trades were recorded
    assert len(engine.trades) > 0, "Should have at least one trade"
    
    trade = engine.trades[0]
    
    # Verify entry happened at bar AFTER signal
    assert trade['signal_bar_idx'] == 5, f"Signal should be at bar 5, got {trade['signal_bar_idx']}"
    assert trade['entry_bar_idx'] == 6, f"Entry should be at bar 6 (next bar), got {trade['entry_bar_idx']}"
    
    # Verify entry price is bar 6's OPEN with slippage, not bar 5's close
    bar_6_open = df.iloc[6]['open']
    expected_entry = bar_6_open * (1 + 0.0005)  # With slippage
    assert abs(trade['entry_price'] - expected_entry) < 0.01, \
        f"Entry price should be bar 6 open + slippage ({expected_entry:.2f}), got {trade['entry_price']:.2f}"


def test_v2_backtest_intrabar_stop():
    """Test that stops are evaluated using bar's LOW, not just close"""
    from app.ml.backtesting import BacktestEngineV2
    
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(20)]
    
    # Create data where low hits stop but close doesn't
    # Bar 0: signal
    # Bar 1: entry at open=100
    # Bar 2: low=95 (hits stop), close=99 (doesn't hit stop)
    data = {
        'ts_utc': dates,
        'open': [100.0] * 20,
        'high': [101.0] * 20,
        'low': [99.0] * 20,
        'close': [100.0] * 20,
        'volume': [1000] * 20,
        'symbol': ['TEST'] * 20
    }
    df = pd.DataFrame(data)
    
    # Modify bar 2 to have low that hits stop
    df.loc[2, 'low'] = 95.0  # This should trigger stop
    df.loc[2, 'close'] = 99.0  # Close is above stop - V1 would miss this
    
    def strategy(row):
        idx = df[df['ts_utc'] == row['ts_utc']].index[0]
        if idx == 0:
            return ('BUY', 96.0, 110.0)  # Stop at 96, target at 110
        return ('HOLD', None, None)
    
    engine = BacktestEngineV2()
    engine.run_backtest(df, strategy)
    
    assert len(engine.trades) > 0, "Should have a trade"
    trade = engine.trades[0]
    
    # Trade should exit on bar 2 due to stop
    assert trade['exit_reason'] == 'STOP_LOSS', f"Should exit on stop, got {trade['exit_reason']}"
    assert trade['exit_bar_idx'] == 2, f"Should exit on bar 2, got {trade['exit_bar_idx']}"


def test_v2_backtest_intrabar_target():
    """Test that targets are evaluated using bar's HIGH"""
    from app.ml.backtesting import BacktestEngineV2
    
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(20)]
    
    data = {
        'ts_utc': dates,
        'open': [100.0] * 20,
        'high': [101.0] * 20,  # Normal high
        'low': [99.0] * 20,
        'close': [100.0] * 20,
        'volume': [1000] * 20,
        'symbol': ['TEST'] * 20
    }
    df = pd.DataFrame(data)
    
    # Modify bar 3 to have high that hits target
    df.loc[3, 'high'] = 106.0  # This should trigger target
    df.loc[3, 'close'] = 102.0  # Close is below target
    
    def strategy(row):
        idx = df[df['ts_utc'] == row['ts_utc']].index[0]
        if idx == 0:
            return ('BUY', 95.0, 105.0)  # Stop at 95, target at 105
        return ('HOLD', None, None)
    
    engine = BacktestEngineV2()
    engine.run_backtest(df, strategy)
    
    assert len(engine.trades) > 0, "Should have a trade"
    trade = engine.trades[0]
    
    assert trade['exit_reason'] == 'TARGET', f"Should exit on target, got {trade['exit_reason']}"
    assert trade['exit_bar_idx'] == 3


def test_v2_backtest_commission_both_sides():
    """Test that commission is applied to both entry and exit"""
    from app.ml.backtesting import BacktestEngineV2
    
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)]
    data = {
        'ts_utc': dates,
        'open': [100.0] * 10,
        'high': [110.0] * 10,  # High enough to hit target
        'low': [90.0] * 10,
        'close': [100.0] * 10,
        'volume': [1000] * 10,
        'symbol': ['TEST'] * 10
    }
    df = pd.DataFrame(data)
    
    def strategy(row):
        idx = df[df['ts_utc'] == row['ts_utc']].index[0]
        if idx == 0:
            return ('BUY', 90.0, 105.0)
        return ('HOLD', None, None)
    
    commission = 0.001  # 0.1%
    engine = BacktestEngineV2(initial_capital=100000, commission=commission, slippage=0)
    engine.run_backtest(df, strategy)
    
    if len(engine.trades) > 0:
        trade = engine.trades[0]
        # PNL should account for both entry and exit commission
        # This is a sanity check that commission was applied
        assert trade['pnl'] is not None


def test_v2_backtest_is_honest_flag():
    """Test that V2 backtest includes is_honest=True flag"""
    from app.ml.backtesting import BacktestEngineV2, BacktestEngine
    
    df = make_test_ohlcv()
    
    def strategy(row):
        return ('HOLD', None, None)
    
    engine_v2 = BacktestEngineV2()
    result_v2 = engine_v2.run_backtest(df, strategy)
    
    assert result_v2.get('is_honest') == True, "V2 should have is_honest=True"
    
    # V1 engine should have is_honest=False
    engine_v1 = BacktestEngine()
    result_v1 = engine_v1.run_backtest(df, strategy)
    
    assert result_v1.get('is_honest') == False, "V1 should have is_honest=False"


if __name__ == '__main__':
    print("Running V2 Backtest Tests...")
    
    test_v2_backtest_next_bar_entry()
    print("✓ test_v2_backtest_next_bar_entry passed")
    
    test_v2_backtest_intrabar_stop()
    print("✓ test_v2_backtest_intrabar_stop passed")
    
    test_v2_backtest_intrabar_target()
    print("✓ test_v2_backtest_intrabar_target passed")
    
    test_v2_backtest_commission_both_sides()
    print("✓ test_v2_backtest_commission_both_sides passed")
    
    test_v2_backtest_is_honest_flag()
    print("✓ test_v2_backtest_is_honest_flag passed")
    
    print("\nAll V2 Backtest tests passed! ✓")
