"""
V2 Labeling System - Triple Barrier with Path Dependency

CRITICAL CHANGES FROM V1:
1. Entry at NEXT bar open (not current close)
2. Slippage penalty applied to entry
3. Path-dependent: checks high/low within each bar
4. Timeouts excluded from training (-1 label)
5. One label definition per dataset (V2 only)

Labels:
    1 = WIN: Target hit before stop
    0 = LOSS: Stop hit before target
   -1 = TIMEOUT: Neither hit within time limit (excluded from training)
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def generate_labels(
    df: pd.DataFrame,
    target_pct: float = 0.015,    # 1.5% target
    stop_pct: float = 0.008,      # 0.8% stop
    window: int = 20,
    use_binary: bool = True       # FIX 2: FORCE BINARY
) -> pd.DataFrame:
    """
    DEPRECATED V1 Labeling - Use generate_labels_v2 for new training.
    
    This function is kept for backward compatibility but logs a warning.
    New training should use generate_labels_v2() with triple barrier.
    """
    logger.warning("⚠️ DEPRECATED: Using V1 point-to-point labeling. "
                   "This may produce unreliable labels. Use generate_labels_v2() instead.")
    
    df = df.copy().sort_values('ts_utc').reset_index(drop=True)
    
    if 'ts_utc' in df.columns:
        assert df['ts_utc'].is_monotonic_increasing, "Data must be sorted chronologically"
    
    targets = []
    
    for idx in range(len(df)):
        if idx + window >= len(df):
            targets.append(None)
            continue
        
        current_close = df.loc[idx, 'close']
        
        target_price_up = current_close * (1 + target_pct)
        stop_price_down = current_close * (1 - stop_pct)
        target_price_down = current_close * (1 - target_pct)
        stop_price_up = current_close * (1 + stop_pct)
        
        future_slice = df.iloc[idx+1:idx+1+window]
        
        # Check LONG trade
        target_hit_up = (future_slice['high'] >= target_price_up).any()
        if target_hit_up:
            target_hit_idx = future_slice[future_slice['high'] >= target_price_up].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            stop_hit = (pre_target['low'] < stop_price_down).any()
            
            if not stop_hit:
                targets.append(1)
                continue
        
        # Check SHORT trade
        target_hit_down = (future_slice['low'] <= target_price_down).any()
        if target_hit_down:
            target_hit_idx = future_slice[future_slice['low'] <= target_price_down].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            stop_hit = (pre_target['high'] > stop_price_up).any()
            
            if not stop_hit:
                targets.append(1)
                continue
        
        targets.append(0)
    
    df['target'] = targets
    
    valid_count = df['target'].notna().sum()
    total_count = len(df)
    trade_count = (df['target'] == 1).sum()
    no_trade_count = (df['target'] == 0).sum()
    
    logger.info(f"V1 Labels: {valid_count}/{total_count} valid, "
                f"TRADE={trade_count}, NO_TRADE={no_trade_count}")
    
    return df


def generate_labels_v2(
    df: pd.DataFrame,
    direction: int = 1,
    target_atr_mult: float = 1.5,
    stop_atr_mult: float = 1.0,
    time_limit_bars: int = 24,
    slippage_pct: float = 0.0005
) -> pd.DataFrame:
    """
    V2 Triple Barrier Labeling with Path Dependency.
    
    This is the CORRECT labeling for trading ML:
    - Entry at NEXT bar open (not current close)
    - Slippage applied to entry price
    - Path-dependent checking within each bar
    - Timeouts excluded from training
    
    Args:
        df: DataFrame with OHLCV and atr_14 columns
        direction: 1=Long, 2=Short
        target_atr_mult: Target distance as ATR multiple (e.g., 1.5)
        stop_atr_mult: Stop distance as ATR multiple (e.g., 1.0)
        time_limit_bars: Maximum bars to hold before timeout
        slippage_pct: Slippage on entry (e.g., 0.0005 = 0.05%)
        
    Returns:
        DataFrame with 'target' column:
            1 = WIN (target hit before stop)
            0 = LOSS (stop hit before target)
           -1 = TIMEOUT (neither hit)
           None = Insufficient future data
    """
    df = df.copy().sort_values('ts_utc').reset_index(drop=True)
    
    # Validate required columns
    required = ['open', 'high', 'low', 'close', 'ts_utc']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure ATR is computed
    if 'atr_14' not in df.columns:
        logger.info("Computing ATR for labeling...")
        from app.ml.feature_engineer import calculate_atr
        df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # Validate data order
    assert df['ts_utc'].is_monotonic_increasing, "Data must be sorted chronologically"
    
    labels = []
    
    for i in range(len(df)):
        # Need at least time_limit_bars + 1 future bars (entry at next bar)
        if i + time_limit_bars + 1 >= len(df):
            labels.append(None)
            continue
        
        atr = df['atr_14'].iloc[i]
        if pd.isna(atr) or atr <= 0:
            labels.append(None)
            continue
        
        # =====================================================================
        # CRITICAL: Entry at NEXT bar open with slippage
        # This simulates reality - you see signal at bar i, enter at bar i+1
        # =====================================================================
        entry = df['open'].iloc[i + 1]  # Next bar open
        
        if direction == 1:  # Long
            entry_slipped = entry * (1 + slippage_pct)
            target = entry_slipped + (target_atr_mult * atr)
            stop = entry_slipped - (stop_atr_mult * atr)
        else:  # Short
            entry_slipped = entry * (1 - slippage_pct)
            target = entry_slipped - (target_atr_mult * atr)
            stop = entry_slipped + (stop_atr_mult * atr)
        
        # =====================================================================
        # PATH-DEPENDENT CHECK: Evaluate each bar's high/low in sequence
        # This is critical - we check if stop is hit BEFORE target within bar
        # =====================================================================
        future_slice = df.iloc[i + 2 : i + 2 + time_limit_bars]  # Start from bar after entry
        
        label = _evaluate_path(future_slice, direction, target, stop)
        labels.append(label)
    
    df['target'] = labels
    
    # Log distribution
    valid = df['target'].notna()
    if valid.sum() > 0:
        valid_df = df[valid]
        wins = (valid_df['target'] == 1).sum()
        losses = (valid_df['target'] == 0).sum()
        timeouts = (valid_df['target'] == -1).sum()
        total = len(valid_df)
        
        logger.info(f"✓ V2 Triple Barrier Labels (dir={direction}):")
        logger.info(f"   Wins: {wins}/{total} ({wins/total*100:.1f}%)")
        logger.info(f"   Losses: {losses}/{total} ({losses/total*100:.1f}%)")
        logger.info(f"   Timeouts: {timeouts}/{total} ({timeouts/total*100:.1f}%)")
        
        # Expected Value calculation
        if wins > 0 and losses > 0:
            r_win = target_atr_mult / stop_atr_mult  # R-multiple for wins
            r_loss = -1.0  # R-multiple for losses
            win_rate = wins / (wins + losses)  # Exclude timeouts
            ev = (win_rate * r_win) + ((1 - win_rate) * r_loss)
            logger.info(f"   Expected R: {ev:.3f} (using win/loss only)")
        
        if wins < 10:
            logger.warning(f"⚠️ Very few wins ({wins}). Check ATR mult or data volume.")
    
    return df


def _evaluate_path(
    future_slice: pd.DataFrame,
    direction: int,
    target: float,
    stop: float
) -> int:
    """
    Evaluate price path through future bars.
    
    Returns:
        1 = Target hit first (win)
        0 = Stop hit first (loss)
       -1 = Neither hit (timeout)
    """
    for _, row in future_slice.iterrows():
        high = row['high']
        low = row['low']
        
        if direction == 1:  # Long
            # Check stop first (conservative - assume worst case intrabar)
            if low <= stop:
                return 0  # Loss
            if high >= target:
                return 1  # Win
        else:  # Short
            # Check stop first  
            if high >= stop:
                return 0  # Loss
            if low <= target:
                return 1  # Win
    
    return -1  # Timeout


def prepare_labeled_data(
    df: pd.DataFrame,
    exclude_timeouts: bool = True
) -> pd.DataFrame:
    """
    Prepare labeled data for training.
    
    Args:
        df: DataFrame with 'target' column
        exclude_timeouts: If True, removes timeout samples (-1)
        
    Returns:
        Filtered DataFrame ready for training
    """
    df = df.copy()
    
    # Remove null labels
    df = df[df['target'].notna()]
    
    # Optionally exclude timeouts
    if exclude_timeouts:
        before = len(df)
        df = df[df['target'] >= 0]
        after = len(df)
        if before > after:
            logger.info(f"Excluded {before - after} timeout samples from training")
    
    return df


# Keep old function name for backward compatibility
def generate_labels_multiclass(
    df: pd.DataFrame,
    buy_target_pct: float = 0.015,
    buy_stop_pct: float = 0.008,
    sell_target_pct: float = 0.015,
    sell_stop_pct: float = 0.008,
    window: int = 20
) -> pd.DataFrame:
    """DEPRECATED: Use generate_labels_v2() instead."""
    logger.warning("DEPRECATED: generate_labels_multiclass is obsolete. Using binary labels.")
    return generate_labels(df, target_pct=buy_target_pct, stop_pct=buy_stop_pct, window=window)
