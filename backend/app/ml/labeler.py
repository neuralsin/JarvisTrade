"""
Spec 4: Labeling - exact window scanning implementation
Generate target labels for training
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_labels(df: pd.DataFrame, target_pct: float = 0.012, stop_pct: float = 0.006, window: int = 10) -> pd.DataFrame:
    """
    Spec 4: For each row t, compute target label:
    - target=1 if future_high >= close * 1.012 AND min_low before target >= close * 0.994
    - target=0 otherwise
    - target=NULL if insufficient future candles
    
    Args:
        df: DataFrame with close, high, low columns
        target_pct: Target percentage gain (default 1.2%)
        stop_pct: Stop loss percentage (default 0.6%)
        window: Forward-looking window (default 10 candles)
    
    Returns:
        DataFrame with 'target' column added
    """
    df = df.copy().sort_values('ts_utc').reset_index(drop=True)
    targets = []
    
    for idx in range(len(df)):
        # Check if we have enough future candles
        if idx + window >= len(df):
            targets.append(None)  # Not enough data
            continue
        
        current_close = df.loc[idx, 'close']
        target_price = current_close * (1 + target_pct)
        stop_price = current_close * (1 - stop_pct)
        
        # Look ahead in the next 'window' candles
        future_slice = df.iloc[idx+1:idx+1+window]
        
        # Find if target is hit
        target_hit_mask = future_slice['high'] >= target_price
        
        if not target_hit_mask.any():
            # Target never hit
            targets.append(0)
            continue
        
        # Find first index where target is hit
        target_hit_idx = future_slice[target_hit_mask].index[0]
        
        # Check if stop was hit before target
        pre_target_slice = df.loc[idx+1:target_hit_idx]
        stop_hit = (pre_target_slice['low'] < stop_price).any()
        
        if stop_hit:
            targets.append(0)  # Stop hit before target
        else:
            targets.append(1)  # Target reached without hitting stop
    
    df['target'] = targets
    
    # Spec 4: Exclude rows with NULL target from training
    valid_count = df['target'].notna().sum()
    total_count = len(df)
    logger.info(f"Generated labels: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
    
    return df
