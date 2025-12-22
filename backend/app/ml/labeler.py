"""
Spec 4: Labeling - exact window scanning implementation
Generate target labels for training
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_labels(df: pd.DataFrame, target_pct: float = 0.008, stop_pct: float = 0.004, window: int = 20) -> pd.DataFrame:
    """
    Spec 4: For each row t, compute target label:
    - target=1 if future_high >= close * 1.008 AND min_low before target >= close * 0.996
    - target=0 otherwise
    - target=NULL if insufficient future candles
    
    UPDATED: Reduced target to 0.8% and stop to 0.4%, increased window to 20
    for better balance between positive and negative samples
    
    Args:
        df: DataFrame with close, high, low columns
        target_pct: Target percentage gain (default 0.8%)
        stop_pct: Stop loss percentage (default 0.4%)
        window: Forward-looking window (default 20 candles)
    
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
    positive_count = (df['target'] == 1).sum()
    negative_count = (df['target'] == 0).sum()
    
    if positive_count > 0:
        pos_ratio = positive_count / valid_count * 100
        logger.info(f"Generated labels: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
        logger.info(f"Class distribution: {positive_count} positive ({pos_ratio:.1f}%), {negative_count} negative ({100-pos_ratio:.1f}%)")
    else:
        logger.warning(f"Generated {valid_count} labels but NO POSITIVE samples! Consider relaxing target_pct or increasing window.")
    
    return df
