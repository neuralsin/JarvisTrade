"""
Spec 4: BINARY Labeling - TRADE vs NO-TRADE
FIX 2: Multi-class was poisoning the pipeline. Binary is correct for trading ML.

Labels:
    1 = TRADE WORTH TAKING: Price will move favorably with good risk/reward
    0 = NO TRADE: Not a good entry

Direction (BUY vs SELL) handled separately by trend filter, NOT the model.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_labels(
    df: pd.DataFrame,
    target_pct: float = 0.015,    # 1.5% target
    stop_pct: float = 0.008,      # 0.8% stop
    window: int = 20,
    use_binary: bool = True       # FIX 2: FORCE BINARY
) -> pd.DataFrame:
    """
    Binary labeling for trade detection.
    
    Labels:
        1 = TRADE: Good entry - price moves favorably with acceptable risk
        0 = NO TRADE: Poor entry - avoid
    
    Args:
        df: DataFrame with OHLC data
        target_pct: Target gain/drop (default 1.5%)
        stop_pct: Stop loss (default 0.8%)
        window: Forward-looking window (default 20)
        use_binary: Force binary labels (default True)
    
    Returns:
        DataFrame with 'target' column (0/1)
    """
    df = df.copy().sort_values('ts_utc').reset_index(drop=True)
    targets = []
    
    for idx in range(len(df)):
        if idx + window >= len(df):
            targets.append(None)
            continue
        
        current_close = df.loc[idx, 'close']
        
        # Define thresholds
        target_price_up = current_close * (1 + target_pct)
        stop_price_down = current_close * (1 - stop_pct)
        target_price_down = current_close * (1 - target_pct)
        stop_price_up = current_close * (1 + stop_pct)
        
        # Future window
        future_slice = df.iloc[idx+1:idx+1+window]
        
        # Check LONG trade: Price rises to target before hitting stop
        target_hit_up = (future_slice['high'] >= target_price_up).any()
        if target_hit_up:
            target_hit_idx = future_slice[future_slice['high'] >= target_price_up].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            stop_hit = (pre_target['low'] < stop_price_down).any()
            
            if not stop_hit:
                targets.append(1)  # GOOD TRADE
                continue
        
        # Check SHORT trade: Price drops to target before hitting stop
        target_hit_down = (future_slice['low'] <= target_price_down).any()
        if target_hit_down:
            target_hit_idx = future_slice[future_slice['low'] <= target_price_down].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            stop_hit = (pre_target['high'] > stop_price_up).any()
            
            if not stop_hit:
                targets.append(1)  # GOOD TRADE (either direction)
                continue
        
        # No good trade opportunity
        targets.append(0)  # NO TRADE
    
    df['target'] = targets
    
    # Log distribution
    valid_count = df['target'].notna().sum()
    total_count = len(df)
    trade_count = (df['target'] == 1).sum()
    no_trade_count = (df['target'] == 0).sum()
    
    logger.info(f"Generated BINARY labels: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
    logger.info(f"Class distribution: TRADE={trade_count} ({trade_count/valid_count*100:.1f}%), "
                f"NO_TRADE={no_trade_count} ({no_trade_count/valid_count*100:.1f}%)")
    
    if trade_count < 10:
        logger.warning(f"Very few TRADE samples ({trade_count}). Consider adjusting thresholds.")
    
    trade_ratio = trade_count / valid_count if valid_count > 0 else 0
    logger.info(f"Trade ratio: {trade_ratio:.1%} (target: 20-40% for good precision)")
    
    return df


# Keep old function for backward compatibility but log deprecation
def generate_labels_multiclass(
    df: pd.DataFrame,
    buy_target_pct: float = 0.015,
    buy_stop_pct: float = 0.008,
    sell_target_pct: float = 0.015,
    sell_stop_pct: float = 0.008,
    window: int = 20
) -> pd.DataFrame:
    """DEPRECATED: Multi-class labeling. Use generate_labels() instead."""
    logger.warning("DEPRECATED: generate_labels_multiclass is using 3-class labeling. "
                   "This poisons the pipeline. Switching to binary.")
    return generate_labels(df, target_pct=buy_target_pct, stop_pct=buy_stop_pct, window=window)
