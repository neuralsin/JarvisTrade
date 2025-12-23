"""
Spec 4: Multi-Class Labeling - HOLD/BUY/SELL
Generate target labels for training with intelligent sell signals
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_labels(
    df: pd.DataFrame,
    buy_target_pct: float = 0.015,
    buy_stop_pct: float = 0.008,
    sell_target_pct: float = 0.015,
    sell_stop_pct: float = 0.008,
    window: int = 20
) -> pd.DataFrame:
    """
    Multi-class labeling for buy/sell signal detection
    
    Labels:
        0 = HOLD: No clear signal
        1 = BUY: Price will rise >1.5% before falling >0.8%
        2 = SELL: Price will fall >1.5% before rising >0.8% (peak exit)
    
    Args:
        df: DataFrame with OHLC data
        buy_target_pct: Buy target gain (default 1.5%)
        buy_stop_pct: Buy stop loss (default 0.8%)
        sell_target_pct: Sell target drop (default 1.5%)
        sell_stop_pct: Sell stop rise (default 0.8%)
        window: Forward-looking window (default 20)
    
    Returns:
        DataFrame with 'target' column (0/1/2)
    """
    df = df.copy().sort_values('ts_utc').reset_index(drop=True)
    targets = []
    
    for idx in range(len(df)):
        if idx + window >= len(df):
            targets.append(None)
            continue
        
        current_close = df.loc[idx, 'close']
        current_high = df.loc[idx, 'high']
        current_low = df.loc[idx, 'low']
        
        # Define thresholds
        buy_target_price = current_close * (1 + buy_target_pct)
        buy_stop_price = current_close * (1 - buy_stop_pct)
        sell_target_price = current_close * (1 - sell_target_pct)
        sell_stop_price = current_close * (1 + sell_stop_pct)
        
        # Future window
        future_slice = df.iloc[idx+1:idx+1+window]
        
        # Check BUY signal: Will price rise significantly?
        buy_target_hit = (future_slice['high'] >= buy_target_price).any()
        if buy_target_hit:
            target_hit_idx = future_slice[future_slice['high'] >= buy_target_price].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            buy_stop_hit = (pre_target['low'] < buy_stop_price).any()
            
            if not buy_stop_hit:
                targets.append(1)  # BUY signal
                continue
        
        # Check SELL signal: Will price fall significantly?
        sell_target_hit = (future_slice['low'] <= sell_target_price).any()
        if sell_target_hit:
            target_hit_idx = future_slice[future_slice['low'] <= sell_target_price].index[0]
            pre_target = df.loc[idx+1:target_hit_idx]
            sell_stop_hit = (pre_target['high'] > sell_stop_price).any()
            
            if not sell_stop_hit:
                targets.append(2)  # SELL signal
                continue
        
        # Neither clear signal
        targets.append(0)  # HOLD
    
    df['target'] = targets
    
    # Log distribution
    valid_count = df['target'].notna().sum()
    total_count = len(df)
    hold_count = (df['target'] == 0).sum()
    buy_count = (df['target'] == 1).sum()
    sell_count = (df['target'] == 2).sum()
    
    logger.info(f"Generated multi-class labels: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
    logger.info(f"Class distribution: HOLD={hold_count} ({hold_count/valid_count*100:.1f}%), "
                f"BUY={buy_count} ({buy_count/valid_count*100:.1f}%), "
                f"SELL={sell_count} ({sell_count/valid_count*100:.1f}%)")
    
    if buy_count == 0 or sell_count == 0:
        logger.warning(f"Imbalanced classes! BUY={buy_count}, SELL={sell_count}. Consider adjusting thresholds.")
    
    return df
