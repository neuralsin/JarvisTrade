"""
Peak Detection Module
Identifies when stock is at local peak and likely to decline
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_peak_features(df: pd.DataFrame, lookback: int = 20) -> Dict[str, pd.Series]:
    """
    Compute 5 peak detection indicators
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        lookback: Number of candles to look back for peak detection
    
    Returns:
        Dictionary of peak indicator series
    """
    features = {}
    
    # 1. Price Position: Distance from recent high
    recent_high = df['high'].rolling(lookback).max()
    features['pct_from_high'] = (df['close'] - recent_high) / recent_high
    features['at_peak'] = (abs(features['pct_from_high']) < 0.005).astype(int)  # Within 0.5%
    
    # 2. RSI Overbought Condition
    if 'rsi_14' in df.columns:
        features['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        features['rsi_extreme'] = (df['rsi_14'] > 80).astype(int)
    else:
        features['rsi_overbought'] = 0
        features['rsi_extreme'] = 0
    
    # 3. Volume Surge: Unusual volume activity
    avg_volume = df['volume'].rolling(lookback).mean()
    features['volume_surge'] = df['volume'] / avg_volume
    features['high_volume'] = (features['volume_surge'] > 2.0).astype(int)
    
    # 4. Bearish Divergence: Price up but momentum down
    price_roc = df['close'].pct_change(5)
    if 'rsi_14' in df.columns:
        rsi_roc = df['rsi_14'].diff(5)
        features['bearish_divergence'] = ((price_roc > 0) & (rsi_roc < 0)).astype(int)
    else:
        features['bearish_divergence'] = 0
    
    # 5. Momentum Slowing: Rate of change decreasing
    price_diff_1 = df['close'].diff(1)
    price_diff_2 = df['close'].diff(2).shift(1)
    features['momentum_slowing'] = (price_diff_1 < price_diff_2).astype(int)
    
    # Composite peak score (0-5)
    features['peak_score'] = (
        features['at_peak'] +
        features['rsi_overbought'] +
        features['high_volume'] +
        features['bearish_divergence'] +
        features['momentum_slowing']
    )
    
    logger.debug(f"Peak detection: score range {features['peak_score'].min()}-{features['peak_score'].max()}")
    
    return features


def is_at_peak(latest_candles: pd.DataFrame, min_score: int = 3) -> Dict:
    """
    Determine if stock is currently at peak
    
    Args:
        latest_candles: Recent candle data (needs minimum 20 candles)
        min_score: Minimum peak score to consider it a peak (default 3/5)
    
    Returns:
        Dict with peak detection results
    """
    if len(latest_candles) < 20:
        return {
            'is_peak': False,
            'reason': 'Insufficient data',
            'score': 0
        }
    
    # Compute peak features
    peak_features = compute_peak_features(latest_candles)
    
    # Get latest values
    latest_score = peak_features['peak_score'].iloc[-1]
    latest_rsi = latest_candles['rsi_14'].iloc[-1] if 'rsi_14' in latest_candles.columns else 50
    latest_pct_from_high = peak_features['pct_from_high'].iloc[-1]
    
    is_peak = latest_score >= min_score
    
    result = {
        'is_peak': bool(is_peak),
        'score': int(latest_score),
        'rsi': float(latest_rsi),
        'pct_from_high': float(latest_pct_from_high),
        'details': {
            'at_peak': bool(peak_features['at_peak'].iloc[-1]),
            'rsi_overbought': bool(peak_features['rsi_overbought'].iloc[-1]),
            'high_volume': bool(peak_features['high_volume'].iloc[-1]),
            'bearish_divergence': bool(peak_features['bearish_divergence'].iloc[-1]),
            'momentum_slowing': bool(peak_features['momentum_slowing'].iloc[-1])
        }
    }
    
    if is_peak:
        reasons = [k for k, v in result['details'].items() if v]
        result['reason'] = f"Peak detected: {', '.join(reasons)}"
        logger.info(f"PEAK DETECTED! Score: {latest_score}/5, RSI: {latest_rsi:.1f}, {result['reason']}")
    else:
        result['reason'] = f"No peak: score {latest_score}/{min_score}"
    
    return result


def should_sell_at_peak(
    latest_candles: pd.DataFrame,
    entry_price: float,
    min_profit_pct: float = 0.01,
    min_peak_score: int = 3
) -> Dict:
    """
    Determine if position should be exited due to peak conditions
    
    Args:
        latest_candles: Recent candle data
        entry_price: Entry price of the position
        min_profit_pct: Minimum profit percentage to allow peak exit (default 1%)
        min_peak_score: Minimum peak score required (default 3/5)
    
    Returns:
        Dict with sell decision and rationale
    """
    if len(latest_candles) < 20:
        return {'should_sell': False, 'reason': 'Insufficient data for peak detection'}
    
    current_price = latest_candles['close'].iloc[-1]
    profit_pct = (current_price - entry_price) / entry_price
    
    # Must be in profit first
    if profit_pct < min_profit_pct:
        return {
            'should_sell': False,
            'reason': f'Not enough profit ({profit_pct*100:.2f}% < {min_profit_pct*100:.1f}%)',
            'profit_pct': profit_pct
        }
    
    # Check for peak
    peak_result = is_at_peak(latest_candles, min_score=min_peak_score)
    
    if peak_result['is_peak']:
        return {
            'should_sell': True,
            'reason': f"Peak exit: {profit_pct*100:.2f}% profit, {peak_result['reason']}",
            'profit_pct': profit_pct,
            'peak_score': peak_result['score'],
            'peak_details': peak_result['details']
        }
    
    return {
        'should_sell': False,
        'reason': peak_result['reason'],
        'profit_pct': profit_pct,
        'peak_score': peak_result['score']
    }
