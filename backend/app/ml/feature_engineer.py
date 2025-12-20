"""
Spec 3: Feature engineering exact algorithms
Vectorized pandas operations for all technical indicators
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range)
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_features(df: pd.DataFrame, nifty_ema200: float = None, vix: float = None) -> pd.DataFrame:
    """
    Spec 3: Compute all features for ML model
    
    Features:
    - returns_1, returns_5
    - ema_20, ema_50, ema_200
    - distance_from_ema200
    - rsi_14, rsi_slope
    - atr_14, atr_percent
    - volume_ratio
    - nifty_trend (requires external nifty_ema200)
    - vix (external)
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume, ts_utc]
        nifty_ema200: current Nifty 200 EMA value (optional)
        vix: current VIX value (optional)
    
    Returns:
        DataFrame with computed features
    """
    df = df.copy().sort_values('ts_utc')
    
    # Returns
    df['returns_1'] = df['close'] / df['close'].shift(1) - 1
    df['returns_5'] = df['close'] / df['close'].shift(5) - 1
    
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Distance from EMA200
    df['distance_from_ema200'] = (df['close'] - df['ema_200']) / df['close']
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(1)
    
    # True Range and ATR
    df['tr_range'] = df['high'] - df['low']
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    df['atr_percent'] = df['atr_14'] / df['close']
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Nifty trend (if provided)
    if nifty_ema200 is not None:
        # This would typically be computed from Nifty data
        # For now, placeholder
        df['nifty_trend'] = 1  # TODO: Implement properly
    else:
        df['nifty_trend'] = 1
    
    # VIX
    df['vix'] = vix if vix is not None else 20.0  # Default placeholder
    
    # Round to 8 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(8)
    
    # Spec 3: Drop rows where ema_200 or atr_14 is null
    df = df.dropna(subset=['ema_200', 'atr_14'])
    
    return df


def extract_feature_vector(row: pd.Series) -> Dict[str, float]:
    """
    Extract feature vector as dictionary for ML prediction
    """
    feature_cols = [
        'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_200',
        'distance_from_ema200', 'rsi_14', 'rsi_slope',
        'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
    ]
    
    features = {col: row.get(col, 0) for col in feature_cols}
    return features


def features_to_dataframe(feature_json: Dict) -> pd.DataFrame:
    """
    Convert feature dictionary to DataFrame for model prediction
    """
    return pd.DataFrame([feature_json])
