"""
Spec 3: Feature engineering exact algorithms
Vectorized pandas operations for all technical indicators
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging
from app.ml.market_data import fetch_nifty_trend, fetch_india_vix

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


def compute_features(df: pd.DataFrame, nifty_ema200: float = None, vix: float = None, instrument_id: str = None, db_session=None) -> pd.DataFrame:
    """
    Spec 3: Compute all features for ML model
    
    Features:
    - returns_1, returns_5
    - ema_20, ema_50, ema_100 (FIX 5: reduced from 200)
    - distance_from_ema100
    - rsi_14, rsi_slope
    - atr_14, atr_percent
    - volume_ratio
    - nifty_trend (requires external nifty_ema200)
    - vix (external)
    - NOTE: sentiment features REMOVED (FIX 4 - contaminating training)
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume, ts_utc]
        nifty_ema200: current Nifty 200 EMA value (optional)
        vix: current VIX value (optional)
        instrument_id: UUID of the instrument for sentiment lookup (optional)
        db_session: Database session for sentiment lookup (optional)
    
    Returns:
        DataFrame with computed features
    """
    df = df.copy().sort_values('ts_utc')
    
    # Returns
    df['returns_1'] = df['close'] / df['close'].shift(1) - 1
    df['returns_5'] = df['close'] / df['close'].shift(5) - 1
    
    # EMAs - FIX 5: Reduced to EMA100 (EMA200 on 1Y data loses too many samples)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Distance from EMA100 (renamed from ema_200)
    df['distance_from_ema100'] = (df['close'] - df['ema_100']) / df['close']
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(1)
    
    # True Range and ATR
    df['tr_range'] = df['high'] - df['low']
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    df['atr_percent'] = df['atr_14'] / df['close']
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Nifty trend - fetch real data
    nifty_trend_value = fetch_nifty_trend()
    df['nifty_trend'] = nifty_trend_value
    
    # VIX - fetch real India VIX
    vix_value = fetch_india_vix()
    df['vix'] = vix_value
    
    # FIX 4: Sentiment features REMOVED - they were contaminating training
    # When NEWS_API_KEY missing, sentiment=0.0 introduced systematic bias
    # Model learned "sentiment=0" as meaningful when it actually meant "unknown"
    # Direction should be handled by trend filter, not sentiment
    
    # Round to 8 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(8)
    
    # FIX 5: Changed from ema_200 to ema_100 - drops fewer rows
    df = df.dropna(subset=['ema_100', 'atr_14'])
    
    return df


def extract_feature_vector(row: pd.Series) -> Dict[str, float]:
    """
    Extract feature vector as dictionary for ML prediction
    FIX 4 & 5: Removed sentiment, renamed ema_200 -> ema_100
    """
    feature_cols = [
        'returns_1', 'returns_5', 'ema_20', 'ema_50', 'ema_100',
        'distance_from_ema100', 'rsi_14', 'rsi_slope',
        'atr_14', 'atr_percent', 'volume_ratio', 'nifty_trend', 'vix'
        # Sentiment features REMOVED - contaminating training
    ]
    
    features = {col: row.get(col, 0) for col in feature_cols}
    return features


def features_to_dataframe(feature_json: Dict) -> pd.DataFrame:
    """
    Convert feature dictionary to DataFrame for model prediction
    """
    return pd.DataFrame([feature_json])
