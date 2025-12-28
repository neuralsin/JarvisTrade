"""
V2 Market Regime Detection Engine
Stateful regime classification with persistence filter.

Regimes:
- TREND_STABLE: ADX > 25 AND ATR_Z < 1 (Trending, low volatility)
- TREND_VOLATILE: ADX > 25 AND ATR_Z >= 1 (Trending, high volatility)
- RANGE_QUIET: ADX < 20 AND ATR_Z < 0 (Ranging, below-average volatility)
- CHOP_PANIC: ATR_Z >= 2 (Extreme volatility - no trading)
"""
import pandas as pd
import numpy as np
from enum import IntEnum
from collections import deque
from statistics import mode as stat_mode
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Market regime classification."""
    TREND_STABLE = 0
    TREND_VOLATILE = 1
    RANGE_QUIET = 2
    CHOP_PANIC = 3


class RegimeDetector:
    """
    Stateful regime detector with persistence filtering.
    
    Uses ADX(14) for trend strength and ATR Z-score for volatility regime.
    Applies mode filter over last N bars to prevent regime flipping.
    """
    
    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_z_volatile_threshold: float = 1.0,
        atr_z_panic_threshold: float = 2.0,
        persistence_bars: int = 5
    ):
        """
        Initialize regime detector.
        
        Args:
            adx_trend_threshold: ADX above this = trending market
            adx_range_threshold: ADX below this = ranging market
            atr_z_volatile_threshold: ATR Z-score above this = volatile
            atr_z_panic_threshold: ATR Z-score above this = panic/chop
            persistence_bars: Number of bars for mode filter
        """
        self.adx_trend = adx_trend_threshold
        self.adx_range = adx_range_threshold
        self.atr_z_volatile = atr_z_volatile_threshold
        self.atr_z_panic = atr_z_panic_threshold
        self.persistence = persistence_bars
        self.regime_history: deque = deque(maxlen=persistence_bars)
        self.current_regime: Optional[MarketRegime] = None
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ADX(14) and ATR Z-score(50).
        
        All indicators are computed on shifted (lagged) data to prevent lookahead.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # True Range (lagged)
        high_prev = df['high'].shift(1)
        low_prev = df['low'].shift(1)
        close_prev = df['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - close_prev)
        tr3 = abs(df['low'] - close_prev)
        df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR(14)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_percent'] = df['atr_14'] / df['close']
        
        # ATR Z-Score (50-bar lookback)
        atr_mean = df['atr_14'].rolling(50).mean()
        atr_std = df['atr_14'].rolling(50).std()
        df['atr_z'] = (df['atr_14'] - atr_mean) / atr_std.replace(0, 1)
        
        # Directional Movement
        high_diff = df['high'] - high_prev
        low_diff = low_prev - df['low']
        
        df['plus_dm'] = np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0
        )
        df['minus_dm'] = np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0
        )
        
        # Smoothed DI
        atr_smooth = df['atr_14']
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / atr_smooth.replace(0, 1))
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / atr_smooth.replace(0, 1))
        
        # DX and ADX
        di_sum = df['plus_di'] + df['minus_di']
        di_diff = abs(df['plus_di'] - df['minus_di'])
        df['dx'] = 100 * (di_diff / di_sum.replace(0, 1))
        df['adx_14'] = df['dx'].rolling(14).mean()
        
        return df
    
    def classify_bar(self, adx: float, atr_z: float) -> MarketRegime:
        """
        Classify single bar regime (before persistence filter).
        
        Args:
            adx: ADX(14) value
            atr_z: ATR Z-score value
            
        Returns:
            MarketRegime classification
        """
        # Panic check first (overrides everything)
        if atr_z >= self.atr_z_panic:
            return MarketRegime.CHOP_PANIC
        
        # Trending regimes
        if adx > self.adx_trend:
            if atr_z < self.atr_z_volatile:
                return MarketRegime.TREND_STABLE
            else:
                return MarketRegime.TREND_VOLATILE
        
        # Ranging regime
        if adx < self.adx_range and atr_z < 0:
            return MarketRegime.RANGE_QUIET
        
        # Default fallback
        return MarketRegime.TREND_VOLATILE
    
    def get_regime(self, adx: float, atr_z: float) -> MarketRegime:
        """
        Get regime with persistence filter (mode of last N bars).
        
        Prevents bar-to-bar regime flipping by requiring consensus.
        
        Args:
            adx: ADX(14) value
            atr_z: ATR Z-score value
            
        Returns:
            Filtered MarketRegime classification
        """
        raw_regime = self.classify_bar(adx, atr_z)
        self.regime_history.append(raw_regime)
        
        if len(self.regime_history) < self.persistence:
            self.current_regime = raw_regime
        else:
            try:
                self.current_regime = stat_mode(list(self.regime_history))
            except Exception:
                self.current_regime = raw_regime
        
        return self.current_regime
    
    def detect_from_df(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for entire DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series of MarketRegime values
        """
        df = self.compute_indicators(df)
        
        # Reset state for fresh detection
        self.regime_history.clear()
        self.current_regime = None
        
        regimes = []
        for i in range(len(df)):
            adx_val = df['adx_14'].iloc[i]
            atr_z_val = df['atr_z'].iloc[i]
            
            if pd.isna(adx_val) or pd.isna(atr_z_val):
                regimes.append(None)
            else:
                regime = self.get_regime(adx_val, atr_z_val)
                regimes.append(regime)
        
        return pd.Series(regimes, index=df.index, name='regime')
    
    def get_regime_name(self, regime: Optional[MarketRegime]) -> str:
        """Get human-readable regime name."""
        if regime is None:
            return "UNKNOWN"
        return regime.name
    
    def reset(self):
        """Reset detector state."""
        self.regime_history.clear()
        self.current_regime = None
