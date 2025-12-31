"""
Data Coordinator - Yahoo to Kite Fallback

Bug fix #15: Creates a unified data fetching interface with automatic fallback
from Yahoo Finance to Kite Connect if primary source fails.
"""
import logging
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when data fetching fails from all sources."""
    pass


class DataCoordinator:
    """
    Bug fix #15: Unified data fetching with automatic fallback.
    
    Primary: Yahoo Finance (no auth required)
    Fallback: Kite Connect (requires auth, used if Yahoo fails)
    """
    
    def __init__(self):
        self.primary_source = 'yahoo'
        self.fallback_source = 'kite'
        self._stats = {'yahoo_success': 0, 'kite_success': 0, 'total_failures': 0}
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = '15m',
        days: int = 60,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with automatic fallback.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            interval: Candle interval ('1m', '5m', '15m', '1h', '1d')
            days: Number of days to fetch (used if dates not provided)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: ts_utc, open, high, low, close, volume
        
        Raises:
            DataFetchError: If all sources fail
        """
        # Calculate dates
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            start_dt = datetime.utcnow() - timedelta(days=days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Try primary source (Yahoo Finance)
        logger.debug(f"Attempting Yahoo Finance for {symbol}")
        df = self._fetch_yahoo(symbol, start_date, end_date, interval)
        
        if df is not None and not df.empty:
            self._stats['yahoo_success'] += 1
            return df
        
        # Fallback to Kite if available
        logger.warning(f"Yahoo failed for {symbol}, attempting Kite fallback")
        df = self._fetch_kite(symbol, start_date, end_date, interval)
        
        if df is not None and not df.empty:
            self._stats['kite_success'] += 1
            return df
        
        # All sources failed
        self._stats['total_failures'] += 1
        raise DataFetchError(f"All data sources failed for {symbol}")
    
    def _fetch_yahoo(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance."""
        try:
            from app.tasks.data_ingestion import fetch_historical_yahoo
            return fetch_historical_yahoo(symbol, start_date, end_date, interval, 'NS')
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {symbol}: {e}")
            return None
    
    def _fetch_kite(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from Kite Connect (fallback)."""
        try:
            from app.tasks.data_ingestion import get_kite_client, fetch_historical_kite
            
            kite = get_kite_client()
            if not kite:
                logger.debug("Kite client not available for fallback")
                return None
            
            # Convert interval format (Yahoo -> Kite)
            interval_map = {
                '1m': 'minute',
                '5m': '5minute',
                '15m': '15minute',
                '30m': '30minute',
                '1h': '60minute',
                '1d': 'day'
            }
            kite_interval = interval_map.get(interval, '15minute')
            
            from_date = datetime.strptime(start_date, '%Y-%m-%d')
            to_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            return fetch_historical_kite(kite, symbol, from_date, to_date, kite_interval)
        except Exception as e:
            logger.warning(f"Kite fetch failed for {symbol}: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Return fetch statistics."""
        return self._stats.copy()


# Singleton instance
_data_coordinator = None


def get_data_coordinator() -> DataCoordinator:
    """Get singleton DataCoordinator instance."""
    global _data_coordinator
    if _data_coordinator is None:
        _data_coordinator = DataCoordinator()
    return _data_coordinator


def fetch_with_fallback(
    symbol: str,
    interval: str = '15m',
    days: int = 60,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for data fetching with fallback.
    
    Args:
        symbol: Stock symbol
        interval: Candle interval
        days: Number of days
        **kwargs: Additional args passed to DataCoordinator.fetch_ohlcv
    
    Returns:
        DataFrame with OHLCV data
    
    Raises:
        DataFetchError: If all sources fail
    """
    coordinator = get_data_coordinator()
    return coordinator.fetch_ohlcv(symbol, interval, days, **kwargs)
