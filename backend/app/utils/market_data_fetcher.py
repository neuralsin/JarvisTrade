"""
Market Data Fetcher - Yahoo Finance ONLY

Single source of truth for all OHLCV data fetching.
Replaces groww_api.py with simpler, more reliable implementation.

Principles:
1. Yahoo Finance only (no Groww auth required)
2. Explicit error types
3. Automatic retry with backoff
4. Rate limit awareness
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================
class DataFetchError(Exception):
    """Base exception for data fetching errors"""
    pass


class SymbolNotFoundError(DataFetchError):
    """Symbol does not exist on Yahoo Finance"""
    pass


class RateLimitError(DataFetchError):
    """Yahoo Finance rate limit hit"""
    pass


class NetworkError(DataFetchError):
    """Network connectivity issue"""
    pass


class InsufficientDataError(DataFetchError):
    """Not enough data points returned"""
    pass


# ============================================================================
# MAIN FETCHER
# ============================================================================
def fetch_ohlcv(
    symbol: str,
    interval: str = '15m',
    days: int = 60,
    min_candles: int = 0
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        interval: Candle interval ('1m', '5m', '15m', '30m', '1h', '1d')
        days: Number of days of history to fetch
        min_candles: Minimum candles required (raises InsufficientDataError if not met)
    
    Returns:
        DataFrame with columns: ts_utc, open, high, low, close, volume
    
    Raises:
        SymbolNotFoundError: If symbol doesn't exist
        RateLimitError: If rate limit exceeded
        NetworkError: If network issues
        InsufficientDataError: If min_candles not met
    """
    # Try NSE first, then BSE
    exchanges = ['NS', 'BO']
    last_error = None
    
    for exchange in exchanges:
        try:
            df = _fetch_from_yahoo(symbol, exchange, interval, days)
            
            if df is not None and len(df) >= min_candles:
                logger.info(f"✓ Fetched {len(df)} candles for {symbol}.{exchange}")
                return df
            
            if df is not None and len(df) < min_candles:
                last_error = InsufficientDataError(
                    f"Got {len(df)} candles, need at least {min_candles}"
                )
                
        except RateLimitError:
            raise  # Don't retry on rate limit
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to fetch {symbol}.{exchange}: {str(e)}")
            continue
    
    # All exchanges failed
    if isinstance(last_error, InsufficientDataError):
        raise last_error
    
    raise SymbolNotFoundError(
        f"Symbol {symbol} not found on Yahoo Finance (tried: {', '.join(exchanges)})"
    )


def _fetch_from_yahoo(
    symbol: str,
    exchange: str,
    interval: str,
    days: int
) -> Optional[pd.DataFrame]:
    """
    Internal: Fetch from Yahoo Finance with retry.
    
    Args:
        symbol: Base stock symbol
        exchange: 'NS' or 'BO'
        interval: Candle interval
        days: Number of days
    
    Returns:
        DataFrame or None
    """
    yahoo_symbol = f"{symbol}.{exchange}"
    
    # Calculate range parameter
    range_param = f'{days}d'
    
    # Yahoo Finance API URL
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    params = {
        'interval': interval,
        'range': range_param
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Retry loop
    for attempt in range(3):
        try:
            logger.debug(f"Yahoo Finance request: {yahoo_symbol} (attempt {attempt + 1})")
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            if response.status_code == 404:
                return None  # Symbol not found on this exchange
            
            if response.status_code != 200:
                raise NetworkError(f"HTTP {response.status_code}")
            
            data = response.json()
            
            # Parse response
            if 'chart' not in data or 'result' not in data['chart']:
                return None
            
            result = data['chart']['result']
            if not result or len(result) == 0:
                return None
            
            result = result[0]
            
            if 'timestamp' not in result or not result['timestamp']:
                return None
            
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Build DataFrame
            rows = []
            for i, ts in enumerate(timestamps):
                if quotes['close'][i] is None:
                    continue  # Skip null candles
                
                rows.append({
                    'ts_utc': datetime.utcfromtimestamp(ts),
                    'open': float(quotes['open'][i] or 0),
                    'high': float(quotes['high'][i] or 0),
                    'low': float(quotes['low'][i] or 0),
                    'close': float(quotes['close'][i]),
                    'volume': int(quotes['volume'][i] or 0)
                })
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            df = df.sort_values('ts_utc').reset_index(drop=True)
            
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {yahoo_symbol}, attempt {attempt + 1}")
            time.sleep(2)
            continue
            
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection error: {str(e)}")
        
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise
    
    return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def fetch_latest_price(symbol: str) -> Optional[float]:
    """
    Fetch the latest closing price for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
    
    Returns:
        Latest close price or None
    """
    try:
        df = fetch_ohlcv(symbol, interval='1d', days=5)
        if df is not None and len(df) > 0:
            return float(df.iloc[-1]['close'])
    except DataFetchError:
        pass
    return None


def fetch_intraday(symbol: str, interval: str = '15m') -> pd.DataFrame:
    """
    Fetch intraday data (last 60 days max).
    
    Args:
        symbol: Stock symbol
        interval: '1m', '5m', '15m', '30m', '1h'
    
    Returns:
        DataFrame with OHLCV data
    """
    return fetch_ohlcv(symbol, interval=interval, days=59)


def fetch_daily(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily data.
    
    Args:
        symbol: Stock symbol
        days: Number of days (max ~730 for Yahoo)
    
    Returns:
        DataFrame with OHLCV data
    """
    return fetch_ohlcv(symbol, interval='1d', days=min(days, 730))


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
def fetch_with_fallback(symbol: str, interval: str = '15m', days: int = 60) -> pd.DataFrame:
    """
    Backward compatible function matching old groww_api.py signature.
    
    This is now just a wrapper around fetch_ohlcv.
    """
    try:
        return fetch_ohlcv(symbol, interval=interval, days=days)
    except DataFetchError as e:
        logger.error(f"Failed to fetch {symbol}: {str(e)}")
        return pd.DataFrame()


# Keep GrowwAPI class for backward compatibility (does nothing now)
class GrowwAPI:
    """DEPRECATED: Use fetch_ohlcv() instead"""
    
    def __init__(self, access_token=None):
        logger.warning("GrowwAPI is deprecated. Use fetch_ohlcv() instead.")
    
    def fetch_historical_data(self, symbol: str, interval: str = '15m', **kwargs) -> pd.DataFrame:
        return fetch_ohlcv(symbol, interval=interval, days=kwargs.get('days', 60))
