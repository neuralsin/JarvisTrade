"""
Groww API client for historical data - PRIMARY data source
Fallback to Yahoo Finance only if Groww fails
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class GrowwAPI:
    """
    Groww API client for historical candle data
    BASE_URL: https://api.groww.in/v1/historical/candles
    """
    
    BASE_URL = "https://api.groww.in/v1/historical/candles"
    
    # Interval mapping
    INTERVAL_MAP = {
        '1m': '1minute',
        '5m': '5minute',
        '15m': '15minute',
        '30m': '30minute',
        '1h': '1hour',
        '1d': '1day'
    }
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize Groww API client
        
        Args:
            access_token: Groww API access token (optional for backtesting data)
        """
        self.access_token = access_token or getattr(settings, 'GROWW_API_TOKEN', None)
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'X-API-VERSION': '1.0'
        })
        if self.access_token:
            self.session.headers['Authorization'] = f'Bearer {self.access_token}'
    
    def _convert_to_nse_symbol(self, symbol: str) -> str:
        """Convert symbol to NSE format (e.g., RELIANCE -> NSE-RELIANCE)"""
        symbol = symbol.upper().strip()
        if not symbol.startswith('NSE-'):
            return f'NSE-{symbol}'
        return symbol
    
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = '15m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 60
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Groww API
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TATAELXSI')
            interval: Candle interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            days: Number of days if dates not provided
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Convert symbol to NSE format
            groww_symbol = self._convert_to_nse_symbol(symbol)
            
            # Convert interval to Groww format
            candle_interval = self.INTERVAL_MAP.get(interval, '15minute')
            
            # Calculate dates if not provided
            if not end_date:
                end_dt = datetime.now()
                end_date = end_dt.strftime('%Y-%m-%d')
                end_time = end_dt.strftime('%H:%M:%S')
            else:
                end_time = '15:30:00'  # Market close
            
            if not start_date:
                start_dt = datetime.now() - timedelta(days=days)
                start_date = start_dt.strftime('%Y-%m-%d')
                start_time = '09:15:00'  # Market open
            else:
                start_time = '09:15:00'
            
            # Build request parameters
            params = {
                'exchange': 'NSE',
                'segment': 'CASH',
                'groww_symbol': groww_symbol,
                'start_time': f'{start_date} {start_time}',
                'end_time': f'{end_date} {end_time}',
                'candle_interval': candle_interval
            }
            
            logger.info(f"Fetching Groww data for {groww_symbol}, interval={candle_interval}, {start_date} to {end_date}")
            
            # Make API request
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'SUCCESS':
                raise ValueError(f"Groww API error: {data.get('message', 'Unknown error')}")
            
            # Parse candles from response
            candles = data.get('payload', {}).get('candles', [])
            
            if not candles:
                logger.warning(f"No candles returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.rename(columns={'timestamp': 'ts_utc'})
            
            # Drop open_interest (not needed for equities)
            df = df.drop(columns=['open_interest'], errors='ignore')
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('ts_utc').reset_index(drop=True)
            
            logger.info(f"✅ Groww API: Fetched {len(df)} candles for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Groww API failed for {symbol}: {str(e)}")
            raise


def fetch_with_fallback(symbol: str, interval: str = '15m', days: int = 60) -> pd.DataFrame:
    """
    PRIMARY: Try Groww API first
    FALLBACK: Use Yahoo Finance if Groww fails
    
    Args:
        symbol: Stock symbol
        interval: Candle interval
        days: Number of days of data
    
    Returns:
        DataFrame with OHLCV data (or empty DataFrame if both fail)
    """
    # TRY GROWW FIRST
    try:
        groww = GrowwAPI()
        df = groww.fetch_historical_data(symbol=symbol, interval=interval, days=days)
        if not df.empty:
            logger.info(f"✅ PRIMARY SOURCE (Groww): {len(df)} candles for {symbol}")
            return df
    except Exception as e:
        logger.warning(f"⚠️ Groww API failed for {symbol}: {str(e)}")
    
    # FALLBACK TO YAHOO
    try:
        from app.tasks.data_ingestion import fetch_historical_yahoo
        from datetime import datetime, timedelta
        
        # Calculate dates for Yahoo Finance (timezone-naive)
        end_dt = datetime.now().replace(tzinfo=None)  # ✅ Make timezone-naive
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        
        logger.info(f"Trying Yahoo Finance for {symbol}: {start_date} to {end_date}, interval={interval}")
        df = fetch_historical_yahoo(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if df is not None and not df.empty:
            # Rename column to match expected format
            df = df.reset_index()
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'ts_utc'})
            logger.info(f"✅ FALLBACK SOURCE (Yahoo): {len(df)} candles for {symbol}")
            return df
    except Exception as e:
        logger.warning(f"⚠️ Yahoo Finance also failed for {symbol}: {str(e)}")
    
    # BOTH FAILED - return empty DataFrame (don't crash training)
    logger.error(f"❌ Both Groww and Yahoo failed for {symbol} with interval={interval}")
    logger.info(f"💡 Suggestion: Try different interval (1d, 1h) or check if symbol exists on NSE")
    return pd.DataFrame()  # Return empty, let caller handle it
