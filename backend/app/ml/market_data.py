"""
Market data utilities - Fetch real-time Nifty50 and India VIX using direct HTTP
"""
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for market data (in-memory, could use Redis)
_market_data_cache = {
    'nifty_trend': {'value': None, 'timestamp': None},
    'india_vix': {'value': None, 'timestamp': None}
}


def fetch_nifty_trend(max_retries: int = 3) -> int:
    """
    Fetch Nifty50 trend using EMA crossover strategy with retry logic
    
    Returns:
        int: 1 for bullish (EMA20 > EMA50), 0 for bearish
    
    Raises:
        None - Always returns a value (defaults to 1 on failure)
    """
    # Check cache (5-minute TTL)
    cached = _market_data_cache['nifty_trend']
    if cached['value'] is not None and cached['timestamp']:
        age = (datetime.utcnow().replace(tzinfo=None) - cached['timestamp']).total_seconds()
        if age < 300:  # 5 minutes
            logger.debug(f"Nifty trend cache hit (age: {age:.0f}s)")
            return cached['value']
    
    
    # Use direct HTTP instead of yfinance
    import requests
    
    for attempt in range(max_retries):
        try:
            # Direct HTTP to Yahoo Finance v8 API
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^NSEI?interval=1d&range=60d"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                if len(timestamps) < 50:
                    logger.warning(f"Insufficient Nifty data ({len(timestamps)} rows, need 50+)")
                    return 1
                
                # Build DataFrame
                import pandas as pd
                df_data = []
                for i, ts in enumerate(timestamps):
                    if quotes['close'][i] is None:
                        continue
                    df_data.append({
                        'timestamp': datetime.fromtimestamp(ts),
                        'Close': float(quotes['close'][i])
                    })
                
                if not df_data:
                    logger.warning("No valid Nifty data")
                    return 1
                
                hist = pd.DataFrame(df_data)
                hist.set_index('timestamp', inplace=True)
                
                # Calculate EMAs
                ema_20 = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
                ema_50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                
                trend = 1 if ema_20 > ema_50 else 0
                
                # Update cache
                _market_data_cache['nifty_trend'] = {
                    'value': trend,
                    'timestamp': datetime.utcnow().replace(tzinfo=None)
                }
                
                logger.info(f"Nifty50 trend: {'Bullish' if trend else 'Bearish'} (EMA20={ema_20:.2f}, EMA50={ema_50:.2f})")
                return trend
            
            elif response.status_code == 429:
                logger.warning(f"Rate limited on Nifty fetch, waiting...")
                import time
                time.sleep(5)
                continue
            else:
                logger.warning(f"Nifty fetch HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    continue
                return 1
        
        except Exception as e:
            logger.error(f"Failed to fetch Nifty trend (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                logger.error("All Nifty fetch attempts failed, defaulting to bullish")
                return 1


def fetch_india_vix(max_retries: int = 3) -> float:
    """
    Fetch current India VIX (volatility index) with retry logic
    
    Returns:
        float: VIX value or 20.0 as default fallback
    
    Raises:
        None - Always returns a value (defaults to 20.0 on failure)
    """
    # Check cache (15-minute TTL)
    cached = _market_data_cache['india_vix']
    if cached['value'] is not None and cached['timestamp']:
        age = (datetime.utcnow().replace(tzinfo=None) - cached['timestamp']).total_seconds()
        if age < 900:  # 15 minutes
            logger.debug(f"India VIX cache hit (age: {age:.0f}s)")
            return cached['value']
    
    
    # Use direct HTTP instead of yfinance
    import requests
    
    for attempt in range(max_retries):
        try:
            # Direct HTTP to Yahoo Finance v8 API
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^INDIAVIX?interval=1d&range=1d"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                if 'timestamp' not in result or not result['timestamp']:
                    logger.warning("No VIX data in response")
                    if attempt < max_retries - 1:
                        continue
                    return 20.0
                
                quotes = result['indicators']['quote'][0]
                vix = float(quotes['close'][-1])  # Latest close
                
                # Validate VIX is in reasonable range (5-60)
                if not (5.0 <= vix <= 60.0):
                    logger.warning(f"VIX value {vix:.2f} outside expected range [5, 60], using default")
                    return 20.0
                
                # Update cache
                _market_data_cache['india_vix'] = {
                    'value': vix,
                    'timestamp': datetime.utcnow().replace(tzinfo=None)
                }
                
                logger.info(f"India VIX: {vix:.2f}")
                return vix
            
            elif response.status_code == 429:
                logger.warning("Rate limited on VIX fetch")
                import time
                time.sleep(5)
                continue
            else:
                logger.warning(f"VIX fetch HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    continue
                return 20.0
        
        except Exception as e:
            logger.error(f"Failed to fetch India VIX (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                logger.error("All VIX fetch attempts failed, defaulting to 20.0")
                return 20.0


def clear_market_data_cache():
    """Clear the market data cache (useful for testing)"""
    global _market_data_cache
    _market_data_cache = {
        'nifty_trend': {'value': None, 'timestamp': None},
        'india_vix': {'value': None, 'timestamp': None}
    }
