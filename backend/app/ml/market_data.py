"""
Market data utilities - Fetch real-time Nifty50 and India VIX
"""
import yfinance as yf
import logging
from typing import Optional
from datetime import datetime, timedelta
from app.utils.yfinance_wrapper import get_rate_limiter

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
        age = (datetime.utcnow() - cached['timestamp']).total_seconds()
        if age < 300:  # 5 minutes
            logger.debug(f"Nifty trend cache hit (age: {age:.0f}s)")
            return cached['value']
    
    logger.debug("Nifty trend cache miss, fetching from Yahoo Finance")
    
    # Use rate-limited wrapper
    rate_limiter = get_rate_limiter()
    
    for attempt in range(max_retries):
        try:
            hist = rate_limiter.get_ticker_history("^NSEI", period="60d")
            
            if hist.empty:
                logger.warning(f"No Nifty50 data available (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return 1  # Default to bullish after all retries
            
            # Validate we have enough data
            if len(hist) < 50:
                logger.warning(f"Insufficient Nifty data ({len(hist)} rows, need 50+)")
                return 1
            
            # Calculate EMAs
            ema_20 = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema_50 = hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            
            trend = 1 if ema_20 > ema_50 else 0
            
            # Update cache
            _market_data_cache['nifty_trend'] = {
                'value': trend,
                'timestamp': datetime.utcnow()
            }
            
            logger.info(f"Nifty50 trend: {'Bullish' if trend else 'Bearish'} (EMA20={ema_20:.2f}, EMA50={ema_50:.2f})")
            return trend
        
        except Exception as e:
            logger.error(f"Failed to fetch Nifty trend (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief pause before retry
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
        age = (datetime.utcnow() - cached['timestamp']).total_seconds()
        if age < 900:  # 15 minutes
            logger.debug(f"India VIX cache hit (age: {age:.0f}s)")
            return cached['value']
    
    logger.debug("India VIX cache miss, fetching from Yahoo Finance")
    
    # Use rate-limited wrapper
    rate_limiter = get_rate_limiter()
    
    for attempt in range(max_retries):
        try:
            hist = rate_limiter.get_ticker_history("^INDIAVIX", period="1d")
            
            if hist.empty:
                logger.warning(f"No India VIX data available (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return 20.0  # Default after all retries
            
            vix = float(hist['Close'].iloc[-1])
            
            # Validate VIX is in reasonable range (5-60)
            if not (5.0 <= vix <= 60.0):
                logger.warning(f"VIX value {vix:.2f} outside expected range [5, 60], using default")
                return 20.0
            
            # Update cache
            _market_data_cache['india_vix'] = {
                'value': vix,
                'timestamp': datetime.utcnow()
            }
            
            logger.info(f"India VIX: {vix:.2f}")
            return vix
        
        except Exception as e:
            logger.error(f"Failed to fetch India VIX (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief pause before retry
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
