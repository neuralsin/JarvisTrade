"""
Market data utilities - Fetch real-time Nifty50 and India VIX
"""
import yfinance as yf
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for market data (in-memory, could use Redis)
_market_data_cache = {
    'nifty_trend': {'value': None, 'timestamp': None},
    'india_vix': {'value': None, 'timestamp': None}
}


def fetch_nifty_trend() -> int:
    """
    Fetch Nifty50 trend using EMA crossover strategy
    Returns: 1 for bullish (EMA20 > EMA50), 0 for bearish
    """
    # Check cache (5-minute TTL)
    cached = _market_data_cache['nifty_trend']
    if cached['value'] is not None and cached['timestamp']:
        age = (datetime.utcnow() - cached['timestamp']).total_seconds()
        if age < 300:  # 5 minutes
            return cached['value']
    
    try:
        ticker = yf.Ticker("^NSEI")  # Nifty50 index
        hist = ticker.history(period="60d")
        
        if hist.empty:
            logger.warning("No Nifty50 data available, using default trend=1")
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
        logger.error(f"Failed to fetch Nifty trend: {str(e)}")
        return 1  # Default to bullish


def fetch_india_vix() -> float:
    """
    Fetch current India VIX (volatility index)
    Returns: VIX value or 20.0 as default
    """
    # Check cache (15-minute TTL)
    cached = _market_data_cache['india_vix']
    if cached['value'] is not None and cached['timestamp']:
        age = (datetime.utcnow() - cached['timestamp']).total_seconds()
        if age < 900:  # 15 minutes
            return cached['value']
    
    try:
        ticker = yf.Ticker("^INDIAVIX")
        hist = ticker.history(period="1d")
        
        if hist.empty:
            logger.warning("No India VIX data available, using default VIX=20.0")
            return 20.0
        
        vix = float(hist['Close'].iloc[-1])
        
        # Update cache
        _market_data_cache['india_vix'] = {
            'value': vix,
            'timestamp': datetime.utcnow()
        }
        
        logger.info(f"India VIX: {vix:.2f}")
        return vix
    
    except Exception as e:
        logger.error(f"Failed to fetch India VIX: {str(e)}")
        return 20.0  # Default fallback


def clear_market_data_cache():
    """Clear the market data cache (useful for testing)"""
    global _market_data_cache
    _market_data_cache = {
        'nifty_trend': {'value': None, 'timestamp': None},
        'india_vix': {'value': None, 'timestamp': None}
    }
