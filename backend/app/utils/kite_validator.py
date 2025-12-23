"""
Simple Kite API symbol validation
No rate limits, no authentication needed for instruments list
"""
from kiteconnect import KiteConnect
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Global cache
_instruments_cache = None


def validate_symbol(symbol: str, exchange: str = "NSE") -> dict:
    """
    Validate symbol using Kite Connect API.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        exchange: Exchange code ('NSE' or 'BSE')
    
    Returns:
        dict with 'valid', 'symbol', 'name', etc.
    """
    global _instruments_cache
    
    try:
        if not settings.KITE_API_KEY:
            return {"valid": False, "error": "Kite API key not configured"}
        
        # Fetch instruments if not cached
        if _instruments_cache is None:
            logger.info(f"Downloading {exchange} instruments list (one-time, ~2MB)...")
            kite = KiteConnect(api_key=settings.KITE_API_KEY)
            _instruments_cache = kite.instruments(exchange)
            logger.info(f"Cached {len(_instruments_cache)} {exchange} instruments")
        
        # Search for symbol
        symbol_upper = symbol.upper()
        matching = [i for i in _instruments_cache if i['tradingsymbol'] == symbol_upper]
        
        if not matching:
            return {
                "valid": False,
                "error": f"Symbol {symbol} not found on {exchange}"
            }
        
        inst = matching[0]
        return {
            "valid": True,
            "symbol": symbol_upper,
            "name": inst.get('name', symbol_upper),
            "exchange": exchange,
            "sector": "N/A",
            "industry": "N/A",
            "currency": "INR"
        }
    
    except Exception as e:
        logger.error(f"Kite validation error: {str(e)}")
        return {"valid": False, "error": f"Validation error: {str(e)}"}
