"""
DEPRECATED: Groww API has been replaced by market_data_fetcher.py

This file is kept for backward compatibility only.
All functions now redirect to the new unified market data fetcher.
"""
from app.utils.market_data_fetcher import (
    fetch_ohlcv,
    fetch_with_fallback,
    GrowwAPI,
    DataFetchError,
    SymbolNotFoundError,
    RateLimitError,
    NetworkError,
    InsufficientDataError
)

# Re-export for backward compatibility
__all__ = [
    'fetch_ohlcv',
    'fetch_with_fallback',
    'GrowwAPI',
    'DataFetchError',
    'SymbolNotFoundError',
    'RateLimitError',
    'NetworkError',
    'InsufficientDataError'
]
