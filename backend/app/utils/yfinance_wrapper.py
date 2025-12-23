"""
Thread-safe Yahoo Finance wrapper with rate limiting and retry logic.
Prevents 429 Too Many Requests errors.
"""
import yfinance as yf
import time
import logging
from threading import Lock
from typing import Optional, Any
from datetime import datetime

try:
    from app.utils.yfinance_config import (
        YFINANCE_MIN_DELAY,
        YFINANCE_MAX_RETRIES,
        YFINANCE_BASE_WAIT,
        YFINANCE_REQUEST_DELAY
    )
except ImportError:
    # Fallback defaults if config file is missing
    YFINANCE_MIN_DELAY = 5.0
    YFINANCE_MAX_RETRIES = 5
    YFINANCE_BASE_WAIT = 10
    YFINANCE_REQUEST_DELAY = 1.0

logger = logging.getLogger(__name__)


class YFinanceRateLimiter:
    """
    Rate limiter for Yahoo Finance API calls.
    Enforces minimum delay between requests and handles retry logic.
    """
    
    def __init__(self, min_delay_seconds: float = None, max_retries: int = None, base_wait: float = None):
        """
        Initialize rate limiter.
        
        Args:
            min_delay_seconds: Minimum delay between API calls (default: from config)
            max_retries: Maximum retry attempts on rate limit errors (default: from config)
            base_wait: Base wait time for exponential backoff (default: from config)
        """
        self.min_delay = min_delay_seconds if min_delay_seconds is not None else YFINANCE_MIN_DELAY
        self.max_retries = max_retries if max_retries is not None else YFINANCE_MAX_RETRIES
        self.base_wait = base_wait if base_wait is not None else YFINANCE_BASE_WAIT
        self.request_delay = YFINANCE_REQUEST_DELAY
        self.last_request_time = None
        self.lock = Lock()
        
        logger.info(
            f"YFinance Rate Limiter initialized: "
            f"min_delay={self.min_delay}s, max_retries={self.max_retries}, "
            f"base_wait={self.base_wait}s, request_delay={self.request_delay}s"
        )
    
    def _wait_if_needed(self):
        """Enforce rate limit by waiting if needed."""
        with self.lock:
            if self.last_request_time is not None:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_delay:
                    sleep_time = self.min_delay - elapsed
                    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def download(self, ticker: str, **kwargs) -> Any:
        """
        Rate-limited wrapper for yf.download().
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional arguments for yf.download()
        
        Returns:
            DataFrame with historical data
        
        Raises:
            Exception: After all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                self._wait_if_needed()
                
                logger.debug(f"Fetching data for {ticker} (attempt {attempt + 1}/{self.max_retries})")
                df = yf.download(ticker, **kwargs)
                
                logger.debug(f"Successfully fetched {len(df)} rows for {ticker}")
                return df
            
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit error
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff using configured base_wait
                        wait_time = self.base_wait * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit for {ticker}. "
                            f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit persist after {self.max_retries} attempts for {ticker}")
                        raise
                
                # For other errors, log and re-raise
                logger.error(f"Error fetching {ticker}: {error_str}")
                raise
        
        raise Exception(f"Failed to fetch {ticker} after {self.max_retries} attempts")
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Rate-limited wrapper for yf.Ticker().
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            yf.Ticker instance
        """
        self._wait_if_needed()
        logger.debug(f"Creating Ticker for {symbol}")
        return yf.Ticker(symbol)
    
    def get_ticker_info(self, symbol: str, max_retries: Optional[int] = None) -> dict:
        """
        Rate-limited ticker.info with retry logic.
        
        Args:
            symbol: Stock ticker symbol
            max_retries: Override default max_retries
        
        Returns:
            dict: Ticker info
        
        Raises:
            Exception: After all retry attempts fail
        """
        retries = max_retries if max_retries is not None else self.max_retries
        
        for attempt in range(retries):
            try:
                # Rate-limit before creating ticker
                self._wait_if_needed()
                ticker = yf.Ticker(symbol)
                
                # Add small delay before accessing .info (separate HTTP request)
                time.sleep(self.request_delay)
                info = ticker.info
                
                if not info or 'symbol' not in info:
                    logger.warning(f"Empty info received for {symbol}")
                    return {}
                
                logger.debug(f"Successfully fetched info for {symbol}")
                return info
            
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit error
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < retries - 1:
                        # Exponential backoff using configured base_wait
                        wait_time = self.base_wait * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit for {symbol} info. "
                            f"Retrying in {wait_time}s (attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit persists after {retries} attempts for {symbol}")
                        raise
                
                logger.error(f"Error fetching info for {symbol}: {error_str}")
                raise
        
        raise Exception(f"Failed to fetch info for {symbol} after {retries} attempts")
    
    def get_ticker_history(
        self, 
        symbol: str, 
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_retries: Optional[int] = None
    ) -> Any:
        """
        Rate-limited ticker.history() with retry logic.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (e.g., '1d', '5d', '1mo', '1y')
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)
            interval: Data interval (e.g., '1m', '5m', '1h', '1d')
            max_retries: Override default max_retries
        
        Returns:
            DataFrame with historical data
        """
        retries = max_retries if max_retries is not None else self.max_retries
        
        for attempt in range(retries):
            try:
                # Rate-limit before creating ticker
                self._wait_if_needed()
                ticker = yf.Ticker(symbol)
                
                kwargs = {'interval': interval}
                if period:
                    kwargs['period'] = period
                if start:
                    kwargs['start'] = start
                if end:
                    kwargs['end'] = end
                
                # Add small delay before accessing .history() (separate HTTP request)
                time.sleep(self.request_delay)
                hist = ticker.history(**kwargs)
                
                logger.debug(f"Successfully fetched {len(hist)} rows for {symbol}")
                return hist
            
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit error
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < retries - 1:
                        # Exponential backoff using configured base_wait
                        wait_time = self.base_wait * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit for {symbol} history. "
                            f"Retrying in {wait_time}s (attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit persists after {retries} attempts for {symbol}")
                        raise
                
                logger.error(f"Error fetching history for {symbol}: {error_str}")
                raise
        
        raise Exception(f"Failed to fetch history for {symbol} after {retries} attempts")


# Global rate limiter instance (thread-safe singleton)
# Uses values from yfinance_config.py (5s delay, 5 retries, 10s base wait)
_rate_limiter = YFinanceRateLimiter()


def get_rate_limiter() -> YFinanceRateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


def configure_rate_limiter(min_delay_seconds: float = 2.0, max_retries: int = 3):
    """
    Configure the global rate limiter.
    
    Args:
        min_delay_seconds: Minimum delay between API calls
        max_retries: Maximum retry attempts
    """
    global _rate_limiter
    _rate_limiter = YFinanceRateLimiter(min_delay_seconds, max_retries)
    logger.info(f"Rate limiter configured: delay={min_delay_seconds}s, retries={max_retries}")
