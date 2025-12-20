"""
Spec 16: Error handling and retries with exponential backoff
Retry decorator for external API calls
"""
import time
import functools
from typing import Callable, Optional, Tuple, Type
import logging

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 5,
    initial_delay: float = 2.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    wait_time = min(delay, max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= backoff
            
            return None  # Should never reach here
        
        return wrapper
    return decorator
