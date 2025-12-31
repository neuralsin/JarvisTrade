"""
Distributed Lock for Celery Tasks

Bug fix #22: Prevents task overlap by using Redis distributed locks.
Ensures only one instance of a task runs at a time.
"""
import logging
from contextlib import contextmanager
from typing import Optional
import time

from app.config import settings

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when unable to acquire a distributed lock."""
    pass


@contextmanager
def distributed_lock(
    lock_name: str,
    timeout: int = 300,
    blocking: bool = False,
    blocking_timeout: int = 10
):
    """
    Bug fix #22: Distributed lock using Redis.
    
    Context manager that acquires a Redis lock and releases it on exit.
    Prevents multiple instances of the same task from running concurrently.
    
    Args:
        lock_name: Unique identifier for the lock (e.g., 'task:signal_generation')
        timeout: Lock expiration time in seconds (prevents deadlocks)
        blocking: If True, wait for lock. If False, raise immediately if unavailable
        blocking_timeout: How long to wait if blocking=True
        
    Yields:
        Lock object if acquired
        
    Raises:
        LockAcquisitionError: If lock cannot be acquired
        
    Example:
        with distributed_lock('task:generate_signals'):
            # Only one worker can execute this at a time
            run_signal_generation()
    """
    import redis
    
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
    except Exception as e:
        logger.error(f"Failed to connect to Redis for lock: {e}")
        # Proceed without locking if Redis unavailable
        yield None
        return
    
    lock = redis_client.lock(
        name=f"jarvistrade:{lock_name}",
        timeout=timeout,
        blocking=blocking,
        blocking_timeout=blocking_timeout
    )
    
    acquired = False
    try:
        acquired = lock.acquire(blocking=blocking, blocking_timeout=blocking_timeout)
        
        if not acquired:
            raise LockAcquisitionError(
                f"Could not acquire lock '{lock_name}'. "
                f"Another instance may be running."
            )
        
        logger.debug(f"Acquired lock: {lock_name}")
        yield lock
        
    finally:
        if acquired:
            try:
                lock.release()
                logger.debug(f"Released lock: {lock_name}")
            except Exception as e:
                logger.warning(f"Error releasing lock {lock_name}: {e}")


def is_task_running(task_name: str) -> bool:
    """
    Check if a task is currently holding a lock.
    
    Args:
        task_name: Name of the task (e.g., 'signal_generation')
        
    Returns:
        True if task lock exists, False otherwise
    """
    import redis
    
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        lock_key = f"jarvistrade:task:{task_name}"
        return redis_client.exists(lock_key) > 0
    except Exception as e:
        logger.warning(f"Error checking task lock: {e}")
        return False


def task_with_lock(lock_name: str, timeout: int = 300):
    """
    Decorator to add distributed locking to a function.
    
    Args:
        lock_name: Unique lock identifier
        timeout: Lock timeout in seconds
        
    Example:
        @task_with_lock('signal_generation', timeout=600)
        def generate_signals():
            # Only one instance can run at a time
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                with distributed_lock(lock_name, timeout):
                    return func(*args, **kwargs)
            except LockAcquisitionError:
                logger.warning(f"Skipping {func.__name__} - lock unavailable")
                return {
                    'status': 'skipped',
                    'reason': 'lock_unavailable',
                    'lock': lock_name
                }
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
