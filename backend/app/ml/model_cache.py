"""
Model Cache - Singleton Pattern for ML Models

Bug fix #16: Prevents disk I/O thrashing by caching loaded models in memory.
Models are loaded once when first requested and reused for subsequent predictions.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Bug fix #16: Singleton cache for ML models.
    
    Prevents loading models from disk on every prediction cycle.
    Thread-safe with lock protection.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_cache()
        return cls._instance
    
    def _init_cache(self):
        """Initialize the cache storage."""
        self._cache: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._cache_lock = threading.Lock()
        logger.info("ModelCache initialized")
    
    def get(self, model_path: str) -> Optional[Any]:
        """
        Get a cached model by path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Cached model or None if not cached
        """
        with self._cache_lock:
            if model_path in self._cache:
                self._access_count[model_path] = self._access_count.get(model_path, 0) + 1
                logger.debug(f"Cache hit for {model_path}")
                return self._cache[model_path]
        return None
    
    def put(self, model_path: str, model: Any) -> None:
        """
        Cache a loaded model.
        
        Args:
            model_path: Path to the model file
            model: The loaded model object
        """
        import time
        with self._cache_lock:
            self._cache[model_path] = model
            self._load_times[model_path] = time.time()
            self._access_count[model_path] = 1
            logger.info(f"Cached model: {model_path}")
    
    def invalidate(self, model_path: str) -> bool:
        """
        Remove a model from cache.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model was in cache, False otherwise
        """
        with self._cache_lock:
            if model_path in self._cache:
                del self._cache[model_path]
                self._load_times.pop(model_path, None)
                self._access_count.pop(model_path, None)
                logger.info(f"Invalidated cache: {model_path}")
                return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cached models.
        
        Returns:
            Number of models cleared
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            self._load_times.clear()
            self._access_count.clear()
            logger.info(f"Cleared {count} models from cache")
            return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cached_models': len(self._cache),
                'total_accesses': sum(self._access_count.values()),
                'models': list(self._cache.keys())
            }
    
    def load_model(self, model_path: str, model_format: str = 'joblib') -> Any:
        """
        Load a model with caching.
        
        Args:
            model_path: Path to the model file
            model_format: 'joblib' or 'keras'
            
        Returns:
            Loaded model
        """
        # Check cache first
        cached = self.get(model_path)
        if cached is not None:
            return cached
        
        # Load from disk
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from disk: {model_path}")
        
        if model_format == 'joblib':
            import joblib
            model = joblib.load(model_path)
        elif model_format == 'keras':
            from tensorflow import keras
            model = keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        # Cache and return
        self.put(model_path, model)
        return model


# Singleton accessor
def get_model_cache() -> ModelCache:
    """Get the singleton ModelCache instance."""
    return ModelCache()


def load_cached_model(model_path: str, model_format: str = 'joblib') -> Any:
    """
    Convenience function to load a model with caching.
    
    Args:
        model_path: Path to the model file
        model_format: 'joblib' or 'keras'
        
    Returns:
        Loaded model
    """
    return get_model_cache().load_model(model_path, model_format)
