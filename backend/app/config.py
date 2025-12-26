"""
Spec 0: Global constants, env vars, and config
Configuration management using pydantic-settings with 12-factor pattern
"""
from pydantic_settings import BaseSettings
from typing import Optional
import sys


class Settings(BaseSettings):
    # Application
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "changeme-in-production"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@db:5432/jarvistrade"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Zerodha Kite API
    KITE_API_KEY: Optional[str] = None
    KITE_API_SECRET: Optional[str] = None
    KITE_REDIRECT_URL: str = "http://localhost:8000/auth/kite/callback"
    
    # Email notifications
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASS: Optional[str] = None
    
    # Trading parameters
    ACCOUNT_CAPITAL: float = 100000.0
    RISK_PER_TRADE: float = 0.01
    MAX_DAILY_LOSS: float = 0.02
    STOP_MULTIPLIER: float = 1.5
    TARGET_MULTIPLIER: float = 2.5
    PROB_MIN: float = 0.50  # Lowered from 0.65 for more signals (Phase 1)
    PROB_STRONG: float = 0.70  # Adjusted accordingly
    MAX_TRADES_PER_DAY: int = 3
    
    # Model auto-activation (Phase 1)
    AUTO_ACTIVATE_MODELS: bool = True  # Auto-activate models with good metrics
    MODEL_MIN_AUC: float = 0.60  # Minimum AUC to auto-activate
    MODEL_MIN_ACCURACY: float = 0.55  # Minimum accuracy to auto-activate
    
    # Feature computation (Phase 1)
    FEATURE_CACHE_SECONDS: int = 60  # Cache features for 60 seconds
    FEATURE_MAX_AGE_SECONDS: int = 120  # Reject features older than 2 minutes
    FRESH_FEATURES_ENABLED: bool = True  # Enable real-time feature computation
    
    # Slippage simulation
    SLIPPAGE_PCT_MIN: float = 0.0001
    SLIPPAGE_PCT_MAX: float = 0.0005
    
    # Commission settings
    COMMISSION_FLAT: float = 20.0
    COMMISSION_PCT: float = 0.0003
    
    # Encryption
    KMS_MASTER_KEY: str = "dev-key-replace-in-production"
    
    # Model hyperparameters
    XGBOOST_N_ESTIMATORS: int = 500
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8
    
    # Circuit breakers
    VIX_THRESHOLD: float = 40.0
    NIFTY_DROP_THRESHOLD: float = 0.04
    
    # High-frequency trading configuration
    SIGNAL_CHECK_INTERVAL: int = 60  # seconds
    POSITION_CHECK_INTERVAL: int = 30  # seconds
    
    # Trading thresholds
    BUY_PROBABILITY_THRESHOLD: float = 0.3
    SELL_PROBABILITY_THRESHOLD: float = 0.5
    
    # Peak detection settings
    PEAK_EXIT_ENABLED: bool = True
    PEAK_MIN_PROFIT_PCT: float = 0.01  # 1% min profit before peak exit
    PEAK_RSI_THRESHOLD: float = 70.0  # Overbought level
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()


def validate_production_config():
    """
    Spec 18: Fail-fast validation
    Validates required env vars in production mode
    """
    if settings.APP_ENV == "production":
        required_vars = []
        
        if not settings.KITE_API_KEY:
            required_vars.append("KITE_API_KEY")
        if not settings.KITE_API_SECRET:
            required_vars.append("KITE_API_SECRET")
        if not settings.SMTP_USER:
            required_vars.append("SMTP_USER")
        if not settings.SMTP_PASS:
            required_vars.append("SMTP_PASS")
        if settings.KMS_MASTER_KEY == "dev-key-replace-in-production":
            required_vars.append("KMS_MASTER_KEY (must be changed from default)")
            
        if required_vars:
            print(f"ERROR: Missing required environment variables in production mode:")
            for var in required_vars:
                print(f"  - {var}")
            sys.exit(1)
    
    # Validate numeric ranges
    if not (0 < settings.RISK_PER_TRADE <= 0.1):
        print("ERROR: RISK_PER_TRADE must be between 0 and 0.1 (0-10%)")
        sys.exit(1)
    
    if not (0 < settings.MAX_DAILY_LOSS <= 0.2):
        print("ERROR: MAX_DAILY_LOSS must be between 0 and 0.2 (0-20%)")
        sys.exit(1)
    
    print(f"✓ Configuration validated for {settings.APP_ENV} environment")


# Run validation on import
validate_production_config()
