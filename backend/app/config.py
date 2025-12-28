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
    
    # News Sentiment (optional - sentiment defaults to 0.0 if not configured)
    NEWS_API_KEY: Optional[str] = None
    
    # Default stock for scheduled retraining
    DEFAULT_RETRAIN_STOCK: str = "RELIANCE"
    
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
    
    # Trading thresholds - FIXED: Raised from 0.3 to 0.65 (matches AUC useful region)
    BUY_PROBABILITY_THRESHOLD: float = 0.65
    SELL_PROBABILITY_THRESHOLD: float = 0.35
    
    # Model auto-activation - FIXED: Accuracy removed (invalid for imbalanced trading data)
    AUTO_ACTIVATE_MODELS: bool = True
    MODEL_MIN_AUC: float = 0.55  # Only AUC matters for ranking quality
    MODEL_MIN_PRECISION_AT_10: float = 0.60  # Precision@TopK must be tradable
    
    # Peak detection settings
    PEAK_EXIT_ENABLED: bool = True
    PEAK_MIN_PROFIT_PCT: float = 0.01  # 1% min profit before peak exit
    PEAK_RSI_THRESHOLD: float = 70.0  # Overbought level
    
    # =========================================================================
    # V2 DUAL-MODEL ARCHITECTURE CONFIGURATION
    # Set TRADING_ENGINE_VERSION to 'v2' to enable dual-model trading
    # =========================================================================
    
    # Engine Version Switch
    TRADING_ENGINE_VERSION: str = "v1"  # "v1" (legacy) or "v2" (dual-model)
    
    # V2 Regime Detection
    V2_ADX_TREND_THRESHOLD: float = 25.0
    V2_ADX_RANGE_THRESHOLD: float = 20.0
    V2_ATR_Z_VOLATILE_THRESHOLD: float = 1.0
    V2_ATR_Z_PANIC_THRESHOLD: float = 2.0
    V2_REGIME_PERSISTENCE_BARS: int = 5
    
    # V2 Model A (Direction Scout)
    V2_DIRECTION_CLASSES: int = 3  # 0=Neutral, 1=Long, 2=Short
    V2_DIRECTION_LOOKAHEAD_BARS: int = 8
    V2_DIRECTION_THRESHOLD_MULTIPLIER: float = 0.5
    
    # V2 Model B (Quality Gatekeeper)
    V2_QUALITY_TARGET_ATR_MULT: float = 1.5
    V2_QUALITY_STOP_ATR_MULT: float = 1.0
    V2_QUALITY_TIME_LIMIT_BARS: int = 24
    V2_QUALITY_SLIPPAGE_PCT: float = 0.0005
    
    # V2 Direction Confidence Thresholds (per regime)
    V2_DIRECTION_CONF_TREND_STABLE: float = 0.55
    V2_DIRECTION_CONF_TREND_VOLATILE: float = 0.60
    V2_DIRECTION_CONF_RANGE_QUIET: float = 0.65
    
    # V2 Quality Hurdles (per regime)
    V2_QUALITY_HURDLE_TREND_STABLE: float = 0.60
    V2_QUALITY_HURDLE_TREND_VOLATILE: float = 0.65
    V2_QUALITY_HURDLE_RANGE_QUIET: float = 0.68
    V2_QUALITY_HURDLE_CHOP_PANIC: float = 0.75
    
    # V2 Expectancy Engine
    V2_MIN_EXPECTANCY_R: float = 0.20
    V2_BOOTSTRAP_AVG_WIN_R: float = 1.5
    V2_BOOTSTRAP_AVG_LOSS_R: float = 1.0
    V2_BOOTSTRAP_WIN_RATE: float = 0.50
    
    # V2 Execution
    V2_MAX_POSITIONS: int = 3
    V2_RISK_PERCENT: float = 0.01
    V2_PANIC_RISK_REDUCTION: float = 0.5
    
    # V2 Exit Engine
    V2_TIME_STOP_BARS: int = 24
    V2_TIME_STOP_MIN_PROFIT_ATR: float = 0.2
    V2_REVERSAL_EXIT_CONF: float = 0.65
    V2_REVERSAL_MIN_BARS_HELD: int = 3
    V2_REVERSAL_PERSIST_BARS: int = 2
    
    # V2 Online Learning
    V2_TRADE_MEMORY_UPDATE_INTERVAL: int = 20
    V2_ROLLING_WINDOW_TRADES: int = 100
    V2_MODEL_KILL_THRESHOLD_TRADES: int = 50
    
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
