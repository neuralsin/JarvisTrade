"""
Phase 1: Fresh Features Computation Pipeline

Real-time feature computation for paper trading execution.
Runs every 60 seconds to compute features for all tracked instruments.
Respects Yahoo Finance rate limits (2000 requests/hour).
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, Feature
from app.ml.feature_engineer import compute_features
from app.config import settings
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Rate limiting tracking
_last_yahoo_fetch = {}  # {symbol: timestamp}


def can_fetch_from_yahoo(symbol: str) -> bool:
    """
    Check if we can fetch from Yahoo Finance (rate limiting).
    Allow one fetch per symbol per FEATURE_CACHE_SECONDS.
    """
    if symbol not in _last_yahoo_fetch:
        return True
    
    elapsed = (datetime.utcnow() - _last_yahoo_fetch[symbol]).total_seconds()
    return elapsed >= settings.FEATURE_CACHE_SECONDS


@celery_app.task(bind=True)
def compute_fresh_features(self, instruments: list = None):
    """
    Phase 1: Compute fresh features for all active instruments.
    
    This task runs every 60 seconds to ensure features are fresh for signal checking.
    Uses caching to respect Yahoo Finance rate limits (2000 req/hour).
    
    Bug fix: Task Overlap - Uses Redis distributed lock to prevent race conditions
    when task is scheduled every 2 minutes but previous run hasn't finished.
    
    Args:
        instruments: Optional list of instrument symbols. If None, uses all instruments.
    
    Returns:
        Dict with computation stats
    """
    if not settings.FRESH_FEATURES_ENABLED:
        logger.info("Fresh features computation disabled in settings")
        return {"status": "disabled"}
    
    # Bug fix: Task Overlap - Use Redis distributed lock to prevent race conditions
    import redis
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        lock = redis_client.lock("fresh_features_lock", timeout=300, blocking=False)
        acquired = lock.acquire(blocking=False)
        
        if not acquired:
            logger.info("Another fresh_features task is running - skipping this execution")
            return {"status": "skipped", "reason": "lock_held"}
    except Exception as e:
        logger.warning(f"Redis lock acquisition failed: {e}. Proceeding without lock.")
        lock = None
        acquired = False
    
    db = SessionLocal()
    
    try:
        # Get instruments to process
        if instruments:
            instrument_objs = db.query(Instrument).filter(
                Instrument.symbol.in_(instruments)
            ).all()
        else:
            # Get all instruments
            instrument_objs = db.query(Instrument).all()
        
        if not instrument_objs:
            logger.warning("No instruments found to compute features")
            return {"status": "no_instruments"}
        
        logger.info(f"Computing fresh features for {len(instrument_objs)} instruments")
        
        stats = {
            'computed': 0,
            'cached': 0,
            'failed': 0,
            'rate_limited': 0
        }
        
        for idx, instrument in enumerate(instrument_objs):
            try:
                # Update progress
                if self:
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'status': f'Processing {instrument.symbol}',
                            'progress': int((idx / len(instrument_objs)) * 100)
                        }
                    )
                
                # Check if we can fetch (rate limiting)
                if not can_fetch_from_yahoo(instrument.symbol):
                    logger.debug(f"Skipping {instrument.symbol} - cached (rate limit)")
                    stats['cached'] += 1
                    continue
                
                # Get latest historical data (last 250 days for EMA200)
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=250)
                
                # First try to get from database
                # Bug fix #1: Use configurable interval instead of hardcoded '1d'
                interval = settings.DEFAULT_INTERVAL
                candles = db.query(HistoricalCandle).filter(
                    HistoricalCandle.instrument_id == instrument.id,
                    HistoricalCandle.timeframe == interval,
                    HistoricalCandle.ts_utc >= start_date
                ).order_by(HistoricalCandle.ts_utc).all()
                
                # Bug fix #1: Check freshness based on interval, not hardcoded 24 hours
                interval_seconds = {
                    '1m': 60, '5m': 300, '15m': 900, '30m': 1800, 
                    '1h': 3600, '1d': 86400, '1wk': 604800
                }.get(interval, 900)  # Default to 15m
                
                needs_fetch = True
                latest_ts = None  # Track for zombie fix
                if candles:
                    latest_candle = max(candles, key=lambda c: c.ts_utc)
                    latest_ts = latest_candle.ts_utc
                    seconds_old = (datetime.utcnow() - latest_candle.ts_utc.replace(tzinfo=None)).total_seconds()
                    if seconds_old < interval_seconds:
                        needs_fetch = False
                
                if needs_fetch:
                    # Fetch latest price using unified market data fetcher
                    from app.utils.market_data_fetcher import fetch_ohlcv, DataFetchError
                    
                    try:
                        yahoo_df = fetch_ohlcv(
                            symbol=instrument.symbol,
                            interval=interval,  # Bug fix #1: Use configured interval
                            days=5  # Just need recent data
                        )
                    except DataFetchError as e:
                        logger.warning(f"No data for {instrument.symbol}: {str(e)}")
                        stats['failed'] += 1
                        continue
                    
                    if yahoo_df is None or yahoo_df.empty:
                        logger.warning(f"No data from Yahoo for {instrument.symbol}")
                        stats['failed'] += 1
                        continue
                    
                    # Store latest candle in database
                    latest_row = yahoo_df.iloc[-1]
                    latest_ts = latest_row['ts_utc']  # Use column, not index
                    
                    # Check if this candle already exists
                    existing = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument.id,
                        HistoricalCandle.timeframe == interval,  # Bug fix #1
                        HistoricalCandle.ts_utc == latest_ts
                    ).first()
                    
                    if not existing:
                        new_candle = HistoricalCandle(
                            instrument_id=instrument.id,
                            timeframe=interval,  # Bug fix #1
                            ts_utc=latest_ts,
                            open=float(latest_row['open']),
                            high=float(latest_row['high']),
                            low=float(latest_row['low']),
                            close=float(latest_row['close']),
                            volume=float(latest_row['volume'])
                        )
                        db.add(new_candle)
                        db.flush()
                    
                    # Update rate limit tracker
                    _last_yahoo_fetch[instrument.symbol] = datetime.utcnow()
                
                # Now get all candles for feature computation
                candles = db.query(HistoricalCandle).filter(
                    HistoricalCandle.instrument_id == instrument.id,
                    HistoricalCandle.timeframe == interval,  # Bug fix #1
                    HistoricalCandle.ts_utc >= start_date
                ).order_by(HistoricalCandle.ts_utc).all()
                
                if len(candles) < 200:
                    logger.warning(f"Insufficient candles for {instrument.symbol}: {len(candles)}/200")
                    stats['failed'] += 1
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'ts_utc': c.ts_utc,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                } for c in candles])
                
                # Compute features
                df_with_features = compute_features(df, str(instrument.id), db)
                
                if df_with_features.empty:
                    logger.warning(f"Feature computation failed for {instrument.symbol}")
                    stats['failed'] += 1
                    continue
                
                # Get latest row with features
                latest_features = df_with_features.iloc[-1]
                
                # Create feature dict
                feature_dict = {
                    'returns_1': float(latest_features.get('returns_1', 0)),
                    'returns_5': float(latest_features.get('returns_5', 0)),
                    'ema_20': float(latest_features.get('ema_20', 0)),
                    'ema_50': float(latest_features.get('ema_50', 0)),
                    'ema_200': float(latest_features.get('ema_200', 0)),
                    'distance_from_ema200': float(latest_features.get('distance_from_ema200', 0)),
                    'rsi_14': float(latest_features.get('rsi_14', 50)),
                    'rsi_slope': float(latest_features.get('rsi_slope', 0)),
                    'atr_14': float(latest_features.get('atr_14', 0)),
                    'atr_percent': float(latest_features.get('atr_percent', 0)),
                    'volume_ratio': float(latest_features.get('volume_ratio', 1)),
                    'nifty_trend': int(latest_features.get('nifty_trend', 1)),
                    'vix': float(latest_features.get('vix', 20.0)),
                    'sentiment_1d': float(latest_features.get('sentiment_1d', 0)),
                    'sentiment_3d': float(latest_features.get('sentiment_3d', 0)),
                    'sentiment_7d': float(latest_features.get('sentiment_7d', 0)),
                    'close': float(latest_features.get('close', 0))
                }
                
                # Check if we already have a recent feature (< FEATURE_MAX_AGE_SECONDS)
                recent_feature = db.query(Feature).filter(
                    Feature.instrument_id == instrument.id,
                    Feature.ts_utc >= datetime.utcnow().replace(tzinfo=None) - timedelta(seconds=settings.FEATURE_MAX_AGE_SECONDS)
                ).first()
                
                # Bug fix #2 & #5: Only update features if new data was actually fetched
                if recent_feature:
                    if needs_fetch:
                        # Only update timestamp when we actually fetched new data
                        recent_feature.feature_json = feature_dict
                        recent_feature.ts_utc = datetime.utcnow()
                        # Track actual data age (Bug fix #5: Feature timestamp fraud)
                        if hasattr(recent_feature, 'source_data_ts'):
                            recent_feature.source_data_ts = latest_ts
                    else:
                        # Bug fix #2: Skip update if no new data - prevents zombie updates
                        logger.debug(f"Skipping feature update for {instrument.symbol} - no new data fetched")
                        stats['cached'] += 1
                        continue
                else:
                    # Create new feature record
                    feature_record = Feature(
                        instrument_id=instrument.id,
                        ts_utc=datetime.utcnow(),
                        feature_json=feature_dict,
                        target=None  # No label for real-time features
                    )
                    # Track source data timestamp if column exists
                    if hasattr(feature_record, 'source_data_ts'):
                        feature_record.source_data_ts = latest_ts
                    db.add(feature_record)
                
                db.commit()
                stats['computed'] += 1
                
                logger.info(f"✓ Fresh features computed for {instrument.symbol}")
                
            except Exception as e:
                logger.error(f"Error computing features for {instrument.symbol}: {str(e)}", exc_info=True)
                stats['failed'] += 1
                continue
        
        logger.info(
            f"Fresh features computation complete: "
            f"Computed={stats['computed']}, Cached={stats['cached']}, "
            f"Failed={stats['failed']}, RateLimited={stats['rate_limited']}"
        )
        
        return {
            "status": "success",
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Fresh features computation failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        db.close()
        # Bug fix: Task Overlap - Release the Redis lock
        if lock and acquired:
            try:
                lock.release()
            except Exception:
                pass  # Lock may have expired


@celery_app.task(bind=True)
def compute_features_for_stock(self, symbol: str):
    """
    Compute fresh features for a single stock (for manual triggering).
    
    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
    """
    return compute_fresh_features.apply(instruments=[symbol])
