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
    
    Args:
        instruments: Optional list of instrument symbols. If None, uses all instruments.
    
    Returns:
        Dict with computation stats
    """
    if not settings.FRESH_FEATURES_ENABLED:
        logger.info("Fresh features computation disabled in settings")
        return {"status": "disabled"}
    
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
                candles = db.query(HistoricalCandle).filter(
                    HistoricalCandle.instrument_id == instrument.id,
                    HistoricalCandle.timeframe == '1d',
                    HistoricalCandle.ts_utc >= start_date
                ).order_by(HistoricalCandle.ts_utc).all()
                
                # If we don't have recent data (within last 24 hours), fetch from Yahoo
                needs_fetch = True
                if candles:
                    latest_candle = max(candles, key=lambda c: c.ts_utc)
                    hours_old = (datetime.utcnow() - latest_candle.ts_utc.replace(tzinfo=None)).total_seconds() / 3600
                    if hours_old < 24:
                        needs_fetch = False
                
                if needs_fetch:
                    # Fetch latest price from Yahoo Finance
                    from app.tasks.data_ingestion import fetch_historical_yahoo
                    
                    yahoo_df = fetch_historical_yahoo(
                        symbol=instrument.symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        interval='1d',
                        exchange='NS'
                    )
                    
                    if yahoo_df is None or yahoo_df.empty:
                        logger.warning(f"No data from Yahoo for {instrument.symbol}")
                        stats['failed'] += 1
                        continue
                    
                    # Store latest candle in database
                    latest_row = yahoo_df.iloc[-1]
                    latest_ts = yahoo_df.index[-1]
                    
                    # Check if this candle already exists
                    existing = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument.id,
                        HistoricalCandle.timeframe == '1d',
                        HistoricalCandle.ts_utc == latest_ts
                    ).first()
                    
                    if not existing:
                        new_candle = HistoricalCandle(
                            instrument_id=instrument.id,
                            timeframe='1d',
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
                    HistoricalCandle.timeframe == '1d',
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
                
                if recent_feature:
                    # Update existing feature
                    recent_feature.feature_json = feature_dict
                    recent_feature.ts_utc = datetime.utcnow()
                else:
                    # Create new feature record
                    feature_record = Feature(
                        instrument_id=instrument.id,
                        ts_utc=datetime.utcnow(),
                        feature_json=feature_dict,
                        target=None  # No label for real-time features
                    )
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


@celery_app.task(bind=True)
def compute_features_for_stock(self, symbol: str):
    """
    Compute fresh features for a single stock (for manual triggering).
    
    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
    """
    return compute_fresh_features.apply(instruments=[symbol])
