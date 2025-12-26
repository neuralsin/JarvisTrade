"""
Check data availability and run real training
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db.database import SessionLocal
from app.db.models import Instrument, Feature, HistoricalCandle
from sqlalchemy import func
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SessionLocal()

try:
    print("\n" + "="*70)
    print("DATA AVAILABILITY CHECK")
    print("="*70 + "\n")
    
    # Check all instruments
    instruments = db.query(Instrument).all()
    
    if not instruments:
        print("❌ No instruments found in database!")
        sys.exit(1)
    
    print(f"Found {len(instruments)} instruments:\n")
    
    best_instrument = None
    max_features = 0
    
    for inst in instruments:
        # Count features
        feature_count = db.query(Feature).filter(
            Feature.instrument_id == inst.id
        ).count()
        
        # Get date range
        date_range = db.query(
            func.min(Feature.ts_utc),
            func.max(Feature.ts_utc)
        ).filter(Feature.instrument_id == inst.id).first()
        
        # Count candles
        candle_count = db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == inst.id
        ).count()
        
        if feature_count > 0:
            min_date, max_date = date_range
            days = (max_date - min_date).days if min_date and max_date else 0
            
            print(f"📊 {inst.symbol}:")
            print(f"   Features: {feature_count}")
            print(f"   Candles:  {candle_count}")
            print(f"   Range:    {min_date.date() if min_date else 'N/A'} to {max_date.date() if max_date else 'N/A'} ({days} days)")
            print()
            
            if feature_count > max_features:
                max_features = feature_count
                best_instrument = inst
        else:
            print(f"❌ {inst.symbol}: NO DATA")
            print()
    
    if not best_instrument:
        print("\n❌ No instruments have feature data!")
        sys.exit(1)
    
    print("="*70)
    print(f"BEST INSTRUMENT FOR TRAINING: {best_instrument.symbol}")
    print(f"Features available: {max_features}")
    print("="*70)
    
    # Now attempt real training
    print("\n🔥 STARTING WET RUN (REAL TRAINING)...\n")
    
    from app.tasks.model_training import train_model
    
    # Get date range for this instrument
    date_range = db.query(
        func.min(Feature.ts_utc),
        func.max(Feature.ts_utc)
    ).filter(Feature.instrument_id == best_instrument.id).first()
    
    min_date, max_date = date_range
    
    print(f"Training {best_instrument.symbol}")
    print(f"Model: XGBoost (multi-class)")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Expected samples: ~{max_features}")
    print("\nStarting training task...")
    
    # Trigger training with correct parameter names  
    result = train_model.delay(
        model_name=f"wet_run_{best_instrument.symbol}",
        instrument_filter=best_instrument.symbol,  # Required parameter
        model_type='xgboost',
        start_date=min_date.strftime('%Y-%m-%d'),
        end_date=max_date.strftime('%Y-%m-%d')
    )
    
    print(f"\n✅ Training task submitted!")
    print(f"Task ID: {result.id}")
    print(f"\nMonitor progress in Celery logs:")
    print(f"docker logs -f jarvistrade_celery_worker")
    
finally:
    db.close()
