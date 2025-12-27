"""
Import TATAELXSI data directly into database
Fetches: 1) 15m interval for 60 days, 2) 1d interval for 2 years
"""
import sys
import os
sys.path.insert(0, '/app')

from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, Feature
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels

print("="*80)
print("TATAELXSI DATA IMPORT SCRIPT")
print("="*80)

db = SessionLocal()

try:
    # 1. Get or create TATAELXSI instrument
    instrument = db.query(Instrument).filter(Instrument.symbol == 'TATAELXSI').first()
    if not instrument:
        print("\n✓ Creating TATAELXSI instrument...")
        instrument = Instrument(
            symbol='TATAELXSI',
            exchange='NSE',
            instrument_type='EQ'
        )
        db.add(instrument)
        db.commit()
        db.refresh(instrument)
        print(f"  Created instrument ID: {instrument.id}")
    else:
        print(f"\n✓ Found existing TATAELXSI (ID: {instrument.id})")
    
    # 2. Fetch and import 15m data (60 days)
    print("\n" + "="*80)
    print("FETCHING 15m DATA (60 days)")
    print("="*80)
    
    try:
        ticker_15m = yf.Ticker("TATAELXSI.NS")
        df_15m = ticker_15m.history(period='60d', interval='15m')
        
        if df_15m.empty:
            print("✗ No 15m data available from Yahoo Finance")
        else:
            print(f"✓ Fetched {len(df_15m)} candles (15m)")
            
            # Delete existing 15m candles
            deleted = db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id,
                HistoricalCandle.timeframe == '15m'
            ).delete()
            print(f"  Deleted {deleted} existing 15m candles")
            
            # Import candles
            candles_added = 0
            for idx, row in df_15m.iterrows():
                candle = HistoricalCandle(
                    instrument_id=instrument.id,
                    ts_utc=idx.to_pydatetime(),
                    timeframe='15m',
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']) if row['Volume'] > 0 else 0
                )
                db.add(candle)
                candles_added += 1
            
            db.commit()
            print(f"✓ Imported {candles_added} candles (15m)")
            
            # Generate features
            print("  Computing features...")
            candle_data = [{
                'ts_utc': c.ts_utc,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id,
                HistoricalCandle.timeframe == '15m'
            ).order_by(HistoricalCandle.ts_utc).all()]
            
            df_candles = pd.DataFrame(candle_data)
            df_features = compute_features(df_candles, instrument_id=str(instrument.id), db_session=db)
            df_labeled = generate_labels(df_features)
            
            # Delete existing features
            db.query(Feature).filter(Feature.instrument_id == instrument.id).delete()
            
            # Import features
            features_added = 0
            for _, row in df_labeled.iterrows():
                feature_dict = row.to_dict()
                ts_utc = feature_dict.pop('ts_utc')
                target = feature_dict.pop('target', None)
                
                feat = Feature(
                    instrument_id=instrument.id,
                    ts_utc=ts_utc,
                    feature_json=feature_dict,
                    target=target
                )
                db.add(feat)
                features_added += 1
            
            db.commit()
            print(f"✓ Generated {features_added} features (15m)")
            
    except Exception as e:
        print(f"✗ Error with 15m data: {e}")
    
    # 3. Fetch and import 1d data (2 years = 730 days)
    print("\n" + "="*80)
    print("FETCHING 1d DATA (730 days / 2 years)")
    print("="*80)
    
    try:
        ticker_1d = yf.Ticker("TATAELXSI.NS")
        df_1d = ticker_1d.history(period='730d', interval='1d')
        
        if df_1d.empty:
            print("✗ No 1d data available from Yahoo Finance")
        else:
            print(f"✓ Fetched {len(df_1d)} candles (1d)")
            
            # Delete existing 1d candles
            deleted = db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id,
                HistoricalCandle.timeframe == '1d'
            ).delete()
            print(f"  Deleted {deleted} existing 1d candles")
            
            # Import candles
            candles_added = 0
            for idx, row in df_1d.iterrows():
                candle = HistoricalCandle(
                    instrument_id=instrument.id,
                    ts_utc=idx.to_pydatetime(),
                    timeframe='1d',
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']) if row['Volume'] > 0 else 0
                )
                db.add(candle)
                candles_added += 1
            
            db.commit()
            print(f"✓ Imported {candles_added} candles (1d)")
            print("  Note: Features will be generated during training")
            
    except Exception as e:
        print(f"✗ Error with 1d data: {e}")
    
    # 4. Summary
    print("\n" + "="*80)
    print("IMPORT SUMMARY")
    print("="*80)
    
    candles_15m = db.query(HistoricalCandle).filter(
        HistoricalCandle.instrument_id == instrument.id,
        HistoricalCandle.timeframe == '15m'
    ).count()
    
    candles_1d = db.query(HistoricalCandle).filter(
        HistoricalCandle.instrument_id == instrument.id,
        HistoricalCandle.timeframe == '1d'
    ).count()
    
    features_count = db.query(Feature).filter(Feature.instrument_id == instrument.id).count()
    
    print(f"✓ TATAELXSI candles (15m): {candles_15m}")
    print(f"✓ TATAELXSI candles (1d): {candles_1d}")
    print(f"✓ TATAELXSI features: {features_count}")
    print(f"\n✓ Ready for training!")
    print(f"  - Use interval='15m' for intraday model ({candles_15m} candles)")
    print(f"  - Use interval='1d' for daily model ({candles_1d} candles)")
    
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
