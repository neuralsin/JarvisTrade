"""
Import TATAELXSI using DIRECT Yahoo Finance API (bypass yfinance)
Your URL works, so we'll use requests library directly!
"""
import sys
sys.path.insert(0, '/app')

from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, Feature
from datetime import datetime
import requests
import pandas as pd
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels

print("="*80)
print("TATAELXSI DATA IMPORT - DIRECT YAHOO API")
print("="*80)

db = SessionLocal()

try:
    # 1. Get or create instrument
    instrument = db.query(Instrument).filter(Instrument.symbol == 'TATAELXSI').first()
    if not instrument:
        print("\n✓ Creating TATAELXSI instrument...")
        instrument = Instrument(symbol='TATAELXSI', exchange='NSE', instrument_type='EQ')
        db.add(instrument)
        db.commit()
        db.refresh(instrument)
    else:
        print(f"\n✓ Found TATAELXSI (ID: {instrument.id})")
    
    # 2. Fetch 15m data (60 days) - DIRECT API CALL
    print("\n" + "="*80)
    print("FETCHING 15m DATA via Direct API")
    print("="*80)
    
    url_15m = "https://query1.finance.yahoo.com/v8/finance/chart/TATAELXSI.NS?interval=15m&range=60d"
    print(f"  URL: {url_15m}")
    
    response_15m = requests.get(url_15m)
    if response_15m.status_code == 200:
        data = response_15m.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        print(f"✓ Fetched {len(timestamps)} candles (15m)")
        
        # Delete existing
        deleted = db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id,
            HistoricalCandle.timeframe == '15m'
        ).delete()
        print(f"  Deleted {deleted} existing candles")
        
        # Import
        candles_added = 0
        for i, ts in enumerate(timestamps):
            if quotes['close'][i] is None:
                continue
            candle = HistoricalCandle(
                instrument_id=instrument.id,
                ts_utc=datetime.fromtimestamp(ts),
                timeframe='15m',
                open=float(quotes['open'][i]),
                high=float(quotes['high'][i]),
                low=float(quotes['low'][i]),
                close=float(quotes['close'][i]),
                volume=int(quotes['volume'][i]) if quotes['volume'][i] else 0
            )
            db.add(candle)
            candles_added += 1
        
        db.commit()
        print(f"✓ Imported {candles_added} candles")
        
        # Generate features
        print("  Computing features...")
        candle_data = [{
            'ts_utc': c.ts_utc, 'open': c.open, 'high': c.high,
            'low': c.low, 'close': c.close, 'volume': c.volume
        } for c in db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id,
            HistoricalCandle.timeframe == '15m'
        ).order_by(HistoricalCandle.ts_utc).all()]
        
        df_candles = pd.DataFrame(candle_data)
        df_features = compute_features(df_candles, instrument_id=str(instrument.id), db_session=db)
        df_labeled = generate_labels(df_features)
        
        # Delete old features
        db.query(Feature).filter(Feature.instrument_id == instrument.id).delete()
        
        # Import
        features_added = 0
        for _, row in df_labeled.iterrows():
            feat_dict = row.to_dict()
            ts_utc = feat_dict.pop('ts_utc')
            target = feat_dict.pop('target', None)
            feat = Feature(
                instrument_id=instrument.id,
                ts_utc=ts_utc,
                feature_json=feat_dict,
                target=target
            )
            db.add(feat)
            features_added += 1
        
        db.commit()
        print(f"✓ Generated {features_added} features")
    else:
        print(f"✗ API returned status {response_15m.status_code}")
    
    # 3. Fetch 1d data (365 days) - DIRECT API CALL
    print("\n" + "="*80)
    print("FETCHING 1d DATA via Direct API")
    print("="*80)
    
    url_1d = "https://query1.finance.yahoo.com/v8/finance/chart/TATAELXSI.NS?interval=1d&range=365d"
    print(f"  URL: {url_1d}")
    
    response_1d = requests.get(url_1d)
    if response_1d.status_code == 200:
        data = response_1d.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        print(f"✓ Fetched {len(timestamps)} candles (1d)")
        
        # Delete existing
        deleted = db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id,
            HistoricalCandle.timeframe == '1d'
        ).delete()
        print(f"  Deleted {deleted} existing candles")
        
        # Import
        candles_added = 0
        for i, ts in enumerate(timestamps):
            if quotes['close'][i] is None:
                continue
            candle = HistoricalCandle(
                instrument_id=instrument.id,
                ts_utc=datetime.fromtimestamp(ts),
                timeframe='1d',
                open=float(quotes['open'][i]),
                high=float(quotes['high'][i]),
                low=float(quotes['low'][i]),
                close=float(quotes['close'][i]),
                volume=int(quotes['volume'][i]) if quotes['volume'][i] else 0
            )
            db.add(candle)
            candles_added += 1
        
        db.commit()
        print(f"✓ Imported {candles_added} candles (1d)")
    else:
        print(f"✗ API returned status {response_1d.status_code}")
    
    # Summary
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
    
    features = db.query(Feature).filter(Feature.instrument_id == instrument.id).count()
    
    print(f"✓ TATAELXSI candles (15m): {candles_15m}")
    print(f"✓ TATAELXSI candles (1d): {candles_1d}")
    print(f"✓ TATAELXSI features: {features}")
    print(f"\n✓ READY FOR TRAINING!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
