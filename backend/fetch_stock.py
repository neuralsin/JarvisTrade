"""
Fetch data for ANY stock using direct Yahoo API
Usage: python fetch_stock.py SYMBOL [EXCHANGE]
Example: python fetch_stock.py HINDUNILVR NS
"""
import sys
import argparse
sys.path.insert(0, '/app')

from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, Feature
from datetime import datetime, timedelta
import requests
import pandas as pd
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels
import time

def fetch_and_import(symbol, exchange='NS'):
    print(f"\n{'='*60}")
    print(f"FETCHING DATA FOR {symbol}.{exchange}")
    print(f"{'='*60}")
    
    db = SessionLocal()
    
    try:
        # 1. Get/Create Instrument
        instrument = db.query(Instrument).filter(
            Instrument.symbol == symbol,
            Instrument.exchange == exchange
        ).first()
        
        if not instrument:
            print(f"Creating instrument {symbol}...")
            instrument = Instrument(symbol=symbol, exchange=exchange, instrument_type='EQ')
            db.add(instrument)
            db.commit()
            db.refresh(instrument)
        
        print(f"Instrument ID: {instrument.id}")
        
        # 2. Define fetches: (interval, days, label)
        fetches = [
            ('15m', 60, 'Intraday'),
            ('1d', 730, 'Daily')
        ]
        
        for interval, days, label in fetches:
            print(f"\n--- Fetching {label} ({interval}) ---")
            
            # Construct URL
            yahoo_symbol = f"{symbol}.{exchange}"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={days}d"
            print(f"URL: {url}")
            
            # Retry logic
            data = None
            for attempt in range(3):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        break
                    elif response.status_code == 429:
                        print(f"  Rate limit (429). Waiting 5s...")
                        time.sleep(5)
                    else:
                        print(f"  HTTP {response.status_code}")
                except Exception as e:
                    print(f"  Error: {e}")
            
            if not data:
                print(f"❌ Failed to fetch {interval} data")
                continue
                
            # Process data
            try:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                # Delete existing
                deleted = db.query(HistoricalCandle).filter(
                    HistoricalCandle.instrument_id == instrument.id,
                    HistoricalCandle.timeframe == interval
                ).delete()
                print(f"  Deleted {deleted} existing candles")
                
                # Import
                candles_added = 0
                for i, ts in enumerate(timestamps):
                    if quotes['close'][i] is None: continue
                    
                    candle = HistoricalCandle(
                        instrument_id=instrument.id,
                        ts_utc=datetime.fromtimestamp(ts),
                        timeframe=interval,
                        open=float(quotes['open'][i] or 0),
                        high=float(quotes['high'][i] or 0),
                        low=float(quotes['low'][i] or 0),
                        close=float(quotes['close'][i] or 0),
                        volume=int(quotes['volume'][i] or 0)
                    )
                    db.add(candle)
                    candles_added += 1
                
                db.commit()
                print(f"✓ Imported {candles_added} candles")
                
            except Exception as e:
                print(f"❌ Error processing data: {e}")
        
        # 3. Generate Features (based on 15m data likely, or 1d)
        # We'll generate features for whichever has data, prioritizing 15m
        print("\n--- Generating Features ---")
        
        # Check what we have
        count_15m = db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id, 
            HistoricalCandle.timeframe == '15m'
        ).count()
        
        target_interval = '15m' if count_15m > 0 else '1d'
        print(f"Using {target_interval} data for features...")
        
        candle_data = [{
            'ts_utc': c.ts_utc, 'open': c.open, 'high': c.high,
            'low': c.low, 'close': c.close, 'volume': c.volume
        } for c in db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id,
            HistoricalCandle.timeframe == target_interval
        ).order_by(HistoricalCandle.ts_utc).all()]
        
        if candle_data:
            df_candles = pd.DataFrame(candle_data)
            df_features = compute_features(df_candles, instrument_id=str(instrument.id), db_session=db)
            df_labeled = generate_labels(df_features)
            
            # Delete old features
            db.query(Feature).filter(Feature.instrument_id == instrument.id).delete()
            
            # Save new
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
            print("❌ No data for features")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_stock.py SYMBOL [EXCHANGE]")
        print("Defaulting to HINDUNILVR NS")
        fetch_and_import('HINDUNILVR', 'NS')
    else:
        sym = sys.argv[1]
        exch = sys.argv[2] if len(sys.argv) > 2 else 'NS'
        fetch_and_import(sym, exch)
