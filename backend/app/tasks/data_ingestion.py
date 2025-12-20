"""
Spec 2: Data ingestion Celery tasks
- NSE bhavcopy fetcher
- yfinance intraday data
- Rate limiting and retry logic
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle
from app.utils.retry import retry_with_backoff
from datetime import datetime, timedelta
import yfinance as yf
import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def fetch_eod_bhavcopy(self, date_str: str = None):
    """
    Spec 2.1: Fetch NSE bhavcopy (end-of-day data)
    
    Args:
        date_str: Date in YYYYMMDD format, defaults to yesterday
    """
    db = SessionLocal()
    
    try:
        if not date_str:
            yesterday = datetime.utcnow() - timedelta(days=1)
            date_str = yesterday.strftime("%Y%m%d")
        
        # NSE bhavcopy URL pattern
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        url = f"https://www.nseindia.com/content/historical/EQUITIES/{date_obj.year}/{date_obj.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
        
        logger.info(f"Fetching NSE bhavcopy for {date_str}")
        
        # Spec 2.1: Retry with exponential backoff
        response = _fetch_with_retry(url)
        
        # Extract CSV from ZIP
        with ZipFile(BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(csv_name))
        
        # Process and upsert data
        count = 0
        for _, row in df.iterrows():
            symbol = row['SYMBOL']
            
            # Upsert instrument
            instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
            if not instrument:
                instrument = Instrument(
                    symbol=symbol,
                    name=row.get('NAME', symbol),
                    exchange='NSE',
                    instrument_type='EQ'
                )
                db.add(instrument)
                db.flush()
            
            # Upsert candle (1d timeframe)
            ts_utc = datetime.strptime(date_str, "%Y%m%d").replace(hour=15, minute=30, tzinfo=None)
            
            candle = db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id,
                HistoricalCandle.timeframe == '1d',
                HistoricalCandle.ts_utc == ts_utc
            ).first()
            
            if not candle:
                candle = HistoricalCandle(
                    instrument_id=instrument.id,
                    timeframe='1d',
                    ts_utc=ts_utc,
                    open=float(row['OPEN']),
                    high=float(row['HIGH']),
                    low=float(row['LOW']),
                    close=float(row['CLOSE']),
                    volume=float(row['TOTTRDQTY'])
                )
                db.add(candle)
                count += 1
        
        db.commit()
        logger.info(f"NSE bhavcopy processed: {count} new candles")
        return {"status": "success", "date": date_str, "candles": count}
    
    except Exception as e:
        logger.error(f"Failed to fetch bhavcopy: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


@retry_with_backoff(max_attempts=5, initial_delay=2.0, exceptions=(requests.exceptions.RequestException,))
def _fetch_with_retry(url: str):
    """
    Spec 2.1: Fetch with retry and rate limiting
    """
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/zip'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


@celery_app.task(bind=True)
def fetch_intraday_yf(self, symbols: list, start_date: str, end_date: str):
    """
    Spec 2.2: Fetch intraday data using yfinance
    
    Args:
        symbols: List of symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    db = SessionLocal()
    
    try:
        total_candles = 0
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching intraday data for {symbol}")
                
                # Download 15min data
                df = yf.download(symbol, interval='15m', start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Get or create instrument
                clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                instrument = db.query(Instrument).filter(Instrument.symbol == clean_symbol).first()
                if not instrument:
                    instrument = Instrument(symbol=clean_symbol, exchange='NSE')
                    db.add(instrument)
                    db.flush()
                
                # Insert candles
                for ts, row in df.iterrows():
                    ts_utc = pd.Timestamp(ts).tz_convert('UTC').to_pydatetime()
                    
                    existing = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument.id,
                        HistoricalCandle.timeframe == '15m',
                        HistoricalCandle.ts_utc == ts_utc
                    ).first()
                    
                    if not existing:
                        candle = HistoricalCandle(
                            instrument_id=instrument.id,
                            timeframe='15m',
                            ts_utc=ts_utc,
                            open=float(row['Open']),
                            high=float(row['High']),
                            low=float(row['Low']),
                            close=float(row['Close']),
                            volume=float(row['Volume'])
                        )
                        db.add(candle)
                        total_candles += 1
                
                db.commit()
            
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        
        logger.info(f"Intraday fetch complete: {total_candles} new candles")
        return {"status": "success", "candles": total_candles}
    
    finally:
        db.close()


@celery_app.task(bind=True)
def fetch_recent_intraday(self):
    """
    Scheduled task to fetch recent intraday data for tracked symbols
    """
    # You would maintain a list of tracked symbols
    # For now, use top NIFTY 50 stocks as example
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    return fetch_intraday_yf(symbols, start_date, end_date)
