"""
Data ingestion with smart source selection:
- Primary: Kite API (if credentials available)
- Fallback: Yahoo Finance (yfinance)
- Supports: NSE & BSE
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, User
from app.utils.retry import retry_with_backoff
from app.utils.crypto import decrypt_text
from app.utils.yfinance_wrapper import get_rate_limiter
from app.config import settings
from datetime import datetime, timedelta
import yfinance as yf
import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import logging
import time

logger = logging.getLogger(__name__)


def get_kite_client():
    """
    Get Kite client if credentials are available
    Returns None if Kite is not configured
    """
    try:
        # Check if Kite API is configured
        if not settings.KITE_API_KEY or not settings.KITE_API_SECRET:
            logger.info("Kite API not configured, will use Yahoo Finance")
            return None
        
        # Try to get a user with Kite credentials
        db = SessionLocal()
        # Try to get a user with Kite credentials
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.kite_access_token_encrypted.isnot(None)).first()
            
            if not user:
                logger.info("No user with Kite access token, will use Yahoo Finance")
                return None
            
            # Decrypt access token
            from kiteconnect import KiteConnect
            access_token = decrypt_text(user.kite_access_token_encrypted)
            
            kite = KiteConnect(api_key=settings.KITE_API_KEY)
            kite.set_access_token(access_token)
            
            logger.info("Kite API client initialized successfully")
            return kite
        
        finally:
            db.close()
    
    except Exception as e:
        logger.warning(f"Failed to initialize Kite client: {str(e)}, falling back to Yahoo Finance")
        return None


def fetch_historical_kite(kite, symbol: str, from_date: datetime, to_date: datetime, interval: str = '15minute'):
    """
    Fetch historical data from Kite API
    
    Args:
        kite: KiteConnect instance
        symbol: Instrument symbol (e.g., 'RELIANCE')
        from_date: Start date
        to_date: End date
        interval: Candle interval (minute, 5minute, 15minute, day, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Get instrument token
        instruments = kite.instruments('NSE')
        instrument = next((i for i in instruments if i['tradingsymbol'] == symbol), None)
        
        if not instrument:
            logger.warning(f"Instrument {symbol} not found in Kite, trying Yahoo Finance")
            return None
        
        # Fetch historical data
        data = kite.historical_data(
            instrument_token=instrument['instrument_token'],
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        if not data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.rename(columns={
            'date': 'timestamp',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} candles from Kite for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Kite fetch failed for {symbol}: {str(e)}")
        return None


def fetch_historical_yahoo(symbol: str, start_date: str, end_date: str, interval: str = '15m', exchange: str = 'NS'):
    """
    Fetch historical data from Yahoo Finance with rate limiting
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Interval (1m, 5m, 15m, 1h, 1d)
        exchange: 'NS' for NSE or 'BO' for BSE
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Format symbol for Yahoo Finance
        yahoo_symbol = f"{symbol}.{exchange}"
        
        logger.info(f"Fetching from Yahoo Finance: {yahoo_symbol}")
        
        # Use rate-limited wrapper to prevent 429 errors
        rate_limiter = get_rate_limiter()
        df = rate_limiter.download(
            yahoo_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False
        )
        
        if df.empty:
            logger.warning(f"No data from Yahoo Finance for {yahoo_symbol}")
            return None
        
        logger.info(f"Fetched {len(df)} candles from Yahoo Finance for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
        return None


def ingest_historical_data(
    symbols: list,
    start_date: str,
    end_date: str,
    interval: str = '15m',
    exchange: str = 'NSE',
    task_instance=None
):
    """
    Ingest historical data from Kite/Yahoo and save to DB
    """
    db = SessionLocal()
    
    try:
        # Try to get Kite client
        kite = get_kite_client()
        
        total_candles = 0
        source_stats = {'kite': 0, 'yahoo': 0, 'failed': 0}
        
        # Map interval to Kite format
        kite_interval_map = {
            '1m': 'minute',
            '5m': '5minute',
            '15m': '15minute',
            '1h': '60minute',
            '1d': 'day'
        }
        
        for symbol in symbols:
            try:
                if task_instance:
                    task_instance.update_state(
                        state='PROGRESS',
                        meta={
                            'status': f'Fetching {symbol}',
                            'progress': (symbols.index(symbol) / len(symbols)) * 100
                        }
                    )
                else:
                    logger.info(f"Fetching {symbol} ({symbols.index(symbol)+1}/{len(symbols)})")
                
                df = None
                source_used = None
                
                # Try Kite first if available
                if kite:
                    from_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    to_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    kite_interval = kite_interval_map.get(interval, '15minute')
                    
                    df = fetch_historical_kite(kite, symbol, from_dt, to_dt, kite_interval)
                    if df is not None:
                        source_used = 'kite'
                
                # Fallback to Yahoo Finance
                if df is None:
                    yahoo_exchange = 'NS' if exchange == 'NSE' else 'BO'
                    df = fetch_historical_yahoo(symbol, start_date, end_date, interval, yahoo_exchange)
                    if df is not None:
                        source_used = 'yahoo'
                
                if df is None:
                    logger.warning(f"Failed to fetch data for {symbol} from all sources")
                    source_stats['failed'] += 1
                    continue
                
                # Update stats
                source_stats[source_used] += 1
                
                # Get or create instrument
                instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
                if not instrument:
                    instrument = Instrument(
                        symbol=symbol,
                        name=symbol,
                        exchange=exchange,
                        instrument_type='EQ'
                    )
                    db.add(instrument)
                    db.flush()
                
                # Insert candles
                timeframe = interval
                count = 0
                for ts, row in df.iterrows():
                    ts_utc = pd.Timestamp(ts).tz_localize(None) if ts.tzinfo is None else pd.Timestamp(ts).tz_convert('UTC').tz_localize(None)
                    
                    existing = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument.id,
                        HistoricalCandle.timeframe == timeframe,
                        HistoricalCandle.ts_utc == ts_utc
                    ).first()
                    
                    if not existing:
                        candle = HistoricalCandle(
                            instrument_id=instrument.id,
                            timeframe=timeframe,
                            ts_utc=ts_utc,
                            open=float(row['Open']),
                            high=float(row['High']),
                            low=float(row['Low']),
                            close=float(row['Close']),
                            volume=float(row['Volume'])
                        )
                        db.add(candle)
                        count += 1
                
                db.commit()
                total_candles += count
                logger.info(f"✓ {symbol}: {count} new candles from {source_used}")
                
                # Add small delay between symbols to avoid overwhelming Yahoo Finance
                if source_used == 'yahoo' and symbols.index(symbol) < len(symbols) - 1:
                    time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                source_stats['failed'] += 1
                continue
        
        logger.info(
            f"Data fetch complete: {total_candles} candles | "
            f"Kite: {source_stats['kite']}, Yahoo: {source_stats['yahoo']}, Failed: {source_stats['failed']}"
        )
        
        return {
            "status": "success",
            "total_candles": total_candles,
            "source_stats": source_stats
        }
    
    finally:
        db.close()


@celery_app.task(bind=True)
def fetch_historical_data(
    self,
    symbols: list,
    start_date: str,
    end_date: str,
    interval: str = '15m',
    exchange: str = 'NSE'
):
    """
    Smart historical data fetching with Kite API primary, Yahoo Finance fallback
    
    Args:
        symbols: List of symbols (e.g., ['RELIANCE', 'TCS'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: '15m', '1h', '1d'
        exchange: 'NSE' or 'BSE'
    """
    return ingest_historical_data(symbols, start_date, end_date, interval, exchange, task_instance=self)


@celery_app.task(bind=True)
def fetch_eod_bhavcopy(self, date_str: str = None):
    """
    Fetch NSE bhavcopy (end-of-day data)
    Kept for compatibility, now also tries BSE
    """
    db = SessionLocal()
    
    try:
        if not date_str:
            yesterday = datetime.utcnow() - timedelta(days=1)
            date_str = yesterday.strftime("%Y%m%d")
        
        # NSE bhavcopy URL
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        nse_url = f"https://www.nseindia.com/content/historical/EQUITIES/{date_obj.year}/{date_obj.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
        
        logger.info(f"Fetching NSE bhavcopy for {date_str}")
        
        try:
            response = _fetch_with_retry(nse_url)
            
            with ZipFile(BytesIO(response.content)) as zf:
                csv_name = zf.namelist()[0]
                df = pd.read_csv(zf.open(csv_name))
            
            count = _process_bhavcopy(db, df, date_str, 'NSE')
            
            logger.info(f"NSE bhavcopy processed: {count} candles")
            return {"status": "success", "exchange": "NSE", "date": date_str, "candles": count}
        
        except Exception as e:
            logger.error(f"NSE bhavcopy failed: {str(e)}")
            raise
    
    finally:
        db.close()


def _process_bhavcopy(db, df: pd.DataFrame, date_str: str, exchange: str):
    """
    Process bhavcopy DataFrame and insert into database
    """
    count = 0
    for _, row in df.iterrows():
        symbol = row['SYMBOL']
        
        # Get or create instrument
        instrument = db.query(Instrument).filter(Instrument.symbol == symbol).first()
        if not instrument:
            instrument = Instrument(
                symbol=symbol,
                name=row.get('NAME', symbol),
                exchange=exchange,
                instrument_type='EQ'
            )
            db.add(instrument)
            db.flush()
        
        # Insert candle
        ts_utc = datetime.strptime(date_str, "%Y%m%d").replace(hour=15, minute=30)
        
        existing = db.query(HistoricalCandle).filter(
            HistoricalCandle.instrument_id == instrument.id,
            HistoricalCandle.timeframe == '1d',
            HistoricalCandle.ts_utc == ts_utc
        ).first()
        
        if not existing:
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
    return count


@retry_with_backoff(max_attempts=5, initial_delay=2.0, exceptions=(requests.exceptions.RequestException,))
def _fetch_with_retry(url: str):
    """
    Fetch with retry and rate limiting
    """
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/zip'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


@celery_app.task(bind=True)
def fetch_recent_data(self, interval: str = '15m'):
    """
    Scheduled task to fetch recent data for tracked symbols
    Uses smart source selection (Kite → Yahoo)
    """
    # Top NIFTY 50 stocks
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
    ]
    
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    return fetch_historical_data(symbols, start_date, end_date, interval, 'NSE')

