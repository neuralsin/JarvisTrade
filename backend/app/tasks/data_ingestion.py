"""
Data ingestion from Yahoo Finance ONLY:
- Data Source: Yahoo Finance (yfinance) for all historical data
- Kite API: Used EXCLUSIVELY for trade execution, NOT for data
- Supports: NSE & BSE exchanges
- Reason: Simplifies architecture, enables model training without Kite credentials
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, User
from app.utils.retry import retry_with_backoff
from app.utils.crypto import decrypt_text
from app.config import settings
from datetime import datetime, timedelta
import requests  # Direct HTTP for Yahoo Finance
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
    Fetch historical data from Yahoo Finance using DIRECT HTTP (no yfinance library).
    Uses the exact URL pattern: https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}.NS?interval={interval}&range={range}
    
    Args:
        symbol: Stock symbol (e.g., 'TATAELXSI', 'HAL', 'RELIANCE')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Interval (15m, 1h, 1d)
        exchange: Primary exchange ('NS' for NSE or 'BO' for BSE)
    
    Returns:
        DataFrame with OHLCV data, or None if all attempts fail
    """
    import requests
    import time
    from datetime import datetime
    
    # Build list of exchanges to try
    exchanges_to_try = []
    if exchange == 'NS':
        exchanges_to_try = ['NS', 'BO']  # Try NSE first, fallback to BSE
    elif exchange == 'BO':
        exchanges_to_try = ['BO', 'NS']  # Try BSE first, fallback to NSE
    else:
        exchanges_to_try = ['NS', 'BO']  # Default: try both
    
    # Calculate range from dates with Yahoo Finance limits
    # Yahoo Finance API has strict limits for intraday data:
    # - 1m: max 7 days
    # - 5m, 15m, 30m: max 60 days
    # - 1h: max 730 days (2 years)
    # - 1d, 1wk, 1mo: no practical limit
    
    INTERVAL_MAX_DAYS = {
        '1m': 7,
        '5m': 60,
        '15m': 60,
        '30m': 60,
        '1h': 730,
        '1d': 3650,  # 10 years
        '1wk': 3650,
        '1mo': 3650
    }
    
    max_days = INTERVAL_MAX_DAYS.get(interval, 60)
    
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        requested_days = (end_dt - start_dt).days
        
        # Cap to API limit
        if requested_days > max_days:
            logger.warning(
                f"Requested {requested_days} days for {interval} interval, "
                f"but Yahoo Finance limits to {max_days} days. Using {max_days}d."
            )
            range_param = f'{max_days}d'
        else:
            range_param = f'{requested_days}d'
    else:
        # Default ranges based on interval
        range_param = f'{max_days}d'
    
    last_error = None
    
    for exch in exchanges_to_try:
        yahoo_symbol = f"{symbol}.{exch}"
        
        # Construct the exact URL pattern that works
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={range_param}"
        
        # Retry loop for 429/rate limits
        for attempt in range(3):
            try:
                logger.info(f"Attempting Yahoo Finance: {url} (Try {attempt+1}/3)")
                
                # Direct HTTP request with User-Agent
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if we got valid data
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        
                        if 'timestamp' not in result or not result['timestamp']:
                            logger.warning(f"⚠ {yahoo_symbol} returned no timestamps")
                            last_error = "No timestamps in response"
                            break  # Don't retry, try next exchange
                        
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        # Build DataFrame
                        df_data = []
                        for i, ts in enumerate(timestamps):
                            # Skip rows with null values
                            if quotes['close'][i] is None:
                                continue
                            
                            df_data.append({
                                'timestamp': datetime.fromtimestamp(ts),
                                'open': float(quotes['open'][i]) if quotes['open'][i] else 0,
                                'high': float(quotes['high'][i]) if quotes['high'][i] else 0,
                                'low': float(quotes['low'][i]) if quotes['low'][i] else 0,
                                'close': float(quotes['close'][i]),
                                'volume': int(quotes['volume'][i]) if quotes['volume'][i] else 0
                            })
                        
                        if not df_data:
                            logger.warning(f"⚠ {yahoo_symbol} returned no valid data rows")
                            last_error = "No valid data rows"
                            break  # Try next exchange
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        
                        logger.info(f"✓ Successfully fetched {len(df)} candles from {yahoo_symbol}")
                        return df
                    else:
                        logger.warning(f"⚠ {yahoo_symbol} returned invalid response structure")
                        last_error = "Invalid response structure"
                        break  # Try next exchange
                        
                elif response.status_code == 429:
                    logger.warning(f"⚠ Rate limit 429 for {yahoo_symbol}. Waiting 5s...")
                    time.sleep(5)
                    continue  # Retry
                else:
                    logger.warning(f"⚠ {yahoo_symbol} HTTP status {response.status_code}")
                    last_error = f"HTTP {response.status_code}"
                    break  # Try next exchange
                
            except Exception as e:
                logger.warning(f"⚠ {yahoo_symbol} failed: {str(e)}")
                last_error = str(e)
                time.sleep(2)  # Small wait on error
                # Continue to retry
        
        # If we successfully returned already, we won't reach here
        # If we broke out of retry loop, continue to next exchange
    
    # All exchanges exhausted
    logger.error(
        f"❌ Failed to fetch {symbol} from Yahoo Finance. "
        f"Tried: {', '.join([f'{symbol}.{e}' for e in exchanges_to_try])}. "
        f"Last error: {last_error}"
    )
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
    Ingest historical data from Yahoo Finance and save to DB.
    
    NOTE: Kite API is NOT used for data fetching - only for trade execution.
    All historical data comes from Yahoo Finance regardless of Kite credential availability.
    This ensures models can train and data can be fetched without broker dependencies.
    """
    db = SessionLocal()
    
    try:
        
        total_candles = 0
        source_stats = {'yahoo': 0, 'failed': 0}  # Removed 'kite' - Yahoo only
        
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
                
                # Fetch from Yahoo Finance (exclusive data source)
                yahoo_exchange = 'NS' if exchange == 'NSE' else 'BO'
                df = fetch_historical_yahoo(symbol, start_date, end_date, interval, yahoo_exchange)
                
                if df is None:
                    logger.warning(f"Failed to fetch data for {symbol} from Yahoo Finance")
                    source_stats['failed'] += 1
                    continue
                
                source_used = 'yahoo'
                
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
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume'])
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
            f"Yahoo: {source_stats['yahoo']}, Failed: {source_stats['failed']}"
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
    Historical data fetching from Yahoo Finance (exclusive data source).
    
    Args:
        symbols: List of symbols (e.g., ['RELIANCE', 'TCS'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: '15m', '1h', '1d'
        exchange: 'NSE' or 'BSE'
    
    Note: Kite API is NOT used for data - only for trade execution.
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
            yesterday = datetime.utcnow().replace(tzinfo=None) - timedelta(days=1)
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
    Scheduled task to fetch recent data for tracked symbols.
    Uses Yahoo Finance exclusively for all data.
    """
    # Top NIFTY 50 stocks
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
    ]
    
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow().replace(tzinfo=None) - timedelta(days=5)).strftime('%Y-%m-%d')
    
    return fetch_historical_data(symbols, start_date, end_date, interval, 'NSE')

