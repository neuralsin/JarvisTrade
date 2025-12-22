"""
Spec 17: Data seeding script
Fetch historical data and compute features
"""
import argparse
from app.db.database import SessionLocal
from app.db.models import Instrument, HistoricalCandle, Feature, User
from app.ml.feature_engineer import compute_features, extract_feature_vector
from app.ml.labeler import generate_labels
from app.tasks.data_ingestion import ingest_historical_data
from datetime import datetime, timedelta
from passlib.context import CryptContext
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
 
def create_superuser(email: str, password: str):
    """
    Create admin user
    """
    db = SessionLocal()
    
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            logger.info(f"User {email} already exists")
            return
        
        user = User(
            email=email,
            password_hash=pwd_context.hash(password)
        )
        db.add(user)
        db.commit()
        
        logger.info(f"Created user: {email}")
    finally:
        db.close()


def seed_instruments():
    """
    Seed initial instruments (NIFTY 50 stocks)
    """
    db = SessionLocal()
    
    # Top NIFTY stocks
    nifty_stocks = [
        ('RELIANCE', 'Reliance Industries'),
        ('TCS', 'Tata Consultancy Services'),
        ('INFY', 'Infosys'),
        ('HDFCBANK', 'HDFC Bank'),
        ('ICICIBANK', 'ICICI Bank'),
        ('HINDUNILVR', 'Hindustan Unilever'),
        ('ITC', 'ITC'),
        ('SBIN', 'State Bank of India'),
        ('BHARTI AIRTEL', 'Bharti Airtel'),
        ('KOTAKBANK', 'Kotak Mahindra Bank')
    ]
    
    try:
        for symbol, name in nifty_stocks:
            existing = db.query(Instrument).filter(Instrument.symbol == symbol).first()
            if not existing:
                inst = Instrument(
                    symbol=symbol,
                    name=name,
                    exchange='NSE',
                    instrument_type='EQ'
                )
                db.add(inst)
        
        db.commit()
        logger.info(f"Seeded {len(nifty_stocks)} instruments")
    finally:
        db.close()

def fetch_historical_data(years: int = 2):
    """
    Fetch historical data for seeded instruments
    """
    db = SessionLocal()
    
    try:
        instruments = db.query(Instrument).all()
        symbols = [i.symbol for i in instruments]  # ingest_historical_data handles exchange specifics
        
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        # Yahoo Finance limits 15m data to last 60 days
        start_date = (datetime.utcnow() - timedelta(days=59)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching 59 days of 15m data for {len(symbols)} symbols (Yahoo limit)")
        
        # Fetch intraday data using the extracted logic
        ingest_historical_data(symbols, start_date, end_date, interval='15m', exchange='NSE')
        
        logger.info("Historical data fetch complete")
    finally:
        db.close()


def compute_and_store_features():
    """
    Compute features and labels for all instruments
    """
    db = SessionLocal()
    
    try:
        instruments = db.query(Instrument).all()
        
        for instrument in instruments:
            logger.info(f"Computing features for {instrument.symbol}")
            
            # Get candles for this instrument
            candles = db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id,
                HistoricalCandle.timeframe == '15m'
            ).order_by(HistoricalCandle.ts_utc).all()
            
            if len(candles) < 250:  # Need enough data for EMA200
                logger.warning(f"Not enough data for {instrument.symbol}, skipping")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'ts_utc': c.ts_utc,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                }
                for c in candles
            ])
            
            # Compute features
            df = compute_features(df)
            
            # Generate labels
            df = generate_labels(df)
            
            # Drop rows with NaN values (e.g. from EWA/rolling calculations)
            df.dropna(inplace=True)
            
            # Store features
            for _, row in df.iterrows():
                if pd.isna(row['target']):
                    continue
                
                feature_vec = extract_feature_vector(row)
                
                existing = db.query(Feature).filter(
                    Feature.instrument_id == instrument.id,
                    Feature.ts_utc == row['ts_utc']
                ).first()
                
                if not existing:
                    feat = Feature(
                        instrument_id=instrument.id,
                        ts_utc=row['ts_utc'],
                        feature_json=feature_vec,
                        target=int(row['target'])
                    )
                    db.add(feat)
            
            db.commit()
            logger.info(f"Stored features for {instrument.symbol}")
    
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Seed JarvisTrade database')
    parser.add_argument('--years', type=int, default=2, help='Years of historical data to fetch')
    parser.add_argument('--email', type=str, default='admin@jarvistrade.com', help='Admin email')
    parser.add_argument('--password', type=str, default='admin123', help='AdminPassword')
    
    args = parser.parse_args()
    
    logger.info("Starting database seeding...")
    
    # Create superuser
    create_superuser(args.email, args.password)
    
    # Seed instruments
    seed_instruments()
    
    # Fetch historical data
    fetch_historical_data(years=args.years)
    
    # Compute features
    compute_and_store_features()
    
    logger.info("Seeding complete!")


if __name__ == '__main__':
    main()
