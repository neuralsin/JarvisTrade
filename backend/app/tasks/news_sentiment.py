"""
News sentiment fetching Celery task
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Instrument
from app.ml.sentiment_analyzer import NewsSentimentAnalyzer
from app.config import settings
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def fetch_and_analyze_news(self):
    """
    Fetch news and analyze sentiment for top instruments
    Scheduled to run daily
    """
    db = SessionLocal()
    
    try:
        # Check if NEWS_API_KEY is configured
        news_api_key = getattr(settings, 'NEWS_API_KEY', None)
        
        if not news_api_key:
            logger.warning("NEWS_API_KEY not configured, skipping news sentiment analysis")
            return {"status": "skipped", "reason": "API key not configured"}
        
        # Get top instruments (e.g., NIFTY 50)
        instruments = db.query(Instrument).limit(20).all()
        
        if not instruments:
            logger.warning("No instruments found for news analysis")
            return {"status": "skipped", "reason": "No instruments"}
        
        # Initialize sentiment analyzer
        analyzer = NewsSentimentAnalyzer()
        
        results = []
        for instrument in instruments:
            try:
                self.update_state(
                    state='PROGRESS',
                    meta={'status': f'Analyzing {instrument.symbol}', 'progress': len(results) / len(instruments) * 100}
                )
                
                sentiment_data = analyzer.process_symbol_news(instrument.symbol, news_api_key)
                
                # Store sentiment data in database
                from app.db.models import NewsSentiment
                from datetime import datetime
                
                ts_utc = datetime.utcnow()
                
                # Check if sentiment already exists for this timestamp
                existing = db.query(NewsSentiment).filter(
                    NewsSentiment.instrument_id == instrument.id,
                    NewsSentiment.ts_utc >= ts_utc.replace(hour=0, minute=0, second=0),
                    NewsSentiment.source == 'newsapi'
                ).first()
                
                if existing:
                    # Update existing record
                    existing.sentiment_1d = sentiment_data['sentiment_1d']
                    existing.sentiment_3d = sentiment_data['sentiment_3d']
                    existing.sentiment_7d = sentiment_data['sentiment_7d']
                    existing.news_count = sentiment_data['news_count']
                else:
                    # Create new record
                    sentiment_record = NewsSentiment(
                        instrument_id=instrument.id,
                        ts_utc=ts_utc,
                        sentiment_1d=sentiment_data['sentiment_1d'],
                        sentiment_3d=sentiment_data['sentiment_3d'],
                        sentiment_7d=sentiment_data['sentiment_7d'],
                        news_count=sentiment_data['news_count'],
                        source='newsapi'
                    )
                    db.add(sentiment_record)
                
                db.commit()
                
                results.append({
                    'symbol': instrument.symbol,
                    'sentiment': sentiment_data
                })
                
                logger.info(f"News sentiment for {instrument.symbol}: {sentiment_data['sentiment_7d']:.3f} (saved to DB)")
            
            except Exception as e:
                logger.error(f"Failed to analyze {instrument.symbol}: {str(e)}")
                db.rollback()
                continue
        
        logger.info(f"News sentiment analysis complete for {len(results)} symbols")
        
        return {
            "status": "success",
            "analyzed": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"News sentiment task failed: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()


@celery_app.task
def update_sentiment_features():
    """
    Update feature table with latest sentiment data
    Run after news fetching
    """
    # This would update the features table with sentiment columns
    # For now, just a placeholder
    logger.info("Sentiment features update task triggered")
    return {"status": "success"}
