"""
Sentiment API routes - view sentiment data and trigger analysis
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import User, NewsSentiment, Instrument
from app.routers.auth import get_current_user
from typing import Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class SentimentResponse(BaseModel):
    symbol: str
    instrument_id: str
    ts_utc: str
    sentiment_1d: float
    sentiment_3d: float
    sentiment_7d: float
    news_count: int
    source: str


@router.get("/")
async def get_all_sentiment(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get latest sentiment data for all instruments
    """
    # Get latest sentiment for each instrument
    from sqlalchemy import func
    
    subquery = db.query(
        NewsSentiment.instrument_id,
        func.max(NewsSentiment.ts_utc).label('max_ts')
    ).group_by(NewsSentiment.instrument_id).subquery()
    
    sentiments = db.query(NewsSentiment, Instrument).join(
        Instrument, NewsSentiment.instrument_id == Instrument.id
    ).join(
        subquery,
        (NewsSentiment.instrument_id == subquery.c.instrument_id) &
        (NewsSentiment.ts_utc == subquery.c.max_ts)
    ).limit(limit).all()
    
    results = []
    for sentiment, instrument in sentiments:
        results.append({
            "symbol": instrument.symbol,
            "instrument_id": str(sentiment.instrument_id),
            "ts_utc": sentiment.ts_utc.isoformat() + "Z",
            "sentiment_1d": float(sentiment.sentiment_1d or 0),
            "sentiment_3d": float(sentiment.sentiment_3d or 0),
            "sentiment_7d": float(sentiment.sentiment_7d or 0),
            "news_count": sentiment.news_count or 0,
            "source": sentiment.source
        })
    
    return {"sentiments": results}


@router.get("/{symbol}")
async def get_sentiment_by_symbol(
    symbol: str,
    days_back: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get sentiment data for a specific symbol
    """
    instrument = db.query(Instrument).filter(Instrument.symbol == symbol.upper()).first()
    
    if not instrument:
        return {"error": f"Instrument {symbol} not found"}
    
    cutoff_date = datetime.utcnow().replace(tzinfo=None) - timedelta(days=days_back)
    
    sentiments = db.query(NewsSentiment).filter(
        NewsSentiment.instrument_id == instrument.id,
        NewsSentiment.ts_utc >= cutoff_date
    ).order_by(NewsSentiment.ts_utc.desc()).all()
    
    results = []
    for sentiment in sentiments:
        results.append({
            "symbol": symbol.upper(),
            "ts_utc": sentiment.ts_utc.isoformat() + "Z",
            "sentiment_1d": float(sentiment.sentiment_1d or 0),
            "sentiment_3d": float(sentiment.sentiment_3d or 0),
            "sentiment_7d": float(sentiment.sentiment_7d or 0),
            "news_count": sentiment.news_count or 0,
            "source": sentiment.source
        })
    
    return {
        "symbol": symbol.upper(),
        "instrument_id": str(instrument.id),
        "data": results
    }


@router.post("/fetch")
async def trigger_sentiment_fetch(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger news sentiment analysis task
    """
    from app.tasks.news_sentiment import fetch_and_analyze_news
    
    task = fetch_and_analyze_news.delay()
    
    logger.info(f"Sentiment fetch task queued: {task.id}")
    
    return {
        "status": "queued",
        "task_id": task.id,
        "message": "Sentiment analysis task has been queued"
    }
