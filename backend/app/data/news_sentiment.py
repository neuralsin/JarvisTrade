"""
News Sentiment Module - V3 Feature

Fetch news and compute sentiment scores to enhance trading decisions.

Sources:
- Finnhub (free tier available)
- NewsAPI (backup)

Features:
- Sentiment scoring (positive/negative/neutral)
- News as feature for Model A/B
- Regime penalty if sentiment conflicts with signal
"""
import os
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Single news item with sentiment"""
    headline: str
    summary: str
    source: str
    timestamp: datetime
    url: str
    sentiment_score: float  # -1 to 1
    sentiment_label: Literal['positive', 'negative', 'neutral']
    relevance: float  # 0 to 1


@dataclass
class NewsSentimentResult:
    """Aggregated sentiment result for a symbol"""
    symbol: str
    timestamp: datetime
    news_count: int
    avg_sentiment: float  # -1 to 1
    sentiment_momentum: float  # Change in sentiment
    bullish_count: int
    bearish_count: int
    neutral_count: int
    top_headlines: List[str]
    confidence: float  # 0 to 1 based on news volume


class NewsSentimentProvider:
    """
    Fetch and score news sentiment.
    
    Uses Finnhub by default (free tier: 60 calls/min).
    """
    
    # Simple sentiment word lists (can be replaced with ML model)
    POSITIVE_WORDS = {
        'surge', 'jump', 'soar', 'gain', 'rally', 'bullish', 'strong',
        'profit', 'growth', 'beat', 'exceed', 'upgrade', 'buy', 'breakout',
        'record', 'high', 'positive', 'momentum', 'boom', 'recover',
        'up', 'rise', 'increase', 'advance', 'improve', 'outperform'
    }
    
    NEGATIVE_WORDS = {
        'crash', 'plunge', 'drop', 'fall', 'bearish', 'weak', 'loss',
        'decline', 'miss', 'downgrade', 'sell', 'warning', 'concern',
        'low', 'negative', 'slump', 'tumble', 'fear', 'risk', 'cut',
        'down', 'decrease', 'retreat', 'underperform', 'collapse'
    }
    
    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        newsapi_key: Optional[str] = None
    ):
        self.finnhub_key = finnhub_api_key or os.getenv('FINNHUB_API_KEY')
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        
        if not self.finnhub_key and not self.newsapi_key:
            logger.warning("⚠️ No news API keys configured. Sentiment will be unavailable.")
    
    async def fetch_finnhub_news(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch news from Finnhub API"""
        if not self.finnhub_key:
            return []
        
        from_date = from_date or (datetime.utcnow() - timedelta(days=7))
        to_date = to_date or datetime.utcnow()
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol.upper(),
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Finnhub API returned {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []
    
    def score_text(self, text: str) -> float:
        """
        Simple rule-based sentiment scoring.
        
        Returns score from -1 (very negative) to 1 (very positive).
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-z]+\b', text_lower))
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total
        return round(score, 3)
    
    def classify_sentiment(self, score: float) -> Literal['positive', 'negative', 'neutral']:
        """Classify sentiment based on score"""
        if score > 0.2:
            return 'positive'
        elif score < -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    async def get_sentiment(
        self,
        symbol: str,
        lookback_days: int = 3
    ) -> NewsSentimentResult:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE')
            lookback_days: Days of news to consider
            
        Returns:
            NewsSentimentResult with aggregated sentiment
        """
        from_date = datetime.utcnow() - timedelta(days=lookback_days)
        to_date = datetime.utcnow()
        
        # Fetch news
        news_items = await self.fetch_finnhub_news(symbol, from_date, to_date)
        
        if not news_items:
            return NewsSentimentResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                news_count=0,
                avg_sentiment=0.0,
                sentiment_momentum=0.0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                top_headlines=[],
                confidence=0.0
            )
        
        # Score each item
        scores = []
        bullish = 0
        bearish = 0
        neutral = 0
        headlines = []
        
        for item in news_items[:50]:  # Limit to 50 most recent
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            text = f"{headline} {summary}"
            
            score = self.score_text(text)
            scores.append(score)
            
            label = self.classify_sentiment(score)
            if label == 'positive':
                bullish += 1
            elif label == 'negative':
                bearish += 1
            else:
                neutral += 1
            
            if headline:
                headlines.append(headline)
        
        # Calculate momentum (recent vs older sentiment)
        if len(scores) >= 2:
            mid = len(scores) // 2
            recent_avg = sum(scores[:mid]) / mid
            older_avg = sum(scores[mid:]) / len(scores[mid:])
            momentum = recent_avg - older_avg
        else:
            momentum = 0.0
        
        # Confidence based on news volume
        confidence = min(1.0, len(news_items) / 10)  # Max confidence at 10+ news items
        
        return NewsSentimentResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            news_count=len(news_items),
            avg_sentiment=round(sum(scores) / len(scores), 3) if scores else 0.0,
            sentiment_momentum=round(momentum, 3),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            top_headlines=headlines[:5],
            confidence=round(confidence, 2)
        )
    
    def get_sentiment_sync(
        self,
        symbol: str,
        lookback_days: int = 3
    ) -> NewsSentimentResult:
        """Synchronous wrapper for get_sentiment"""
        return asyncio.run(self.get_sentiment(symbol, lookback_days))


# =============================================================================
# FEATURE EXTRACTION for ML Models
# =============================================================================

def sentiment_to_features(result: NewsSentimentResult) -> Dict[str, float]:
    """
    Convert sentiment result to features for ML models.
    
    Returns dict of features to add to feature set.
    """
    return {
        'news_sentiment_avg': result.avg_sentiment,
        'news_sentiment_momentum': result.sentiment_momentum,
        'news_bullish_ratio': result.bullish_count / max(1, result.news_count),
        'news_bearish_ratio': result.bearish_count / max(1, result.news_count),
        'news_confidence': result.confidence,
        'news_count': min(result.news_count, 20) / 20  # Normalized 0-1
    }


def sentiment_signal_conflict(
    sentiment: NewsSentimentResult,
    signal_direction: Literal[1, -1]  # 1=long, -1=short
) -> float:
    """
    Check if sentiment conflicts with signal direction.
    
    Returns conflict score (0 = no conflict, 1 = strong conflict).
    Used as a penalty in decision engine.
    """
    sentiment_direction = 1 if sentiment.avg_sentiment > 0.1 else (-1 if sentiment.avg_sentiment < -0.1 else 0)
    
    # No conflict if sentiment is neutral or matches signal
    if sentiment_direction == 0 or sentiment_direction == signal_direction:
        return 0.0
    
    # Conflict exists - scale by sentiment strength and confidence
    conflict = abs(sentiment.avg_sentiment) * sentiment.confidence
    return min(1.0, conflict)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

@lru_cache(maxsize=1)
def get_news_provider() -> NewsSentimentProvider:
    """Get singleton news provider instance"""
    return NewsSentimentProvider()


async def get_symbol_sentiment(symbol: str) -> Dict[str, float]:
    """
    Convenience function to get sentiment features for a symbol.
    
    Use in decision engine or feature engineering.
    """
    provider = get_news_provider()
    result = await provider.get_sentiment(symbol)
    return sentiment_to_features(result)
