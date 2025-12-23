"""
News sentiment analysis integration
"""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Fetch and analyze news sentiment using FinBERT
    """
    def __init__(self):
        # Load FinBERT model for financial sentiment
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.model.eval()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text
        
        Returns:
            dict: {positive: float, negative: float, neutral: float, label: str, score: float}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs[0].cpu().numpy()
        
        labels = ['positive', 'negative', 'neutral']
        sentiment_dict = {label: float(prob) for label, prob in zip(labels, probs)}
        
        # Get dominant sentiment
        max_label = max(sentiment_dict, key=sentiment_dict.get)
        max_score = sentiment_dict[max_label]
        
        # Convert to -1 to 1 scale
        sentiment_score = probs[0] - probs[1]  # positive - negative
        
        return {
            **sentiment_dict,
            'label': max_label,
            'score': float(sentiment_score)
        }
    
    def fetch_news_newsapi(self, symbol, api_key, days_back=7):
        """
        Fetch news from NewsAPI
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{symbol} stock",
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles
        else:
            logger.error(f"Failed to fetch news: {response.status_code}")
            return []
    
    def process_symbol_news(self, symbol, api_key):
        """
        Fetch and analyze news for a symbol
        
        Returns:
            dict: Aggregated sentiment metrics
        """
        articles = self.fetch_news_newsapi(symbol, api_key)
        
        if not articles:
            return {
                'sentiment_1d': 0.0,
                'sentiment_3d': 0.0,
                'sentiment_7d': 0.0,
                'news_count': 0
            }
        
        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append({
                'score': sentiment['score'],
                'published_at': article.get('publishedAt')
            })
        
        # Calculate rolling averages
        now = datetime.utcnow()
        
        def avg_sentiment_window(days):
            cutoff = now - timedelta(days=days)
            scores = [s['score'] for s in sentiments 
                     if datetime.fromisoformat(s['published_at'].replace('Z', '+00:00')) > cutoff]
            return sum(scores) / len(scores) if scores else 0.0
        
        return {
            'sentiment_1d': avg_sentiment_window(1),
            'sentiment_3d': avg_sentiment_window(3),
            'sentiment_7d': avg_sentiment_window(7),
            'news_count': len(articles)
        }


# Sentiment feature integration for dataframes
def integrate_sentiment_features(df, symbol, api_key=None):
    """
    Add sentiment features to existing dataframe
    
    This is called in feature_engineer.py
    """
    if not api_key:
        logger.warning("No NewsAPI key provided, skipping sentiment analysis")
        df['sentiment_1d'] = 0.0
        df['sentiment_3d'] = 0.0
        df['sentiment_7d'] = 0.0
        return df
    
    analyzer = NewsSentimentAnalyzer()
    sentiment_data = analyzer.process_symbol_news(symbol, api_key)
    
    # Add as constant features (same for all rows in this batch)
    for key, value in sentiment_data.items():
        df[key.replace('_', '_sentiment_')] = value
    
    return df
