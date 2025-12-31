"""
V3 API Endpoints for Dashboard
Provides V3-specific data: risk controls, sentiment, correlation, performance
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import asyncio
import logging

from app.db.database import get_db
from app.db.models import Trade, Instrument
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/v3", tags=["v3"])


# Try to import V3 modules
try:
    from app.ml.risk_controls import (
        get_override_controller,
        get_capital_feedback,
        get_missed_tracker
    )
    from app.trading.correlation_manager import get_correlation_manager
    from app.ml.performance_tracker import get_performance_tracker
    HAS_V3_CONTROLS = True
except ImportError:
    HAS_V3_CONTROLS = False
    logger.warning("V3 risk controls not available")

try:
    from app.data.news_sentiment import get_news_provider, sentiment_to_features
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False
    logger.info("News sentiment module not available")


@router.get("/status")
async def get_v3_status():
    """Get V3 module availability status"""
    return {
        "v3_enabled": HAS_V3_CONTROLS,
        "news_enabled": HAS_NEWS,
        "modules": {
            "risk_controls": HAS_V3_CONTROLS,
            "news_sentiment": HAS_NEWS,
            "correlation_manager": HAS_V3_CONTROLS,
            "performance_tracker": HAS_V3_CONTROLS
        }
    }


@router.get("/risk-controls")
async def get_risk_controls():
    """Get current risk control status"""
    if not HAS_V3_CONTROLS:
        return {"error": "V3 controls not available"}
    
    override = get_override_controller()
    capital = get_capital_feedback()
    missed = get_missed_tracker()
    
    return {
        "override": override.get_status(),
        "capital_curve": capital.get_status(),
        "missed_opportunities": missed.get_opportunity_cost()
    }


@router.post("/override-mode")
async def set_override_mode(mode: str, reason: str = ""):
    """Set trading override mode"""
    if not HAS_V3_CONTROLS:
        raise HTTPException(status_code=400, detail="V3 controls not available")
    
    override = get_override_controller()
    try:
        override.set_mode(mode, reason)
        return {"status": "ok", "mode": mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/correlation")
async def get_correlation_status():
    """Get current correlation/sector exposure"""
    if not HAS_V3_CONTROLS:
        return {"error": "V3 controls not available"}
    
    manager = get_correlation_manager()
    return {
        "positions": len(manager.open_positions),
        "sector_exposure": manager.get_sector_exposure(),
        "summary": manager.get_position_summary()
    }


@router.get("/performance")
async def get_performance_stats():
    """Get performance stats with decay weighting"""
    if not HAS_V3_CONTROLS:
        return {"error": "V3 controls not available"}
    
    tracker = get_performance_tracker()
    return tracker.to_dict()


@router.get("/sentiment")
async def get_sentiment(symbols: str):
    """Get news sentiment for symbols (comma-separated)"""
    if not HAS_NEWS:
        return {"error": "News sentiment not available", "sentiments": {}}
    
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        return {"sentiments": {}}
    
    provider = get_news_provider()
    sentiments = {}
    
    for symbol in symbol_list[:10]:  # Limit to 10 symbols
        try:
            result = await provider.get_sentiment(symbol)
            sentiments[symbol] = {
                "avg_sentiment": result.avg_sentiment,
                "sentiment_momentum": result.sentiment_momentum,
                "bullish_count": result.bullish_count,
                "bearish_count": result.bearish_count,
                "neutral_count": result.neutral_count,
                "news_count": result.news_count,
                "confidence": result.confidence,
                "top_headlines": result.top_headlines[:3]
            }
        except Exception as e:
            logger.error(f"Failed to get sentiment for {symbol}: {e}")
            sentiments[symbol] = {"error": str(e)}
    
    return {"sentiments": sentiments}


@router.get("/dashboard-stats")
async def get_v3_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get V3 stats for dashboard display.
    
    This extends the regular dashboard stats with V3-specific data.
    """
    stats = {
        "engine_version": "v2",  # Assume V2/V3 if this endpoint is called
    }
    
    # Get risk controls
    if HAS_V3_CONTROLS:
        override = get_override_controller()
        capital = get_capital_feedback()
        correlation = get_correlation_manager()
        tracker = get_performance_tracker()
        
        stats["override_mode"] = override.current_mode
        stats["current_drawdown"] = capital.current_drawdown * 100
        stats["risk_multiplier"] = capital.risk_multiplier
        stats["risk_reduction"] = (1 - capital.risk_multiplier) * 100
        stats["sector_exposure"] = correlation.get_sector_exposure()
        stats["regime_stats"] = tracker.get_regime_summary()
    
    return stats
