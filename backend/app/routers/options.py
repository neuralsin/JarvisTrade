"""
Options trading API router
Calculate Greeks, analyze strategies
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from app.db.models import User
from app.routers.auth import get_current_user
from app.trading.options import OptionsGreeks, OptionsStrategy
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class GreeksRequest(BaseModel):
    spot: float
    strike: float
    time_to_expiry_days: int
    risk_free_rate: float = 0.06
    volatility: float = 0.20
    option_type: str = 'call'  # 'call' or 'put'


class StrategyRequest(BaseModel):
    strategy_type: str  # 'iron_condor', 'bull_call_spread', 'straddle'
    spot: float
    expiry_days: int = 30
    parameters: Optional[dict] = {}


@router.post("/greeks")
async def calculate_greeks(
    request: GreeksRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate option Greeks
    """
    T = request.time_to_expiry_days / 365
    
    greeks = OptionsGreeks.calculate_greeks(
        S=request.spot,
        K=request.strike,
        T=T,
        r=request.risk_free_rate,
        sigma=request.volatility,
        option_type=request.option_type
    )
    
    return {
        "status": "success",
        "input": {
            "spot": request.spot,
            "strike": request.strike,
            "expiry_days": request.time_to_expiry_days,
            "volatility": request.volatility * 100,  # Display as percentage
            "type": request.option_type
        },
        "greeks": greeks
    }


@router.post("/strategy/analyze")
async def analyze_strategy(
    request: StrategyRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze options strategy
    """
    strategy_func = {
        'iron_condor': OptionsStrategy.iron_condor,
        'bull_call_spread': OptionsStrategy.bull_call_spread,
        'straddle': OptionsStrategy.straddle
    }.get(request.strategy_type)
    
    if not strategy_func:
        return {
            "status": "error",
            "message": f"Unknown strategy: {request.strategy_type}"
        }
    
    # Call strategy function with parameters
    if request.strategy_type == 'iron_condor':
        result = strategy_func(
            spot=request.spot,
            wing_width=request.parameters.get('wing_width', 100),
            body_width=request.parameters.get('body_width', 50),
            expiry_days=request.expiry_days
        )
    elif request.strategy_type == 'bull_call_spread':
        result = strategy_func(
            spot=request.spot,
            width=request.parameters.get('width', 50),
            expiry_days=request.expiry_days
        )
    elif request.strategy_type == 'straddle':
        result = strategy_func(
            spot=request.spot,
            expiry_days=request.expiry_days
        )
    else:
        result = {}
    
    return {
        "status": "success",
        "strategy": result
    }


@router.post("/iv/calculate")
async def calculate_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    expiry_days: int,
    option_type: str = 'call',
    current_user: User = Depends(get_current_user)
):
    """
    Calculate implied volatility from market price
    """
    T = expiry_days / 365
    
    iv = OptionsGreeks.calculate_implied_volatility(
        market_price=market_price,
        S=spot,
        K=strike,
        T=T,
        option_type=option_type
    )
    
    return {
        "status": "success",
        "implied_volatility": iv,
        "implied_volatility_pct": iv * 100
    }


@router.get("/strategies/list")
async def list_strategies(
    current_user: User = Depends(get_current_user)
):
    """
    List available options strategies
    """
    return {
        "strategies": [
            {
                "name": "Iron Condor",
                "type": "iron_condor",
                "description": "Sell OTM call & put, buy further OTM call & put. Limited risk, limited profit.",
                "market_outlook": "Neutral (low volatility)",
                "parameters": ["wing_width", "body_width"]
            },
            {
                "name": "Bull Call Spread",
                "type": "bull_call_spread",
                "description": "Buy ATM call, sell OTM call. Limited risk, limited profit.",
                "market_outlook": "Moderately bullish",
                "parameters": ["width"]
            },
            {
                "name": "Long Straddle",
                "type": "straddle",
                "description": "Buy ATM call & put. Profits from big moves in either direction.",
                "market_outlook": "High volatility expected",
                "parameters": []
            }
        ]
    }
