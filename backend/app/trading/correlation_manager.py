"""
Correlation Manager - V3 Fix

Prevent stacking correlated risk by limiting exposure to similar positions.

Features:
- Sector caps (max trades per sector)
- Correlation matrix for open positions
- Dynamic correlation checking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Stock to sector mapping (extend as needed)
STOCK_SECTORS = {
    # Technology
    'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT', 'HCLTECH': 'IT',
    'LTIM': 'IT', 'MPHASIS': 'IT', 'PERSISTENT': 'IT', 'COFORGE': 'IT',
    
    # Banking
    'HDFCBANK': 'BANKING', 'ICICIBANK': 'BANKING', 'SBIN': 'BANKING',
    'AXISBANK': 'BANKING', 'KOTAKBANK': 'BANKING', 'INDUSINDBK': 'BANKING',
    'BANKBARODA': 'BANKING', 'PNB': 'BANKING', 'IDFCFIRSTB': 'BANKING',
    
    # Finance
    'BAJFINANCE': 'FINANCE', 'BAJAJFINSV': 'FINANCE', 'HDFC': 'FINANCE',
    'CHOLAFIN': 'FINANCE', 'MUTHOOTFIN': 'FINANCE', 'M&MFIN': 'FINANCE',
    
    # Auto
    'TATAMOTORS': 'AUTO', 'M&M': 'AUTO', 'MARUTI': 'AUTO', 'BAJAJ-AUTO': 'AUTO',
    'HEROMOTOCO': 'AUTO', 'EICHERMOT': 'AUTO', 'ASHOKLEY': 'AUTO',
    
    # Pharma
    'SUNPHARMA': 'PHARMA', 'DRREDDY': 'PHARMA', 'CIPLA': 'PHARMA',
    'DIVISLAB': 'PHARMA', 'AUROPHARMA': 'PHARMA', 'BIOCON': 'PHARMA',
    
    # Energy
    'RELIANCE': 'ENERGY', 'ONGC': 'ENERGY', 'BPCL': 'ENERGY', 'IOC': 'ENERGY',
    'GAIL': 'ENERGY', 'HINDPETRO': 'ENERGY', 'POWERGRID': 'ENERGY',
    
    # Metals
    'TATASTEEL': 'METALS', 'JSWSTEEL': 'METALS', 'HINDALCO': 'METALS',
    'VEDL': 'METALS', 'SAIL': 'METALS', 'NMDC': 'METALS', 'COALINDIA': 'METALS',
    
    # Consumer
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
    'DABUR': 'FMCG', 'MARICO': 'FMCG', 'GODREJCP': 'FMCG',
    
    # Telecom
    'BHARTIARTL': 'TELECOM', 'IDEA': 'TELECOM',
    
    # Infra/Construction
    'LT': 'INFRA', 'ADANIPORTS': 'INFRA', 'ULTRACEMCO': 'INFRA',
    'GRASIM': 'INFRA', 'SHREECEM': 'INFRA', 'ACC': 'INFRA',
}


@dataclass
class PositionRisk:
    """Risk info for an open position"""
    symbol: str
    sector: str
    direction: int  # 1=long, -1=short
    risk_amount: float
    entry_time: pd.Timestamp


class CorrelationManager:
    """
    Manage position correlations to prevent stacking risk.
    
    Rules:
    1. Max 2 positions per sector
    2. Max 50% of capital in correlated positions
    3. Long + Short in same sector reduces net risk
    """
    
    def __init__(
        self,
        max_per_sector: int = 2,
        max_sector_exposure_pct: float = 0.5,
        max_total_correlation: float = 0.7
    ):
        self.max_per_sector = max_per_sector
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_total_correlation = max_total_correlation
        
        self.open_positions: Dict[str, PositionRisk] = {}
        self.sector_mapping = STOCK_SECTORS.copy()
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        # Clean symbol (remove .NS, .BSE etc)
        clean = symbol.upper().split('.')[0]
        return self.sector_mapping.get(clean, 'OTHER')
    
    def add_position(
        self,
        symbol: str,
        direction: int,
        risk_amount: float
    ) -> None:
        """Register a new open position"""
        sector = self.get_sector(symbol)
        self.open_positions[symbol] = PositionRisk(
            symbol=symbol,
            sector=sector,
            direction=direction,
            risk_amount=risk_amount,
            entry_time=pd.Timestamp.now()
        )
        logger.info(f"Added position: {symbol} ({sector}), {len(self.open_positions)} total")
    
    def remove_position(self, symbol: str) -> None:
        """Remove a closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.info(f"Removed position: {symbol}, {len(self.open_positions)} remaining")
    
    def get_sector_exposure(self) -> Dict[str, Dict]:
        """Get current exposure by sector"""
        exposure = defaultdict(lambda: {'count': 0, 'net_direction': 0, 'total_risk': 0.0})
        
        for pos in self.open_positions.values():
            exposure[pos.sector]['count'] += 1
            exposure[pos.sector]['net_direction'] += pos.direction
            exposure[pos.sector]['total_risk'] += pos.risk_amount
        
        return dict(exposure)
    
    def can_add_position(
        self,
        symbol: str,
        direction: int,
        risk_amount: float,
        total_capital: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added without violating correlation rules.
        
        Args:
            symbol: Stock symbol
            direction: 1=long, -1=short
            risk_amount: Risk in this position
            total_capital: Total account capital
            
        Returns:
            (allowed, reason)
        """
        sector = self.get_sector(symbol)
        exposure = self.get_sector_exposure()
        
        # Rule 1: Max positions per sector
        sector_exp = exposure.get(sector, {'count': 0, 'net_direction': 0, 'total_risk': 0.0})
        
        if sector_exp['count'] >= self.max_per_sector:
            # Exception: opposite direction reduces risk
            if sector_exp['net_direction'] * direction < 0:
                logger.info(f"Allowing opposite position in {sector} (hedging)")
            else:
                return False, f"MAX_SECTOR_POSITIONS ({sector}: {sector_exp['count']}/{self.max_per_sector})"
        
        # Rule 2: Max sector exposure as % of capital
        new_sector_risk = sector_exp['total_risk'] + risk_amount
        if new_sector_risk / total_capital > self.max_sector_exposure_pct:
            return False, f"SECTOR_EXPOSURE_LIMIT ({sector}: {new_sector_risk/total_capital*100:.1f}%)"
        
        # Rule 3: Check total correlated exposure
        # For now, sum up same-direction positions
        total_correlated_risk = sum(
            p.risk_amount for p in self.open_positions.values()
            if p.direction == direction
        )
        total_correlated_risk += risk_amount
        
        if total_correlated_risk / total_capital > self.max_total_correlation:
            return False, f"TOTAL_CORRELATION_LIMIT ({total_correlated_risk/total_capital*100:.1f}%)"
        
        return True, "OK"
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions for display"""
        exposure = self.get_sector_exposure()
        
        return {
            'total_positions': len(self.open_positions),
            'sectors': {
                sector: {
                    'count': info['count'],
                    'net_direction': 'LONG' if info['net_direction'] > 0 else 'SHORT' if info['net_direction'] < 0 else 'NEUTRAL',
                    'total_risk': info['total_risk']
                }
                for sector, info in exposure.items()
            }
        }
    
    def calculate_portfolio_correlation(
        self,
        price_history: Dict[str, pd.Series],
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation between open positions.
        
        Args:
            price_history: Dict of symbol -> price series
            window: Rolling window for correlation
            
        Returns:
            Correlation matrix DataFrame
        """
        symbols = list(self.open_positions.keys())
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Build returns DataFrame
        returns_data = {}
        for symbol in symbols:
            if symbol in price_history:
                returns_data[symbol] = price_history[symbol].pct_change()
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()


# Singleton instance
_correlation_manager: Optional[CorrelationManager] = None


def get_correlation_manager() -> CorrelationManager:
    """Get singleton correlation manager"""
    global _correlation_manager
    if _correlation_manager is None:
        _correlation_manager = CorrelationManager()
    return _correlation_manager


def check_correlation_before_trade(
    symbol: str,
    direction: int,
    risk_amount: float,
    total_capital: float
) -> Tuple[bool, str]:
    """
    Convenience function to check if a trade is allowed.
    
    Use this in decision engine before executing trades.
    """
    manager = get_correlation_manager()
    return manager.can_add_position(symbol, direction, risk_amount, total_capital)
