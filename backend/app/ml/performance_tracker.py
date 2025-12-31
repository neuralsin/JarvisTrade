"""
Performance Tracker with Exponential Decay - V3 Fix

Track trading performance with recency weighting.

Old data should matter less than recent data.

Features:
- Exponential decay on statistics
- Per-regime tracking
- Per-direction tracking (long/short)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record for tracking"""
    timestamp: datetime
    symbol: str
    direction: int  # 1=long, -1=short
    r_multiple: float  # Actual R achieved
    regime: str
    pnl: float
    entry_price: float
    exit_price: float
    bars_held: int


@dataclass
class DecayingStats:
    """Statistics with exponential decay weighting"""
    
    # Raw data
    values: deque = field(default_factory=lambda: deque(maxlen=500))
    weights: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Decay parameters
    half_life: int = 30  # Half-life in number of trades
    
    def add(self, value: float, weight: float = 1.0) -> None:
        """Add a new value with default weight of 1.0"""
        self.values.append(value)
        self.weights.append(weight)
        self._decay_weights()
    
    def _decay_weights(self) -> None:
        """Apply exponential decay to all weights"""
        decay = 0.5 ** (1 / self.half_life)
        for i in range(len(self.weights) - 1):
            self.weights[i] *= decay
    
    @property
    def weighted_mean(self) -> float:
        """Get weighted mean with decay"""
        if not self.values:
            return 0.0
        total_weight = sum(self.weights)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in zip(self.values, self.weights)) / total_weight
    
    @property
    def weighted_std(self) -> float:
        """Get weighted standard deviation"""
        if len(self.values) < 2:
            return 0.0
        mean = self.weighted_mean
        total_weight = sum(self.weights)
        if total_weight == 0:
            return 0.0
        variance = sum(w * (v - mean) ** 2 for v, w in zip(self.values, self.weights)) / total_weight
        return np.sqrt(variance)
    
    @property
    def count(self) -> int:
        """Number of values"""
        return len(self.values)
    
    @property
    def effective_count(self) -> float:
        """Effective count accounting for decay"""
        return sum(self.weights)
    
    def recent_mean(self, n: int = 10) -> float:
        """Mean of most recent n values"""
        if not self.values:
            return 0.0
        recent = list(self.values)[-n:]
        return sum(recent) / len(recent) if recent else 0.0


class PerformanceTracker:
    """
    Track trading performance with exponential decay.
    
    Recent trades matter more than old trades.
    """
    
    def __init__(self, half_life: int = 30):
        """
        Initialize tracker.
        
        Args:
            half_life: Number of trades after which weight drops to 50%
        """
        self.half_life = half_life
        
        # Overall stats
        self.overall_r = DecayingStats(half_life=half_life)
        self.overall_win_rate = DecayingStats(half_life=half_life)
        self.overall_pnl = DecayingStats(half_life=half_life)
        
        # Per-regime stats
        self.regime_stats: Dict[str, Dict[str, DecayingStats]] = {}
        
        # Per-direction stats
        self.long_r = DecayingStats(half_life=half_life)
        self.short_r = DecayingStats(half_life=half_life)
        
        # Trade history
        self.trades: List[TradeRecord] = []
    
    def _get_regime_stats(self, regime: str) -> Dict[str, DecayingStats]:
        """Get or create stats for a regime"""
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                'r': DecayingStats(half_life=self.half_life),
                'win_rate': DecayingStats(half_life=self.half_life),
                'pnl': DecayingStats(half_life=self.half_life)
            }
        return self.regime_stats[regime]
    
    def record_trade(
        self,
        symbol: str,
        direction: int,
        r_multiple: float,
        regime: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        bars_held: int
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Stock symbol
            direction: 1=long, -1=short
            r_multiple: Actual R achieved
            regime: Market regime at entry
            pnl: Profit/loss in currency
            entry_price: Entry price
            exit_price: Exit price
            bars_held: Number of bars held
        """
        # Create record
        record = TradeRecord(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            r_multiple=r_multiple,
            regime=regime,
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            bars_held=bars_held
        )
        self.trades.append(record)
        
        # Update overall stats
        is_win = 1.0 if r_multiple > 0 else 0.0
        self.overall_r.add(r_multiple)
        self.overall_win_rate.add(is_win)
        self.overall_pnl.add(pnl)
        
        # Update per-regime stats
        regime_stats = self._get_regime_stats(regime)
        regime_stats['r'].add(r_multiple)
        regime_stats['win_rate'].add(is_win)
        regime_stats['pnl'].add(pnl)
        
        # Update per-direction stats
        if direction == 1:
            self.long_r.add(r_multiple)
        else:
            self.short_r.add(r_multiple)
        
        logger.info(f"📊 Recorded trade: {symbol} {r_multiple:+.2f}R, "
                   f"overall avg: {self.overall_r.weighted_mean:+.2f}R")
    
    def get_expectancy(self, regime: Optional[str] = None) -> float:
        """
        Get expected R per trade (weighted by recency).
        
        Args:
            regime: Optional regime filter
            
        Returns:
            Expected R multiple
        """
        if regime and regime in self.regime_stats:
            return self.regime_stats[regime]['r'].weighted_mean
        return self.overall_r.weighted_mean
    
    def get_win_rate(self, regime: Optional[str] = None) -> float:
        """Get win rate (weighted by recency)"""
        if regime and regime in self.regime_stats:
            return self.regime_stats[regime]['win_rate'].weighted_mean
        return self.overall_win_rate.weighted_mean
    
    def get_direction_stats(self) -> Dict[str, float]:
        """Get stats by direction"""
        return {
            'long_avg_r': self.long_r.weighted_mean,
            'long_count': self.long_r.count,
            'short_avg_r': self.short_r.weighted_mean,
            'short_count': self.short_r.count
        }
    
    def get_regime_summary(self) -> Dict[str, Dict]:
        """Get summary stats for each regime"""
        return {
            regime: {
                'avg_r': stats['r'].weighted_mean,
                'win_rate': stats['win_rate'].weighted_mean,
                'count': stats['r'].count,
                'effective_count': stats['r'].effective_count
            }
            for regime, stats in self.regime_stats.items()
        }
    
    def should_trade_regime(
        self,
        regime: str,
        min_expectancy: float = 0.0,
        min_trades: int = 10
    ) -> Tuple[bool, str]:
        """
        Determine if we should trade in a given regime based on history.
        
        Args:
            regime: Current regime
            min_expectancy: Minimum expected R to trade
            min_trades: Minimum historical trades to decide
            
        Returns:
            (should_trade, reason)
        """
        if regime not in self.regime_stats:
            return True, "NO_HISTORY"
        
        stats = self.regime_stats[regime]
        
        if stats['r'].count < min_trades:
            return True, f"INSUFFICIENT_DATA ({stats['r'].count}/{min_trades})"
        
        expectancy = stats['r'].weighted_mean
        
        if expectancy < min_expectancy:
            return False, f"LOW_EXPECTANCY ({expectancy:.2f}R < {min_expectancy:.2f}R)"
        
        return True, f"OK (expectancy={expectancy:.2f}R)"
    
    def get_recent_performance(self, n: int = 20) -> Dict:
        """Get performance of last n trades"""
        if not self.trades:
            return {'count': 0}
        
        recent = self.trades[-n:]
        r_values = [t.r_multiple for t in recent]
        
        return {
            'count': len(recent),
            'avg_r': np.mean(r_values),
            'win_rate': sum(1 for r in r_values if r > 0) / len(r_values),
            'total_pnl': sum(t.pnl for t in recent),
            'best_r': max(r_values),
            'worst_r': min(r_values)
        }
    
    def get_drawdown_stats(self) -> Dict:
        """Calculate drawdown statistics"""
        if not self.trades:
            return {'max_drawdown': 0, 'current_drawdown': 0}
        
        # Calculate equity curve
        pnls = [t.pnl for t in self.trades]
        equity = np.cumsum(pnls)
        
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Drawdown at each point
        drawdowns = running_max - equity
        
        return {
            'max_drawdown': float(np.max(drawdowns)) if len(drawdowns) > 0 else 0,
            'current_drawdown': float(drawdowns[-1]) if len(drawdowns) > 0 else 0,
            'max_equity': float(running_max[-1]) if len(running_max) > 0 else 0
        }
    
    def to_dict(self) -> Dict:
        """Export tracker state to dict"""
        return {
            'overall': {
                'avg_r': self.overall_r.weighted_mean,
                'win_rate': self.overall_win_rate.weighted_mean,
                'total_trades': len(self.trades)
            },
            'direction': self.get_direction_stats(),
            'regimes': self.get_regime_summary(),
            'recent': self.get_recent_performance(),
            'drawdown': self.get_drawdown_stats()
        }


# Singleton instance
_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get singleton performance tracker"""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker
