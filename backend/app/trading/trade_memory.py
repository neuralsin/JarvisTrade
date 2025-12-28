"""
V2 Trade Memory System
Stores and analyzes executed trades for online learning.

Features:
- Tracks all trade parameters and outcomes
- Computes rolling statistics by direction and regime
- Detects model degradation
- Triggers retraining when necessary
"""
from app.db.database import SessionLocal
from app.db.models import Trade, Model
from app.config import settings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TradeMemory:
    """
    Trade memory system for online learning.
    
    Tracks trade outcomes and updates expectancy statistics.
    """
    
    def __init__(self, user_id: str, db):
        """
        Initialize trade memory.
        
        Args:
            user_id: User ID
            db: Database session
        """
        self.user_id = user_id
        self.db = db
        self.memory: List[Dict] = []
        self.stats_by_direction: Dict = defaultdict(lambda: {'wins': 0, 'losses': 0, 'r_sum': 0})
        self.stats_by_regime: Dict = defaultdict(lambda: {'wins': 0, 'losses': 0, 'r_sum': 0})
    
    def load_from_db(self, lookback_trades: int = None):
        """
        Load trade history from database.
        
        Args:
            lookback_trades: Number of recent trades to load
        """
        lookback = lookback_trades or settings.V2_ROLLING_WINDOW_TRADES
        
        trades = self.db.query(Trade).filter(
            Trade.user_id == self.user_id,
            Trade.status == 'closed',
            Trade.pnl.isnot(None)
        ).order_by(Trade.exit_ts.desc()).limit(lookback).all()
        
        self.memory = []
        for trade in trades:
            self.memory.append(self._trade_to_dict(trade))
        
        self._compute_stats()
        logger.info(f"Loaded {len(self.memory)} trades into memory")
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade record to memory dict."""
        # Determine direction from side
        direction = 1 if trade.side == 'BUY' else 2
        
        # Calculate R multiple
        entry = float(trade.entry_price)
        exit_price = float(trade.exit_price) if trade.exit_price else entry
        stop = float(trade.stop_price) if trade.stop_price else entry
        
        if direction == 1:  # Long
            risk = entry - stop
            pnl = exit_price - entry
        else:  # Short
            risk = stop - entry
            pnl = entry - exit_price
        
        r_multiple = pnl / risk if risk > 0 else 0
        
        # Extract regime from metadata
        metadata = trade.metadata_json or {}
        regime = metadata.get('regime', 'UNKNOWN')
        
        return {
            'trade_id': str(trade.id),
            'direction': direction,
            'regime': regime,
            'entry_price': entry,
            'exit_price': exit_price,
            'stop_price': stop,
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'r_multiple': r_multiple,
            'exit_reason': trade.exit_reason,
            'prob_direction': metadata.get('prob_direction', 0),
            'prob_quality': metadata.get('prob_quality', 0),
            'expected_r': metadata.get('expected_r', 0),
            'entry_ts': trade.entry_ts,
            'exit_ts': trade.exit_ts
        }
    
    def add_trade(self, trade_dict: Dict):
        """Add a completed trade to memory."""
        self.memory.append(trade_dict)
        
        # Keep only rolling window
        if len(self.memory) > settings.V2_ROLLING_WINDOW_TRADES:
            self.memory = self.memory[-settings.V2_ROLLING_WINDOW_TRADES:]
        
        # Update stats
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute rolling statistics."""
        self.stats_by_direction = defaultdict(lambda: {'wins': 0, 'losses': 0, 'r_sum': 0, 'trades': 0})
        self.stats_by_regime = defaultdict(lambda: {'wins': 0, 'losses': 0, 'r_sum': 0, 'trades': 0})
        
        for trade in self.memory:
            direction = trade.get('direction', 1)
            regime = trade.get('regime', 'UNKNOWN')
            r = trade.get('r_multiple', 0)
            is_win = r > 0
            
            # By direction
            self.stats_by_direction[direction]['trades'] += 1
            self.stats_by_direction[direction]['r_sum'] += r
            if is_win:
                self.stats_by_direction[direction]['wins'] += 1
            else:
                self.stats_by_direction[direction]['losses'] += 1
            
            # By regime
            self.stats_by_regime[regime]['trades'] += 1
            self.stats_by_regime[regime]['r_sum'] += r
            if is_win:
                self.stats_by_regime[regime]['wins'] += 1
            else:
                self.stats_by_regime[regime]['losses'] += 1
    
    def get_expectancy_stats(self) -> Dict:
        """
        Get expectancy statistics for expectancy engine.
        
        Returns:
            Dict with avg_win_r, avg_loss_r, win_rate by direction
        """
        result = {}
        
        for direction in [1, 2]:
            dir_name = 'long' if direction == 1 else 'short'
            stats = self.stats_by_direction[direction]
            
            if stats['trades'] == 0:
                result[f'avg_win_r_{dir_name}'] = settings.V2_BOOTSTRAP_AVG_WIN_R
                result[f'avg_loss_r_{dir_name}'] = settings.V2_BOOTSTRAP_AVG_LOSS_R
                result[f'win_rate_{dir_name}'] = settings.V2_BOOTSTRAP_WIN_RATE
                continue
            
            # Compute from trades
            wins = [t['r_multiple'] for t in self.memory 
                    if t['direction'] == direction and t['r_multiple'] > 0]
            losses = [abs(t['r_multiple']) for t in self.memory 
                      if t['direction'] == direction and t['r_multiple'] <= 0]
            
            result[f'avg_win_r_{dir_name}'] = np.mean(wins) if wins else settings.V2_BOOTSTRAP_AVG_WIN_R
            result[f'avg_loss_r_{dir_name}'] = np.mean(losses) if losses else settings.V2_BOOTSTRAP_AVG_LOSS_R
            result[f'win_rate_{dir_name}'] = stats['wins'] / stats['trades']
        
        return result
    
    def get_regime_stats(self) -> Dict:
        """Get statistics segmented by regime."""
        result = {}
        
        for regime, stats in self.stats_by_regime.items():
            if stats['trades'] == 0:
                continue
            
            result[regime] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['wins'] / stats['trades'],
                'avg_r': stats['r_sum'] / stats['trades'],
                'total_r': stats['r_sum']
            }
        
        return result
    
    def check_model_health(self) -> Dict:
        """
        Check if models are still performing.
        
        Triggers kill signal if rolling expectancy goes negative.
        """
        if len(self.memory) < settings.V2_MODEL_KILL_THRESHOLD_TRADES:
            return {'healthy': True, 'reason': 'INSUFFICIENT_DATA'}
        
        # Check rolling expectancy
        recent = self.memory[-settings.V2_MODEL_KILL_THRESHOLD_TRADES:]
        total_r = sum(t.get('r_multiple', 0) for t in recent)
        avg_r = total_r / len(recent)
        
        if avg_r < 0:
            return {
                'healthy': False,
                'reason': f'NEGATIVE_EXPECTANCY (Avg R = {avg_r:.2f} over {len(recent)} trades)',
                'recommendation': 'RETRAIN_OR_DISABLE'
            }
        
        # Check by direction
        for direction in [1, 2]:
            dir_name = 'long' if direction == 1 else 'short'
            dir_trades = [t for t in recent if t['direction'] == direction]
            
            if len(dir_trades) >= 20:
                dir_r = sum(t.get('r_multiple', 0) for t in dir_trades) / len(dir_trades)
                if dir_r < -0.5:
                    return {
                        'healthy': False,
                        'reason': f'DIRECTION_{dir_name.upper()}_FAILING (Avg R = {dir_r:.2f})',
                        'recommendation': f'DISABLE_{dir_name.upper()}_TRADES'
                    }
        
        return {'healthy': True, 'reason': 'MODELS_PERFORMING'}
    
    def get_summary(self) -> Dict:
        """Get complete memory summary."""
        if not self.memory:
            return {'trades': 0, 'message': 'No trade history'}
        
        total_r = sum(t.get('r_multiple', 0) for t in self.memory)
        wins = sum(1 for t in self.memory if t.get('r_multiple', 0) > 0)
        losses = len(self.memory) - wins
        
        return {
            'trades': len(self.memory),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.memory),
            'total_r': total_r,
            'avg_r': total_r / len(self.memory),
            'by_direction': dict(self.stats_by_direction),
            'by_regime': self.get_regime_stats(),
            'health': self.check_model_health()
        }


def get_trade_memory(user_id: str, db) -> TradeMemory:
    """Factory function for trade memory."""
    memory = TradeMemory(user_id, db)
    memory.load_from_db()
    return memory
