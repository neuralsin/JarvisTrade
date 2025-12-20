"""
Spec 7: Paper trading simulator with exact mechanics
Simulates order fills, slippage, commission, and P&L tracking
"""
import random
from typing import Dict, Optional
from datetime import datetime
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class PaperSimulator:
    """
    Spec 7: Paper trading simulator
    - Simulates fills when tick price reaches order price
    - Applies slippage and commission
    - Tracks stop and target hits
    """
    
    def __init__(self):
        self.commission_flat = settings.COMMISSION_FLAT
        self.commission_pct = settings.COMMISSION_PCT
        self.slippage_min = settings.SLIPPAGE_PCT_MIN
        self.slippage_max = settings.SLIPPAGE_PCT_MAX
    
    def calculate_slippage(self, price: float) -> float:
        """
        Spec 7: Slippage calculation
        slippage = random between (SLIPPAGE_PCT_MIN * price, SLIPPAGE_PCT_MAX * price)
        """
        slippage_pct = random.uniform(self.slippage_min, self.slippage_max)
        return price * slippage_pct
    
    def calculate_commission(self, price: float, qty: int) -> float:
        """
        Spec 7: Commission model (flat + percentage)
        """
        trade_value = price * qty
        commission = self.commission_flat + (trade_value * self.commission_pct)
        return commission
    
    def simulate_entry(self, order_params: Dict) -> Dict:
        """
        Simulate order entry
        
        Args:
            order_params: {
                'symbol': str,
                'entry_price': float,
                'qty': int,
                'stop': float,
                'target': float
            }
        
        Returns:
            Simulated fill result
        """
        entry_price = order_params['entry_price']
        qty = order_params['qty']
        
        # Apply slippage
        slippage = self.calculate_slippage(entry_price)
        filled_price = entry_price + slippage
        
        # Calculate commission
        commission = self.calculate_commission(filled_price, qty)
        
        result = {
            'status': 'filled',
            'filled_price': filled_price,
            'filled_qty': qty,
            'commission': commission,
            'slippage': slippage,
            'fill_time': datetime.utcnow().isoformat() + 'Z'
        }
        
        logger.info(f"Paper trade entry: {order_params['symbol']} @ {filled_price:.2f} (slippage: {slippage:.2f})")
        
        return result
    
    def check_exit_conditions(self, current_tick: Dict, trade_params: Dict) -> Optional[Dict]:
        """
        Spec 7: Check if stop or target is hit by current tick
        
        Args:
            current_tick: {'last_price': float, 'high': float, 'low': float}
            trade_params: {'stop': float, 'target': float, 'entry_price': float, 'qty': int}
        
        Returns:
            Exit result if condition met, None otherwise
        """
        current_low = current_tick.get('low', current_tick['last_price'])
        current_high = current_tick.get('high', current_tick['last_price'])
        
        stop = trade_params['stop']
        target = trade_params['target']
        qty = trade_params['qty']
        entry_price = trade_params['entry_price']
        
        # Check stop hit
        if current_low <= stop:
            exit_price = stop
            slippage = self.calculate_slippage(exit_price)
            filled_price = exit_price - slippage  # Slippage works against us on exit too
            commission = self.calculate_commission(filled_price, qty)
            
            pnl = (filled_price - entry_price) * qty - commission
            
            return {
                'status': 'stopped',
                'exit_price': filled_price,
                'commission': commission,
                'pnl': pnl,
                'exit_time': datetime.utcnow().isoformat() + 'Z',
                'reason': 'STOP_HIT'
            }
        
        # Check target hit
        if current_high >= target:
            exit_price = target
            slippage = self.calculate_slippage(exit_price)
            filled_price = exit_price + slippage  # Slippage in our favor on target
            commission = self.calculate_commission(filled_price, qty)
            
            pnl = (filled_price - entry_price) * qty - commission
            
            return {
                'status': 'target_reached',
                'exit_price': filled_price,
                'commission': commission,
                'pnl': pnl,
                'exit_time': datetime.utcnow().isoformat() + 'Z',
                'reason': 'TARGET_HIT'
            }
        
        return None
    
    def calculate_current_pnl(self, entry_price: float, current_price: float, qty: int) -> float:
        """
        Calculate unrealized P&L for open position
        """
        return (current_price - entry_price) * qty
