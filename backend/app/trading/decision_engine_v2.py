"""
V2 Decision Engine
Bi-directional trading with regime-aware position sizing and ATR-based exits.

Supports:
- Long AND Short trades
- Regime-adjusted risk (halved in CHOP_PANIC)
- ATR-based stop/target calculation
- Reversal exit detection
- Time-based exits
"""
from app.db.database import SessionLocal
from app.db.models import User, Trade, Instrument, SystemState
from app.config import settings
from datetime import datetime, date
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DecisionEngineV2:
    """
    V2 Decision Engine with bi-directional trading support.
    
    Key differences from V1:
    - Supports SELL (short) signals
    - Regime-aware position sizing
    - ATR-based stop/target (not fixed percentages)
    - Reversal exit capability
    """
    
    def __init__(self, user_id: str, db):
        """
        Initialize V2 decision engine.
        
        Args:
            user_id: User ID
            db: Database session
        """
        self.user_id = user_id
        self.db = db
        self.user = db.query(User).filter(User.id == user_id).first()
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if system-wide kill switch is active."""
        state = self.db.query(SystemState).filter(
            SystemState.key == 'kill_switch'
        ).first()
        
        if state and state.value == 'true':
            return False, "KILL_SWITCH_ENABLED"
        return True, ""
    
    def check_daily_loss(self) -> Tuple[bool, str]:
        """Check if daily loss limit reached."""
        today = date.today()
        
        # Get today's closed trades
        trades_today = self.db.query(Trade).filter(
            Trade.user_id == self.user_id,
            Trade.status == 'closed',
            Trade.exit_ts >= datetime.combine(today, datetime.min.time())
        ).all()
        
        total_pnl = sum(t.pnl or 0 for t in trades_today)
        capital = settings.ACCOUNT_CAPITAL
        loss_pct = abs(total_pnl) / capital if total_pnl < 0 else 0
        
        if loss_pct >= settings.MAX_DAILY_LOSS:
            return False, f"DAILY_LOSS_LIMIT ({loss_pct*100:.1f}% >= {settings.MAX_DAILY_LOSS*100:.0f}%)"
        
        return True, ""
    
    def check_trade_count(self) -> Tuple[bool, str]:
        """Check if max daily trade count reached."""
        today = date.today()
        
        trades_today = self.db.query(Trade).filter(
            Trade.user_id == self.user_id,
            Trade.entry_ts >= datetime.combine(today, datetime.min.time())
        ).count()
        
        if trades_today >= settings.MAX_TRADES_PER_DAY:
            return False, f"MAX_TRADES_PER_DAY ({trades_today} >= {settings.MAX_TRADES_PER_DAY})"
        
        return True, ""
    
    def check_existing_position(self, instrument_id: str) -> Tuple[bool, str]:
        """Check if already have open position in instrument."""
        existing = self.db.query(Trade).filter(
            Trade.user_id == self.user_id,
            Trade.instrument_id == instrument_id,
            Trade.status == 'open'
        ).first()
        
        if existing:
            return False, f"EXISTING_POSITION (ID: {existing.id})"
        
        return True, ""
    
    def check_max_positions(self) -> Tuple[bool, str]:
        """Check if max concurrent positions reached."""
        open_positions = self.db.query(Trade).filter(
            Trade.user_id == self.user_id,
            Trade.status == 'open'
        ).count()
        
        if open_positions >= settings.V2_MAX_POSITIONS:
            return False, f"MAX_POSITIONS ({open_positions} >= {settings.V2_MAX_POSITIONS})"
        
        return True, ""
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        regime_name: str
    ) -> Tuple[int, float]:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            direction: 1=Long, 2=Short
            regime_name: Current market regime name
            
        Returns:
            Tuple of (quantity, risk_amount)
        """
        capital = settings.ACCOUNT_CAPITAL
        risk_pct = settings.V2_RISK_PERCENT
        
        # Reduce risk in panic regime
        if regime_name == 'CHOP_PANIC':
            risk_pct *= settings.V2_PANIC_RISK_REDUCTION
        
        risk_amount = capital * risk_pct
        
        # Calculate stop distance
        if direction == 1:  # Long
            stop_distance = entry_price - stop_price
        else:  # Short
            stop_distance = stop_price - entry_price
        
        if stop_distance <= 0:
            logger.warning(f"Invalid stop distance: {stop_distance}")
            return 0, 0
        
        qty = int(risk_amount / stop_distance)
        
        return max(qty, 1), risk_amount
    
    def decide(self, signal: Dict, features: Dict) -> Dict:
        """
        Make trading decision based on V2 signal.
        
        Args:
            signal: Signal dict from SignalGeneratorV2
            features: Current market features
            
        Returns:
            Decision dict with action and details
        """
        decision = {
            'action': 'NO_TRADE',
            'reason': '',
            'signal': signal,
            'timestamp': datetime.utcnow()
        }
        
        # Run all checks
        checks = [
            ('kill_switch', self.check_kill_switch()),
            ('daily_loss', self.check_daily_loss()),
            ('trade_count', self.check_trade_count()),
            ('existing_position', self.check_existing_position(signal.get('instrument_id', ''))),
            ('max_positions', self.check_max_positions())
        ]
        
        for check_name, (passed, reason) in checks:
            if not passed:
                decision['reason'] = reason
                logger.info(f"Trade rejected: {reason}")
                return decision
        
        # All checks passed - calculate position size
        qty, risk_amount = self.calculate_position_size(
            entry_price=signal['entry_price'],
            stop_price=signal['stop_price'],
            direction=signal['direction'],
            regime_name=signal.get('regime', 'TREND_STABLE')
        )
        
        if qty == 0:
            decision['reason'] = 'INVALID_POSITION_SIZE'
            return decision
        
        # Build execution details
        decision['action'] = 'EXECUTE'
        decision['reason'] = 'ALL_CHECKS_PASSED'
        decision['execution'] = {
            'direction': signal['direction'],
            'signal_type': signal['signal_type'],
            'quantity': qty,
            'entry_price': signal['entry_price'],
            'stop_price': signal['stop_price'],
            'target_price': signal['target_price'],
            'risk_amount': risk_amount,
            'regime': signal.get('regime'),
            'prob_direction': signal.get('prob_direction'),
            'prob_quality': signal.get('prob_quality'),
            'expected_r': signal.get('expected_r')
        }
        
        logger.info(f"Trade approved: {signal['signal_type']} {signal['stock_symbol']} "
                    f"qty={qty} @ {signal['entry_price']:.2f}")
        
        return decision


class ExitEngineV2:
    """
    V2 Exit Engine with multiple exit conditions.
    
    Exit Conditions:
    1. Stop Loss: ATR-based
    2. Take Profit: ATR-based
    3. Time Stop: Max bars held with minimum profit check
    4. Reversal Exit: Direction model flips with confirmation
    """
    
    def __init__(self, db):
        """
        Initialize V2 exit engine.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def check_stop_loss(self, trade: Trade, current_price: float) -> Tuple[bool, str]:
        """Check if stop loss hit."""
        if trade.stop_price is None:
            return False, ""
        
        direction = 1 if trade.side == 'BUY' else 2
        
        if direction == 1:  # Long
            if current_price <= float(trade.stop_price):
                return True, "STOP_LOSS_HIT"
        else:  # Short
            if current_price >= float(trade.stop_price):
                return True, "STOP_LOSS_HIT"
        
        return False, ""
    
    def check_target(self, trade: Trade, current_price: float) -> Tuple[bool, str]:
        """Check if target hit."""
        if trade.target_price is None:
            return False, ""
        
        direction = 1 if trade.side == 'BUY' else 2
        
        if direction == 1:  # Long
            if current_price >= float(trade.target_price):
                return True, "TARGET_HIT"
        else:  # Short
            if current_price <= float(trade.target_price):
                return True, "TARGET_HIT"
        
        return False, ""
    
    def check_time_stop(
        self,
        trade: Trade,
        current_price: float,
        bars_held: int,
        atr: float
    ) -> Tuple[bool, str]:
        """
        Check time-based exit.
        
        Exits if:
        - Held for >= TIME_STOP_BARS
        - AND profit < TIME_STOP_MIN_PROFIT_ATR * ATR
        """
        if bars_held < settings.V2_TIME_STOP_BARS:
            return False, ""
        
        entry = float(trade.entry_price)
        direction = 1 if trade.side == 'BUY' else 2
        
        if direction == 1:
            profit = current_price - entry
        else:
            profit = entry - current_price
        
        min_profit = settings.V2_TIME_STOP_MIN_PROFIT_ATR * atr
        
        if profit < min_profit:
            return True, f"TIME_STOP (held {bars_held} bars, profit={profit:.2f} < {min_profit:.2f})"
        
        return False, ""
    
    def check_reversal_exit(
        self,
        trade: Trade,
        direction_prediction: Dict,
        bars_held: int,
        reversal_bars: int = 0
    ) -> Tuple[bool, str]:
        """
        Check for reversal exit.
        
        Exits if:
        - Model A direction flips
        - With confidence > threshold
        - After minimum bars held
        - And persists for N bars
        """
        if bars_held < settings.V2_REVERSAL_MIN_BARS_HELD:
            return False, ""
        
        trade_direction = 1 if trade.side == 'BUY' else 2
        predicted_direction = direction_prediction.get('direction', 0)
        confidence = direction_prediction.get('confidence', 0)
        
        # Check if direction flipped
        is_flipped = (trade_direction == 1 and predicted_direction == 2) or \
                     (trade_direction == 2 and predicted_direction == 1)
        
        if is_flipped and confidence >= settings.V2_REVERSAL_EXIT_CONF:
            if reversal_bars >= settings.V2_REVERSAL_PERSIST_BARS:
                return True, f"REVERSAL_EXIT (model flipped to {predicted_direction} with {confidence:.2f} conf)"
        
        return False, ""
    
    def evaluate_exit(
        self,
        trade: Trade,
        current_price: float,
        bars_held: int,
        atr: float,
        direction_prediction: Optional[Dict] = None,
        reversal_bars: int = 0
    ) -> Dict:
        """
        Evaluate all exit conditions.
        
        Args:
            trade: Trade record
            current_price: Current price
            bars_held: Number of bars since entry
            atr: Current ATR
            direction_prediction: Latest direction model prediction
            reversal_bars: Number of consecutive reversal signals
            
        Returns:
            Exit decision dict
        """
        # Check in priority order
        
        # 1. Stop Loss (highest priority)
        should_exit, reason = self.check_stop_loss(trade, current_price)
        if should_exit:
            return {'should_exit': True, 'reason': reason, 'exit_type': 'stop'}
        
        # 2. Target
        should_exit, reason = self.check_target(trade, current_price)
        if should_exit:
            return {'should_exit': True, 'reason': reason, 'exit_type': 'target'}
        
        # 3. Reversal (if direction prediction available)
        if direction_prediction:
            should_exit, reason = self.check_reversal_exit(
                trade, direction_prediction, bars_held, reversal_bars
            )
            if should_exit:
                return {'should_exit': True, 'reason': reason, 'exit_type': 'reversal'}
        
        # 4. Time Stop
        should_exit, reason = self.check_time_stop(trade, current_price, bars_held, atr)
        if should_exit:
            return {'should_exit': True, 'reason': reason, 'exit_type': 'time'}
        
        return {'should_exit': False, 'reason': 'HOLD', 'exit_type': None}


def get_decision_engine(user_id: str, db) -> DecisionEngineV2:
    """Factory function for V2 decision engine."""
    return DecisionEngineV2(user_id, db)


def get_exit_engine(db) -> ExitEngineV2:
    """Factory function for V2 exit engine."""
    return ExitEngineV2(db)
