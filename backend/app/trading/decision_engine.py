"""
Spec 6: Decision engine with exact code flow
Makes trading decisions based on model probability and safety checks
"""
from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.config import settings
from app.db.models import User, Trade, SystemState
from app.ml.feature_engineer import features_to_dataframe
from sqlalchemy import func
from datetime import datetime, timedelta
import math
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Spec 6: Decision engine exact flow
    - Checks market safety and model probability
    - Validates indicators (trend, RSI, ATR)
    - Calculates position size based on risk
    - Enforces concurrency locks and safety limits
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.prob_min = settings.PROB_MIN
        self.prob_strong = settings.PROB_STRONG
        self.account_capital = settings.ACCOUNT_CAPITAL
        self.risk_per_trade = settings.RISK_PER_TRADE
        self.stop_multiplier = settings.STOP_MULTIPLIER
        self.target_multiplier = settings.TARGET_MULTIPLIER
        self.max_daily_loss = settings.MAX_DAILY_LOSS
        self.max_trades_per_day = settings.MAX_TRADES_PER_DAY
    
    def check_kill_switch(self) -> bool:
        """
        Spec 8: Check if kill switch is enabled
        """
        kill_switch = self.db.query(SystemState).filter(SystemState.key == "kill_switch").first()
        return kill_switch and kill_switch.value == "true"
    
    def check_daily_loss(self, user_id: str) -> float:
        """
        Bug fix #7: Check user's daily loss INCLUDING unrealized PnL (MTM)
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Realized P&L from closed trades
        daily_pnl = self.db.query(func.sum(Trade.pnl)).filter(
            Trade.user_id == user_id,
            Trade.created_at >= today_start,
            Trade.pnl.isnot(None)
        ).scalar() or 0
        
        # Bug fix #7: Include unrealized P&L from open positions (MTM)
        from sqlalchemy import text
        open_trades = self.db.execute(text("""
            SELECT entry_price, quantity, stop_loss, action FROM paper_trades
            WHERE user_id = :user_id AND status = 'ACTIVE'
        """), {"user_id": str(user_id)}).fetchall()
        
        unrealized_pnl = 0
        for trade in open_trades:
            entry_price, quantity, stop_loss, action = trade
            if entry_price and quantity and stop_loss:
                # Conservative MTM: assume worst case is stop loss hit
                if action == 'BUY':
                    unrealized = (stop_loss - entry_price) * quantity
                else:  # SELL/SHORT
                    unrealized = (entry_price - stop_loss) * quantity
                unrealized_pnl += min(unrealized, 0)  # Only count losses
        
        total_pnl = daily_pnl + unrealized_pnl
        loss_pct = abs(total_pnl) / self.account_capital if total_pnl < 0 else 0
        return loss_pct
    
    def check_daily_trade_count(self, user_id: str) -> int:
        """
        Count trades today
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        count = self.db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= today_start
        ).count()
        
        return count
    
    def decide_and_execute(self, feature_json: Dict, model, user_id: str, 
                          instrument_id: str, mode: str, current_price: float) -> Dict:
        """
        Spec 6: Main decision and execution flow
        
        Pseudocode from spec:
        - Check probability threshold
        - Check market safety
        - Check trend and indicators
        - Calculate position size
        - Execute or reject
        """
        # Spec 8: Kill switch check
        if self.check_kill_switch():
            return {
                "action": "NO_TRADE",
                "reason": "KILL_SWITCH_ENABLED",
                "prob": None
            }
        
        # Get user
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"action": "NO_TRADE", "reason": "USER_NOT_FOUND"}
        
        # Check auto-execute for live mode
        if mode == "live" and not user.auto_execute:
            return {"action": "NO_TRADE", "reason": "AUTO_EXECUTE_DISABLED"}
        
        # Spec 8: Daily loss limit
        daily_loss_pct = self.check_daily_loss(user_id)
        if daily_loss_pct >= self.max_daily_loss:
            logger.warning(f"Daily loss limit reached for user {user_id}: {daily_loss_pct:.2%}")
            return {"action": "NO_TRADE", "reason": "DAILY_LOSS_LIMIT", "prob": None}
        
        # Max trades per day
        trade_count = self.check_daily_trade_count(user_id)
        if trade_count >= self.max_trades_per_day:
            return {"action": "NO_TRADE", "reason": "MAX_TRADES_PER_DAY", "prob": None}
        
        # Predict - Handle both binary and multi-class models
        X = features_to_dataframe(feature_json)
        proba = model.predict_proba(X)[0]
        
        # =========================================================================
        # INVERTED SIGNAL WEAPONIZATION - Apply inversion at prediction time
        # =========================================================================
        is_inverted = getattr(model, 'is_inverted', False) if hasattr(model, 'is_inverted') else False
        
        # Check if model has DB metadata (for models loaded from disk)
        model_metadata = getattr(model, 'metadata', {})
        if isinstance(model_metadata, dict):
            is_inverted = model_metadata.get('is_inverted', is_inverted)
        
        if is_inverted:
            logger.info("🔄 Applying inversion to prediction (inverted model)")
            # Flip all probabilities
            proba = 1.0 - proba
        
        # Detect model type by probability array shape
        if len(proba) == 3:
            # Multi-class model (HOLD/BUY/SELL)
            # proba = [HOLD_prob, BUY_prob, SELL_prob]
            predicted_class = int(model.predict(X)[0])
            confidence = float(proba[predicted_class])
            
            # Map class to action
            class_to_action = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action_type = class_to_action.get(predicted_class, "HOLD")
            
            # For multi-class, we check BUY or SELL confidence
            if action_type == "HOLD" or confidence < self.prob_min:
                return {"action": "NO_TRADE", "reason": "LOW_CONFIDENCE", "prob": float(confidence), "predicted_class": action_type}
            
            # Use the confidence of the predicted action
            prob = confidence
            
        elif len(proba) == 2:
            # Binary model (old style)
            prob = float(proba[1])  # BUY probability
            action_type = "BUY"
            
            # Spec 6: Probability threshold
            if prob < self.prob_min:
                return {"action": "NO_TRADE", "reason": "PROB_BELOW_THRESHOLD", "prob": prob}
        else:
            raise ValueError(f"Unexpected probability shape: {len(proba)}. Expected 2 (binary) or 3 (multi-class)")
        
        # Market safety check - comprehensive validation
        market_safety = self._check_market_safety(feature_json)
        if not market_safety['safe']:
            return {
                "action": "NO_TRADE", 
                "reason": f"MARKET_UNSAFE: {market_safety['reason']}", 
                "prob": float(prob)
            }
        
        # Spec 6: Indicator checks
        rsi_14 = feature_json.get('rsi_14', 50)
        atr_percent = feature_json.get('atr_percent', 0)
        
        # Bug fix #9: V1 Long-Only Bias - make RSI filter strategy-specific
        # Only apply RSI < 45 filter for mean-reversion strategies, not momentum/trends
        strategy = getattr(model, 'strategy', None) or feature_json.get('strategy', 'default')
        if strategy == 'mean_reversion' and rsi_14 < 45:
            return {"action": "NO_TRADE", "reason": "RSI_TOO_LOW_FOR_MEAN_REVERSION", "prob": float(prob)}
        
        if atr_percent > 0.05:
            return {"action": "NO_TRADE", "reason": "ATR_TOO_HIGH", "prob": float(prob)}
        
        # Spec 6: Position sizing
        atr_14 = feature_json.get('atr_14', 0)
        entry_price = current_price
        stop_price = entry_price - (self.stop_multiplier * atr_14)
        
        if stop_price >= entry_price:
            return {"action": "NO_TRADE", "reason": "INVALID_STOP", "prob": float(prob)}
        
        risk_per_share = entry_price - stop_price
        risk_amount = self.account_capital * self.risk_per_trade
        qty = math.floor(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # =========================================================================
        # INVERTED SIGNAL WEAPONIZATION - Position size penalty
        # Inverted models get 0.7x position size until proven
        # =========================================================================
        if is_inverted:
            from app.ml.model_inverter import get_position_size_multiplier
            qty = math.floor(qty * get_position_size_multiplier(is_inverted=True))
            logger.info(f"🔄 Inverted model: applying 0.7x position size penalty → qty={qty}")
        
        if qty <= 0:
            return {"action": "NO_TRADE", "reason": "QTY_ZERO", "prob": float(prob)}
        
        target_price = entry_price + (self.target_multiplier * risk_per_share)
        
        # Build order params
        order_params = {
            'user_id': user_id,
            'instrument_id': instrument_id,
            'mode': mode,
            'entry_price': entry_price,
            'qty': qty,
            'stop': stop_price,
            'target': target_price,
            'probability': prob,
            'reason': 'SIGNAL_TRIGGERED'
        }
        
        return {
            "action": "EXECUTE",
            "order_params": order_params,
            "prob": float(prob)
        }
    
    def _check_market_safety(self, feature_json: Dict) -> Dict:
        """
        Comprehensive market safety checks with validation
        
        Args:
            feature_json: Dictionary containing feature values
        
        Returns:
            Dict with 'safe' (bool) and 'reason' (str)
        
        Raises:
            None - Always returns a dict, handles all errors gracefully
        """
        # Validate input
        if not isinstance(feature_json, dict):
            logger.error(f"Invalid feature_json type: {type(feature_json)}")
            return {'safe': False, 'reason': 'Invalid feature data'}
        
        # 1. Check Nifty trend
        nifty_trend = feature_json.get('nifty_trend', 1)
        if nifty_trend != 1:
            return {'safe': False, 'reason': 'Nifty trend bearish'}
        
        # 2. Check VIX (volatility index)
        vix = feature_json.get('vix', 20.0)
        if not isinstance(vix, (int, float)):
            logger.warning(f"Invalid VIX type: {type(vix)}, using default")
            vix = 20.0
        
        VIX_MAX = 35.0  # Don't trade in extreme volatility
        if vix > VIX_MAX:
            return {'safe': False, 'reason': f'VIX too high ({vix:.1f} > {VIX_MAX})'}
        
        # 3. Check for opening gap (using returns)
        returns_1 = feature_json.get('returns_1', 0)
        GAP_THRESHOLD = 0.03  # 3% gap
        if abs(returns_1) > GAP_THRESHOLD:
            return {'safe': False, 'reason': f'Large gap detected ({returns_1*100:.1f}%)'}
        
        # 4. Check ATR volatility
        atr_percent = feature_json.get('atr_percent', 0)
        ATR_MAX = 0.05  # 5% max volatility
        if atr_percent > ATR_MAX:
            return {'safe': False, 'reason': f'ATR too high ({atr_percent*100:.1f}%)'}
        
        # 5. Check volume (avoid low liquidity)
        volume_ratio = feature_json.get('volume_ratio', 1.0)
        VOLUME_MIN = 0.5  # At least 50% of average volume
        if volume_ratio < VOLUME_MIN:
            return {'safe': False, 'reason': f'Low volume ({volume_ratio:.2f}x average)'}
        
        # All checks passed
        logger.debug("All market safety checks passed")
        return {'safe': True, 'reason': 'All safety checks passed'}
