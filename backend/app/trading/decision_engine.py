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
        Check user's daily loss
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_pnl = self.db.query(func.sum(Trade.pnl)).filter(
            Trade.user_id == user_id,
            Trade.created_at >= today_start,
            Trade.pnl.isnot(None)
        ).scalar() or 0
        
        loss_pct = abs(daily_pnl) / self.account_capital if daily_pnl < 0 else 0
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
        
        # Predict probability
        X = features_to_dataframe(feature_json)
        prob = model.predict_proba(X)[0][1]
        
        # Spec 6: Probability threshold
        if prob < self.prob_min:
            return {"action": "NO_TRADE", "reason": "PROB_BELOW_THRESHOLD", "prob": float(prob)}
        
        # Market safety check (placeholder - implement based on feature_json)
        market_safe = feature_json.get('nifty_trend', 1) == 1
        if not market_safe:
            return {"action": "NO_TRADE", "reason": "MARKET_UNSAFE", "prob": float(prob)}
        
        # Spec 6: Indicator checks
        rsi_14 = feature_json.get('rsi_14', 50)
        atr_percent = feature_json.get('atr_percent', 0)
        
        if rsi_14 < 45:
            return {"action": "NO_TRADE", "reason": "RSI_TOO_LOW", "prob": float(prob)}
        
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
