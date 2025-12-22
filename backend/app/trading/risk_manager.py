"""
Spec 8: Risk manager with circuit breakers and safety features
"""
from sqlalchemy.orm import Session
from app.db.models import SystemState
from app.config import settings
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Spec 8: Risk management and circuit breakers
    - VIX threshold check
    - NIFTY drop detection
    - WebSocket status monitoring
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vix_threshold = settings.VIX_THRESHOLD
        self.nifty_drop_threshold = settings.NIFTY_DROP_THRESHOLD
    
    def check_circuit_breakers(self, current_vix: float = None, nifty_intraday_change: float = None) -> Dict:
        """
        Spec 8: Circuit breaker checks
        - If VIX > 40, disable auto-execution
        - If NIFTY falls > 4% intraday, disable auto-execution
        
        Returns:
            {"safe": bool, "reasons": [str]}
        """
        reasons = []
        safe = True
        
        # VIX check
        if current_vix and current_vix > self.vix_threshold:
            reasons.append(f"VIX_HIGH: {current_vix:.2f} > {self.vix_threshold}")
            safe = False
            self._disable_auto_execution("VIX_CIRCUIT_BREAKER")
        
        # NIFTY drop check
        if nifty_intraday_change and nifty_intraday_change < -self.nifty_drop_threshold:
            reasons.append(f"NIFTY_DROP: {nifty_intraday_change:.2%} < -{self.nifty_drop_threshold:.2%}")
            safe = False
            self._disable_auto_execution("MARKET_CRASH_CIRCUIT_BREAKER")
        
        # WebSocket status check
        kws_status = self._get_kws_status()
        if kws_status == "DOWN":
            reasons.append("WEBSOCKET_DOWN")
            # Don't disable auto-execution, but warn
            logger.warning("Kite WebSocket is down, using fallback data")
        
        return {"safe": safe, "reasons": reasons}
    
    def _disable_auto_execution(self, reason: str):
        """
        Disable auto-execution system-wide
        """
        # Set kill switch
        kill_switch = self.db.query(SystemState).filter(SystemState.key == "kill_switch").first()
        if kill_switch:
            kill_switch.value = "true"
        else:
            kill_switch = SystemState(key="kill_switch", value="true")
            self.db.add(kill_switch)
        
        self.db.commit()
        logger.critical(f"AUTO-EXECUTION DISABLED: {reason}")
    
    def _get_kws_status(self) -> str:
        """
        Get Kite WebSocket status
        """
        kws = self.db.query(SystemState).filter(SystemState.key == "kws_status").first()
        return kws.value if kws else "DOWN"
    
    def set_kws_status(self, status: str):
        """
        Update WebSocket status
        """
        kws = self.db.query(SystemState).filter(SystemState.key == "kws_status").first()
        if kws:
            kws.value = status
        else:
            kws = SystemState(key="kws_status", value=status)
            self.db.add(kws)
        self.db.commit()
