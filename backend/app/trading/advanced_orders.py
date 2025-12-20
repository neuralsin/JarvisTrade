"""
Advanced order types: Limit, Stop-Limit, Bracket Orders
Extension of kite_client.py
"""
from app.trading.kite_client import KiteClient
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedOrderManager:
    """
    Manages advanced order types for live trading
    """
    def __init__(self, kite_client: KiteClient):
        self.kite = kite_client
    
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        transaction_type: str = "BUY",
        product: str = "MIS"
    ) -> Dict:
        """
        Place limit order - executes only at specified price or better
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            limit_price: Limit price
            transaction_type: BUY or SELL
            product: MIS (intraday) or CNC (delivery)
        """
        try:
            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "quantity": qty,
                "price": limit_price,
                "order_type": "LIMIT",
                "transaction_type": transaction_type,
                "product": product,
                "validity": "DAY"
            }
            
            order_id = self.kite.kite.place_order(**order_params)
            
            logger.info(f"Limit order placed: {symbol} @ ₹{limit_price}, Order ID: {order_id}")
            
            return {
                "status": "success",
                "order_id": order_id,
                "order_type": "limit",
                "price": limit_price
            }
        
        except Exception as e:
            logger.error(f"Limit order failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def place_stop_limit_order(
        self,
        symbol: str,
        qty: int,
        trigger_price: float,
        limit_price: float,
        transaction_type: str = "BUY",
        product: str = "MIS"
    ) -> Dict:
        """
        Place stop-limit order
        Becomes limit order when trigger price is reached
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            trigger_price: Trigger/stop price
            limit_price: Limit price after trigger
            transaction_type: BUY or SELL
        """
        try:
            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "quantity": qty,
                "price": limit_price,
                "trigger_price": trigger_price,
                "order_type": "SL",  # Stop Loss (Stop-Limit in Kite)
                "transaction_type": transaction_type,
                "product": product,
                "validity": "DAY"
            }
            
            order_id = self.kite.kite.place_order(**order_params)
            
            logger.info(f"Stop-limit order placed: {symbol}, Trigger: ₹{trigger_price}, Limit: ₹{limit_price}")
            
            return {
                "status": "success",
                "order_id": order_id,
                "order_type": "stop_limit",
                "trigger_price": trigger_price,
                "limit_price": limit_price
            }
        
        except Exception as e:
            logger.error(f"Stop-limit order failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        stop_loss: float,
        target: float,
        transaction_type: str = "BUY",
        stoploss_value: Optional[float] = None,
        squareoff_value: Optional[float] = None
    ) -> Dict:
        """
        Place bracket order (BO)
        Entry + automatic stop loss + target
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            price: Entry price
            stop_loss: Stop loss price
            target: Target price
            stoploss_value: Alternatively, SL as points from entry
            squareoff_value: Alternatively, target as points from entry
        """
        try:
            # Calculate absolute values if not provided
            if stoploss_value is None:
                stoploss_value = abs(price - stop_loss)
            
            if squareoff_value is None:
                squareoff_value = abs(target - price)
            
            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "quantity": qty,
                "price": price,
                "order_type": "LIMIT",
                "transaction_type": transaction_type,
                "product": "BO",  # Bracket Order
                "validity": "DAY",
                "stoploss": stoploss_value,
                "squareoff": squareoff_value
            }
            
            order_id = self.kite.kite.place_order(**order_params)
            
            logger.info(f"Bracket order placed: {symbol} @ ₹{price}, SL: ₹{stop_loss}, Target: ₹{target}")
            
            return {
                "status": "success",
                "order_id": order_id,
                "order_type": "bracket",
                "entry_price": price,
                "stop_loss": stop_loss,
                "target": target
            }
        
        except Exception as e:
            logger.error(f"Bracket order failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        qty: Optional[int] = None
    ) -> Dict:
        """
        Modify existing order
        """
        try:
            modify_params = {}
            
            if price is not None:
                modify_params['price'] = price
            if trigger_price is not None:
                modify_params['trigger_price'] = trigger_price
            if qty is not None:
                modify_params['quantity'] = qty
            
            self.kite.kite.modify_order(order_id, **modify_params)
            
            logger.info(f"Order modified: {order_id}")
            
            return {
                "status": "success",
                "order_id": order_id,
                "modifications": modify_params
            }
        
        except Exception as e:
            logger.error(f"Order modification failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel pending order
        """
        try:
            self.kite.kite.cancel_order(order_id)
            
            logger.info(f"Order cancelled: {order_id}")
            
            return {
                "status": "success",
                "order_id": order_id
            }
        
        except Exception as e:
            logger.error(f"Order cancellation failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get current status of an order
        """
        try:
            order_history = self.kite.kite.order_history(order_id)
            
            if order_history:
                latest_status = order_history[-1]
                return {
                    "status": "success",
                    "order_id": order_id,
                    "order_status": latest_status['status'],
                    "filled_quantity": latest_status.get('filled_quantity', 0),
                    "pending_quantity": latest_status.get('pending_quantity', 0),
                    "average_price": latest_status.get('average_price', 0),
                    "history": order_history
                }
            else:
                return {"status": "error", "message": "Order not found"}
        
        except Exception as e:
            logger.error(f"Failed to get order status: {str(e)}")
            return {"status": "error", "message": str(e)}
