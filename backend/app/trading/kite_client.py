"""
Zerodha Kite API client integration
"""
from kiteconnect import KiteConnect
from app.config import settings
from app.utils.crypto import decrypt_text
from app.utils.retry import retry_with_backoff
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KiteClient:
    """
    Wrapper for Zerodha Kite API
    Handles OAuth, order placement, and WebSocket (future)
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        
        if access_token:
            self.kite.set_access_token(access_token)
    
    def get_login_url(self) -> str:
        """
        Spec 18: Get Kite OAuth login URL
        """
        return self.kite.login_url()
    
    def generate_session(self, request_token: str) -> str:
        """
        Generate access token from request token
        """
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = data["access_token"]
        self.kite.set_access_token(access_token)
        return access_token
    
    @retry_with_backoff(max_attempts=3, exceptions=(Exception,))
    def place_order(self, symbol: str, transaction_type: str, quantity: int, 
                    order_type: str = "MARKET", product: str = "MIS") -> Dict:
        """
        Spec 6 & 16: Place order with retry logic
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            transaction_type: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MARKET", "LIMIT", etc.
            product: "MIS" (intraday), "CNC" (delivery)
        
        Returns:
            Order response with order_id
        """
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product
            )
            
            logger.info(f"Order placed: {order_id} for {symbol}")
            return {"order_id": order_id, "status": "success"}
        
        except Exception as e:
            logger.error(f"Order placement failed: {str(e)}")
            # Spec 16: Handle specific error codes
            error_msg = str(e)
            if "INSUFFICIENT_FUNDS" in error_msg or "insufficient funds" in error_msg.lower():
                raise Exception("INSUFFICIENT_FUNDS: " + error_msg)
            elif "INVALID_INSTRUMENT" in error_msg or "invalid" in error_msg.lower():
                raise Exception("INVALID_INSTRUMENT: " + error_msg)
            else:
                raise
    
    @retry_with_backoff(max_attempts=5, exceptions=(Exception,))
    def get_order_status(self, order_id: str) -> Dict:
        """
        Spec 6: Get order status and handle partial fills
        """
        orders = self.kite.orders()
        for order in orders:
            if order['order_id'] == order_id:
                return {
                    "order_id": order_id,
                    "status": order['status'],
                    "filled_quantity": order['filled_quantity'],
                    "pending_quantity": order['pending_quantity'],
                    "average_price": order['average_price']
                }
        
        return {"order_id": order_id, "status": "NOT_FOUND"}
    
    @retry_with_backoff(max_attempts=3, exceptions=(Exception,))
    def cancel_order(self, order_id: str):
        """
        Cancel pending order
        """
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        logger.info(f"Order cancelled: {order_id}")
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get real-time quote
        """
        instrument = f"{exchange}:{symbol}"
        quote = self.kite.quote(instrument)[instrument]
        
        return {
            "symbol": symbol,
            "last_price": quote['last_price'],
            "open": quote['ohlc']['open'],
            "high": quote['ohlc']['high'],
            "low": quote['ohlc']['low'],
            "close": quote['ohlc']['close'],
            "volume": quote['volume']
        }


def get_kite_client_for_user(user) -> Optional[KiteClient]:
    """
    Create KiteClient instance for a user with decrypted credentials
    """
    if not user.kite_api_key_encrypted or not user.kite_api_secret_encrypted:
        logger.warning(f"Kite credentials not configured for user {user.email}")
        return None
    
    api_key = decrypt_text(user.kite_api_key_encrypted)
    api_secret = decrypt_text(user.kite_api_secret_encrypted)
    access_token = decrypt_text(user.kite_access_token_encrypted) if user.kite_access_token_encrypted else None
    
    try:
        return KiteClient(api_key, api_secret, access_token)
    except Exception as e:
        logger.error(f"Failed to create Kite client: {str(e)}")
        return None
