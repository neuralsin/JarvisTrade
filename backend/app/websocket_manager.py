"""
Phase 6: WebSocket Manager for Real-Time Signal Streaming

Manages WebSocket connections and broadcasts signal events to connected clients.
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Supports:
    - User-specific connections
    - Broadcasting to specific users
    - Event types: signal_generated, trade_executed, models_updated
    """
    
    def __init__(self):
        # user_id -> list of WebSocket connections
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, user_id: str, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            user_id: User UUID as string
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()
        
        if user_id not in self.connections:
            self.connections[user_id] = []
        
        self.connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}. Total connections: {len(self.connections[user_id])}")
        
        # Send connection confirmation
        await websocket.send_json({
            "event": "connected",
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def disconnect(self, user_id: str, websocket: WebSocket):
        """
        Remove a WebSocket connection.
        
        Args:
            user_id: User UUID as string
            websocket: WebSocket instance to remove
        """
        if user_id in self.connections:
            if websocket in self.connections[user_id]:
                self.connections[user_id].remove(websocket)
                logger.info(f"WebSocket disconnected for user {user_id}")
            
            # Clean up empty user entries
            if not self.connections[user_id]:
                del self.connections[user_id]
    
    async def broadcast_to_user(self, user_id: str, event: dict):
        """
        Broadcast an event to all of a user's connections.
        
        Args:
            user_id: User UUID as string
            event: Event data dict (must include 'event' key)
        """
        if user_id not in self.connections:
            return
        
        # Send to all user's connections
        dead_connections = []
        for websocket in self.connections[user_id]:
            try:
                await websocket.send_json(event)
            except Exception as e:
                logger.error(f"Failed to send to WebSocket: {e}")
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(user_id, ws)
    
    async def broadcast_signal(self, user_id: str, signal_data: dict):
        """
        Broadcast a signal event to user.
        
        Args:
            user_id: User UUID as string
            signal_data: Signal log data
        """
        event = {
            "event": "signal_generated",
            "data": signal_data
        }
        await self.broadcast_to_user(user_id, event)
    
    async def broadcast_trade(self, user_id: str, trade_data: dict):
        """
        Broadcast a trade execution event to user.
        
        Args:
            user_id: User UUID as string
            trade_data: Trade data
        """
        event = {
            "event": "trade_executed",
            "data": trade_data
        }
        await self.broadcast_to_user(user_id, event)
    
    async def broadcast_models_updated(self, user_id: str, model_count: int):
        """
        Broadcast models updated event to user.
        
        Args:
            user_id: User UUID as string
            model_count: Number of selected models
        """
        event = {
            "event": "models_updated",
            "data": {"selected_count": model_count}
        }
        await self.broadcast_to_user(user_id, event)
    
    def get_connection_count(self, user_id: str = None) -> int:
        """
        Get number of active connections.
        
        Args:
            user_id: Optional user ID to get count for specific user
        
        Returns:
            Number of active connections
        """
        if user_id:
            return len(self.connections.get(user_id, []))
        return sum(len(conns) for conns in self.connections.values())


# Global instance
ws_manager = WebSocketManager()


# Helper function to emit signals from execution layer
async def emit_signal_event(user_id: str, signal_log):
    """
    Emit signal event to WebSocket.
    
    Call this from execution.py after creating SignalLog.
    """
    signal_data = {
        "id": str(signal_log.id),
        "timestamp": signal_log.ts_utc.isoformat(),
        "stock_symbol": signal_log.instrument.symbol if signal_log.instrument else "UNKNOWN",
        "model_name": signal_log.model.name if signal_log.model else "UNKNOWN",
        "probability": signal_log.probability,
        "action": signal_log.action,
        "reason": signal_log.reason,
        "trade_id": str(signal_log.trade_id) if signal_log.trade_id else None
    }
    
    await ws_manager.broadcast_signal(user_id, signal_data)
