"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.utils.logging import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="JarvisTrade API",
    description="ML-driven trading platform for NSE/BSE stocks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "service": "jarvistrade-backend"
    }

# Import and include routers
from app.routers import auth, dashboard, trades, models, admin, settings as settings_router, buckets, backtest, options, sentiment, instruments, trading_controls

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(settings_router.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(buckets.router, prefix="/api/v1/buckets", tags=["buckets"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtesting"])
app.include_router(options.router, prefix="/api/v1/options", tags=["options"])
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["sentiment"])
app.include_router(instruments.router, prefix="/api/v1/instruments", tags=["instruments"])
app.include_router(trading_controls.router)  # Phase 5: Trading controls

# Phase 6: WebSocket endpoint for real-time signals
from fastapi import WebSocket, WebSocketDisconnect
from app.websocket_manager import ws_manager
from app.routers.auth import decode_token

@app.websocket("/ws/signals/{token}")
async def websocket_signals(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time signal updates.
    
    Args:
        token: JWT token for authentication
    """
    try:
        # Decode token to get user
        payload = decode_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        # Connect WebSocket
        await ws_manager.connect(user_id, websocket)
        
        # Keep connection alive
        while True:
            # Wait for messages (ping/pong)
            data = await websocket.receive_text()
            
            # Echo ping as pong
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id, websocket)
        logger.info (f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        ws_manager.disconnect(user_id, websocket)

@app.on_event("startup")
async def startup_event():
    logger.info(f"JarvisTrade backend starting in {settings.APP_ENV} mode")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("JarvisTrade backend shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
