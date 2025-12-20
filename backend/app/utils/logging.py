"""
Spec 15: Structured JSON logging
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs
    Format: {ts, level, service, message, extra}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "jarvistrade-backend",
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'extra'):
            log_data['extra'] = record.extra
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.format Exception(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging():
    """
    Configure application-wide logging with JSON format
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
