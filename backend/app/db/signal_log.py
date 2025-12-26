"""
Phase 2: SignalLog model for real-time signal tracking

This model stores all trading signals (both executed and rejected) for:
- Real-time monitoring in the UI
- Historical analysis and debugging
- Model performance tracking
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import uuid


class SignalLog(Base):
    """
    Phase 2: Track all trading signals for real-time monitoring.
    
    Records both EXECUTE and REJECT decisions with full context:
    - Model probability and decision
    - Feature snapshot at decision time
    - Filter results (which filters passed/failed)
    - Rejection reason if applicable
    
    This enables:
    - Live signal monitoring in UI
    - Model performance analysis
    - Debugging why trades were/weren't taken
    """
    __tablename__ = "signal_logs"
    __table_args__ = (
        Index('idx_signal_logs_user_ts', 'user_id', 'ts_utc', postgresql_ops={'ts_utc': 'DESC'}),
        Index('idx_signal_logs_model', 'model_id'),
        Index('idx_signal_logs_instrument', 'instrument_id'),
        Index('idx_signal_logs_action', 'action'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timestamp
    ts_utc = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # References
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id', ondelete='SET NULL'), nullable=True)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Decision details
    probability = Column(Float)  # Model probability (0.0 to 1.0)
    action = Column(Text, nullable=False)  # 'EXECUTE' or 'REJECT'
    reason = Column(Text)  # Why executed or why rejected (e.g., 'PROB_BELOW_THRESHOLD', 'RSI_TOO_LOW')
    
    # Context snapshots (JSONB for flexibility)
    filters_passed = Column(JSONB)  # Which filters passed: {'vix': true, 'rsi': false, ...}
    feature_snapshot = Column(JSONB)  # Feature values at decision time
    
    # If executed, link to the trade
    trade_id = Column(UUID(as_uuid=True), ForeignKey('trades.id', ondelete='SET NULL'), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model = relationship("Model")
    instrument = relationship("Instrument")
    user = relationship("User")
    trade = relationship("Trade")
    
    def __repr__(self):
        return f"<SignalLog {self.instrument.symbol if self.instrument else 'Unknown'} {self.action} @ {self.ts_utc}>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'timestamp': self.ts_utc.isoformat(),
            'symbol': self.instrument.symbol if self.instrument else None,
            'model_name': self.model.name if self.model else None,
            'probability': self.probability,
            'action': self.action,
            'reason': self.reason,
            'filters_passed': self.filters_passed,
            'trade_id': str(self.trade_id) if self.trade_id else None
        }
