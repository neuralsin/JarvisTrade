"""
Spec 1: SQLAlchemy ORM models matching database schema
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base
import uuid


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    kite_api_key_encrypted = Column(Text)
    kite_api_secret_encrypted = Column(Text)
    kite_access_token_encrypted = Column(Text)
    kite_request_token = Column(Text)
    auto_execute = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    trades = relationship("Trade", back_populates="user")
    equity_snapshots = relationship("EquitySnapshot", back_populates="user")


class Instrument(Base):
    __tablename__ = "instruments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(Text, unique=True, nullable=False)
    name = Column(Text)
    exchange = Column(Text)
    instrument_type = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    candles = relationship("HistoricalCandle", back_populates="instrument")
    features = relationship("Feature", back_populates="instrument")
    trades = relationship("Trade", back_populates="instrument")
    sentiments = relationship("NewsSentiment", back_populates="instrument")


class HistoricalCandle(Base):
    __tablename__ = "historical_candles"
    __table_args__ = (
        Index('idx_candles_instrument_time', 'instrument_id', 'timeframe', 'ts_utc'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey('instruments.id', ondelete='CASCADE'))
    timeframe = Column(Text, nullable=False)
    ts_utc = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    instrument = relationship("Instrument", back_populates="candles")


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (
        Index('idx_features_instrument_time', 'instrument_id', 'ts_utc'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey('instruments.id', ondelete='CASCADE'))
    ts_utc = Column(DateTime(timezone=True), nullable=False)
    feature_json = Column(JSONB)
    target = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    instrument = relationship("Instrument", back_populates="features")


class Model(Base):
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    model_type = Column(Text, default='xgboost')
    model_path = Column(Text)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    metrics_json = Column(JSONB)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    trades = relationship("Trade", back_populates="model")


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        CheckConstraint("mode IN ('paper','live')", name='check_trade_mode'),
        Index('idx_trades_user_mode', 'user_id', 'mode', 'created_at'),
        Index('idx_trades_status', 'status'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    instrument_id = Column(UUID(as_uuid=True), ForeignKey('instruments.id', ondelete='CASCADE'))
    mode = Column(Text, nullable=False)
    entry_ts = Column(DateTime(timezone=True))
    entry_price = Column(Float)
    exit_ts = Column(DateTime(timezone=True))
    exit_price = Column(Float)
    qty = Column(Integer)
    stop_price = Column(Float)
    target_price = Column(Float)
    pnl = Column(Float)
    reason = Column(Text)
    probability = Column(Float)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'))
    status = Column(Text, default='open')
    kite_order_id = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="trades")
    instrument = relationship("Instrument", back_populates="trades")
    model = relationship("Model", back_populates="trades")
    logs = relationship("TradeLog", back_populates="trade")


class TradeLog(Base):
    __tablename__ = "trade_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey('trades.id', ondelete='CASCADE'))
    log_text = Column(Text)
    log_level = Column(Text, default='INFO')
    ts = Column(DateTime(timezone=True), server_default=func.now())
    
    trade = relationship("Trade", back_populates="logs")


class SystemState(Base):
    __tablename__ = "system_state"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(Text, unique=True, nullable=False)
    value = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (
        CheckConstraint("mode IN ('paper','live')", name='check_equity_mode'),
        Index('idx_equity_snapshots_user', 'user_id', 'mode', 'ts_utc'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    mode = Column(Text)
    ts_utc = Column(DateTime(timezone=True), nullable=False)
    equity_value = Column(Float, nullable=False)
    daily_pnl = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="equity_snapshots")


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    __table_args__ = (
        Index('idx_sentiment_instrument_time', 'instrument_id', 'ts_utc'),
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey('instruments.id', ondelete='CASCADE'))
    ts_utc = Column(DateTime(timezone=True), nullable=False)
    sentiment_1d = Column(Float)
    sentiment_3d = Column(Float)
    sentiment_7d = Column(Float)
    news_count = Column(Integer)
    source = Column(Text, default='newsapi')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    instrument = relationship("Instrument", back_populates="sentiments")
