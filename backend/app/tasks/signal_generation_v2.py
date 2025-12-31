"""
V2 Signal Generation
Dual-model architecture with regime-aware gating and expectancy filtering.

Signal Flow:
1. Detect market regime (TREND_STABLE, TREND_VOLATILE, RANGE_QUIET, CHOP_PANIC)
2. Model A: Predict direction (Long/Short/Neutral)
3. Gate 1: Check direction confidence vs regime-specific threshold
4. Model B: Predict quality P(win) conditional on direction
5. Gate 2: Check quality vs regime-specific hurdle
6. Expectancy: Calculate E[R] and filter if < minimum
7. Rank: Sort candidates by priority score (E[R] / ATR)
8. Execute: Top N candidates
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Signal, Model, User, Instrument, Feature, HistoricalCandle
from app.config import settings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ExpectancyEngine:
    """
    Calculates and tracks rolling expectancy.
    
    Uses bootstrap defaults on cold start, then updates from trade memory.
    """
    
    def __init__(self):
        """Initialize with bootstrap values."""
        self.avg_win_r_long = settings.V2_BOOTSTRAP_AVG_WIN_R
        self.avg_loss_r_long = settings.V2_BOOTSTRAP_AVG_LOSS_R
        self.avg_win_r_short = settings.V2_BOOTSTRAP_AVG_WIN_R
        self.avg_loss_r_short = settings.V2_BOOTSTRAP_AVG_LOSS_R
        self.win_rate_long = settings.V2_BOOTSTRAP_WIN_RATE
        self.win_rate_short = settings.V2_BOOTSTRAP_WIN_RATE
    
    def update_from_memory(self, trade_memory: List[Dict]):
        """
        Update expectancy statistics from trade history.
        
        Args:
            trade_memory: List of trade records with direction and r_multiple
        """
        if len(trade_memory) < settings.V2_TRADE_MEMORY_UPDATE_INTERVAL:
            return  # Use bootstrap values
        
        # Separate by direction
        long_trades = [t for t in trade_memory if t.get('direction') == 1]
        short_trades = [t for t in trade_memory if t.get('direction') == 2]
        
        # Update long stats
        if len(long_trades) >= 10:
            long_wins = [t for t in long_trades if t.get('r_multiple', 0) > 0]
            long_losses = [t for t in long_trades if t.get('r_multiple', 0) <= 0]
            
            if long_wins:
                self.avg_win_r_long = np.mean([t['r_multiple'] for t in long_wins])
            if long_losses:
                self.avg_loss_r_long = abs(np.mean([t['r_multiple'] for t in long_losses]))
            self.win_rate_long = len(long_wins) / len(long_trades)
        
        # Update short stats
        if len(short_trades) >= 10:
            short_wins = [t for t in short_trades if t.get('r_multiple', 0) > 0]
            short_losses = [t for t in short_trades if t.get('r_multiple', 0) <= 0]
            
            if short_wins:
                self.avg_win_r_short = np.mean([t['r_multiple'] for t in short_wins])
            if short_losses:
                self.avg_loss_r_short = abs(np.mean([t['r_multiple'] for t in short_losses]))
            self.win_rate_short = len(short_wins) / len(short_trades)
    
    def calculate(self, prob_quality: float, direction: int) -> float:
        """
        Calculate expected R for a trade.
        
        E[R] = P(win) * avg_win_R - P(loss) * avg_loss_R
        
        Args:
            prob_quality: Probability of winning (from Model B)
            direction: 1=Long, 2=Short
            
        Returns:
            Expected R multiple
        """
        if direction == 1:
            return prob_quality * self.avg_win_r_long - (1 - prob_quality) * self.avg_loss_r_long
        else:
            return prob_quality * self.avg_win_r_short - (1 - prob_quality) * self.avg_loss_r_short
    
    def get_stats(self) -> Dict:
        """Return current expectancy statistics."""
        return {
            'avg_win_r_long': self.avg_win_r_long,
            'avg_loss_r_long': self.avg_loss_r_long,
            'avg_win_r_short': self.avg_win_r_short,
            'avg_loss_r_short': self.avg_loss_r_short,
            'win_rate_long': self.win_rate_long,
            'win_rate_short': self.win_rate_short
        }


class SignalGeneratorV2:
    """
    V2 Signal generator with dual-model architecture.
    
    Implements the complete signal pipeline:
    Regime -> Direction Gate -> Quality Gate -> Expectancy -> Ranking
    """
    
    def __init__(self):
        """Initialize signal generator."""
        from app.ml.regime_detector import RegimeDetector, MarketRegime
        
        self.regime_detector = RegimeDetector(
            adx_trend_threshold=settings.V2_ADX_TREND_THRESHOLD,
            adx_range_threshold=settings.V2_ADX_RANGE_THRESHOLD,
            atr_z_volatile_threshold=settings.V2_ATR_Z_VOLATILE_THRESHOLD,
            atr_z_panic_threshold=settings.V2_ATR_Z_PANIC_THRESHOLD,
            persistence_bars=settings.V2_REGIME_PERSISTENCE_BARS
        )
        self.expectancy_engine = ExpectancyEngine()
        self.direction_models: Dict[str, 'DirectionScout'] = {}
        self.quality_models_long: Dict[str, 'QualityGatekeeper'] = {}
        self.quality_models_short: Dict[str, 'QualityGatekeeper'] = {}
        self.MarketRegime = MarketRegime
    
    def load_models(self, stock_symbol: str, model_dir: str = "models"):
        """
        Load V2 models for a stock.
        
        Args:
            stock_symbol: Stock symbol
            model_dir: Directory containing model files
        """
        from app.ml.model_a_direction import DirectionScout
        from app.ml.model_b_quality import QualityGatekeeper
        
        model_path = Path(model_dir)
        
        # Load direction model
        direction_path = model_path / f"{stock_symbol}_v2_direction.joblib"
        if direction_path.exists():
            self.direction_models[stock_symbol] = DirectionScout()
            self.direction_models[stock_symbol].load(str(direction_path))
            logger.info(f"Loaded direction model for {stock_symbol}")
        
        # Load quality models (separate for long/short)
        quality_long_path = model_path / f"{stock_symbol}_v2_quality_long.joblib"
        if quality_long_path.exists():
            self.quality_models_long[stock_symbol] = QualityGatekeeper()
            self.quality_models_long[stock_symbol].load(str(quality_long_path))
            logger.info(f"Loaded quality (long) model for {stock_symbol}")
        
        quality_short_path = model_path / f"{stock_symbol}_v2_quality_short.joblib"
        if quality_short_path.exists():
            self.quality_models_short[stock_symbol] = QualityGatekeeper()
            self.quality_models_short[stock_symbol].load(str(quality_short_path))
            logger.info(f"Loaded quality (short) model for {stock_symbol}")
    
    def get_direction_hurdle(self, regime) -> float:
        """Get direction confidence threshold for regime."""
        hurdles = {
            self.MarketRegime.TREND_STABLE: settings.V2_DIRECTION_CONF_TREND_STABLE,
            self.MarketRegime.TREND_VOLATILE: settings.V2_DIRECTION_CONF_TREND_VOLATILE,
            self.MarketRegime.RANGE_QUIET: settings.V2_DIRECTION_CONF_RANGE_QUIET,
            self.MarketRegime.CHOP_PANIC: 1.0  # Reject all trades in panic
        }
        return hurdles.get(regime, 0.65)
    
    def get_quality_hurdle(self, regime) -> float:
        """Get quality probability threshold for regime."""
        hurdles = {
            self.MarketRegime.TREND_STABLE: settings.V2_QUALITY_HURDLE_TREND_STABLE,
            self.MarketRegime.TREND_VOLATILE: settings.V2_QUALITY_HURDLE_TREND_VOLATILE,
            self.MarketRegime.RANGE_QUIET: settings.V2_QUALITY_HURDLE_RANGE_QUIET,
            self.MarketRegime.CHOP_PANIC: settings.V2_QUALITY_HURDLE_CHOP_PANIC
        }
        return hurdles.get(regime, 0.65)
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        stock_symbol: str,
        instrument_id: str
    ) -> Optional[Dict]:
        """
        Generate signal using V2 dual-model logic.
        
        Args:
            df: Recent OHLCV data (minimum 50 bars)
            stock_symbol: Stock symbol
            instrument_id: Database instrument ID
            
        Returns:
            Signal dict or None if no trade
        """
        if len(df) < 50:
            logger.debug(f"[{stock_symbol}] Insufficient data: {len(df)} bars")
            return None
        
        # Check if models are loaded
        if stock_symbol not in self.direction_models:
            logger.warning(f"[{stock_symbol}] No direction model loaded")
            return None
        
        # =================================================================
        # STEP 0: Detect Market Regime
        # =================================================================
        df_regime = self.regime_detector.compute_indicators(df)
        adx = df_regime['adx_14'].iloc[-1]
        atr_z = df_regime['atr_z'].iloc[-1]
        
        if pd.isna(adx) or pd.isna(atr_z):
            logger.debug(f"[{stock_symbol}] Missing regime indicators")
            return None
        
        regime = self.regime_detector.get_regime(adx, atr_z)
        
        # CHOP_PANIC = no trading
        if regime == self.MarketRegime.CHOP_PANIC:
            logger.info(f"[{stock_symbol}] CHOP_PANIC regime - no trade")
            return None
        
        # =================================================================
        # STEP 1: Direction Gate (Model A)
        # =================================================================
        direction_model = self.direction_models[stock_symbol]
        df_features_a = direction_model.compute_features(df)
        
        # Get latest features
        from app.ml.model_a_direction import DirectionScout
        X_a = df_features_a[DirectionScout.FEATURE_COLUMNS].iloc[-1:].values
        
        direction_pred = direction_model.predict(X_a)
        direction = direction_pred['direction']
        conf_dir = direction_pred['confidence']
        
        # Check against regime-specific threshold
        direction_hurdle = self.get_direction_hurdle(regime)
        
        if direction == 0:  # Neutral
            logger.debug(f"[{stock_symbol}] Direction: NEUTRAL - no trade")
            return None
        
        if conf_dir < direction_hurdle:
            logger.debug(f"[{stock_symbol}] Direction confidence {conf_dir:.3f} < {direction_hurdle} - no trade")
            return None
        
        # =================================================================
        # STEP 2: Quality Gate (Model B)
        # =================================================================
        if direction == 1:
            if stock_symbol not in self.quality_models_long:
                logger.warning(f"[{stock_symbol}] No quality (long) model loaded")
                return None
            quality_model = self.quality_models_long[stock_symbol]
        else:
            if stock_symbol not in self.quality_models_short:
                logger.warning(f"[{stock_symbol}] No quality (short) model loaded")
                return None
            quality_model = self.quality_models_short[stock_symbol]
        
        df_features_b = quality_model.compute_features(df, direction=direction)
        
        from app.ml.model_b_quality import QualityGatekeeper
        X_b = df_features_b[QualityGatekeeper.FEATURE_COLUMNS].iloc[-1:].values
        
        prob_quality = quality_model.predict(X_b)
        quality_hurdle = self.get_quality_hurdle(regime)
        
        if prob_quality < quality_hurdle:
            logger.debug(f"[{stock_symbol}] Quality prob {prob_quality:.3f} < {quality_hurdle} - no trade")
            return None
        
        # =================================================================
        # STEP 3: Expectancy Filter
        # =================================================================
        expected_r = self.expectancy_engine.calculate(prob_quality, direction)
        
        if expected_r < settings.V2_MIN_EXPECTANCY_R:
            logger.debug(f"[{stock_symbol}] Expected R {expected_r:.3f} < {settings.V2_MIN_EXPECTANCY_R} - no trade")
            return None
        
        # =================================================================
        # STEP 4: Calculate Priority Score
        # =================================================================
        atr_percent = df_regime['atr_percent'].iloc[-1]
        safe_atr = max(atr_percent, 0.003)
        priority = expected_r / safe_atr
        
        # =================================================================
        # STEP 5: Build Signal
        # =================================================================
        signal_type = 'BUY' if direction == 1 else 'SELL'
        current_price = df['close'].iloc[-1]
        atr = df_regime['atr_14'].iloc[-1]
        
        # Calculate stop and target
        if direction == 1:  # Long
            stop_price = current_price - (settings.V2_QUALITY_STOP_ATR_MULT * atr)
            target_price = current_price + (settings.V2_QUALITY_TARGET_ATR_MULT * atr)
        else:  # Short
            stop_price = current_price + (settings.V2_QUALITY_STOP_ATR_MULT * atr)
            target_price = current_price - (settings.V2_QUALITY_TARGET_ATR_MULT * atr)
        
        signal = {
            'signal_type': signal_type,
            'direction': direction,
            'stock_symbol': stock_symbol,
            'instrument_id': instrument_id,
            'regime': regime.name,
            'prob_direction': float(conf_dir),
            'prob_quality': float(prob_quality),
            'expected_r': float(expected_r),
            'priority': float(priority),
            'entry_price': float(current_price),
            'stop_price': float(stop_price),
            'target_price': float(target_price),
            'atr': float(atr),
            'atr_percent': float(atr_percent),
            'adx': float(adx),
            'atr_z': float(atr_z),
            'timestamp': datetime.utcnow(),
            'engine_version': 'v2'
        }
        
        logger.info(
            f"[{stock_symbol}] V2 SIGNAL: {signal_type} | "
            f"Regime={regime.name} | Dir={conf_dir:.2f} | Qual={prob_quality:.2f} | E[R]={expected_r:.2f}"
        )
        
        return signal
    
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Rank signals by priority score and return top N.
        
        Args:
            signals: List of signal dicts
            
        Returns:
            Sorted and filtered list of top signals
        """
        if not signals:
            return []
        
        # Sort by priority (E[R] / ATR) descending
        sorted_signals = sorted(signals, key=lambda x: x.get('priority', 0), reverse=True)
        
        # Return top N
        return sorted_signals[:settings.V2_MAX_POSITIONS]


@celery_app.task(bind=True)
def generate_signals_v2(self):
    """
    V2 Signal generation Celery task.
    
    Called by beat scheduler if TRADING_ENGINE_VERSION == 'v2'.
    """
    if settings.TRADING_ENGINE_VERSION != 'v2':
        logger.debug("V2 signal generation skipped - engine version is v1")
        return {"status": "skipped", "reason": "v1_mode"}
    
    db = SessionLocal()
    
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Initializing V2 signal generator...', 'progress': 5})
        
        generator = SignalGeneratorV2()
        
        # Get active users with paper trading enabled
        users = db.query(User).filter(User.paper_trading_enabled == True).all()
        
        if not users:
            return {"status": "success", "signals": 0, "message": "No users with paper trading enabled"}
        
        signals_generated = 0
        all_signals = []
        
        for user in users:
            # Get user's watchlist or selected stocks
            # For now, use default stocks
            from app.tasks.data_ingestion import NIFTY_50_STOCKS
            stocks = NIFTY_50_STOCKS[:settings.MAX_STOCKS_TO_SCAN]  # Bug fix #20: Use configurable limit
            
            for stock_symbol in stocks:
                try:
                    # Load models if not loaded
                    generator.load_models(stock_symbol)
                    
                    # Get instrument
                    instrument = db.query(Instrument).filter(Instrument.symbol == stock_symbol).first()
                    if not instrument:
                        continue
                    
                    # Get recent candles
                    candles = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument.id
                    ).order_by(HistoricalCandle.ts_utc.desc()).limit(100).all()
                    
                    if len(candles) < 50:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        'ts_utc': c.ts_utc,
                        'open': float(c.open),
                        'high': float(c.high),
                        'low': float(c.low),
                        'close': float(c.close),
                        'volume': int(c.volume)
                    } for c in reversed(candles)])
                    
                    # Generate signal
                    signal = generator.generate_signal(df, stock_symbol, str(instrument.id))
                    
                    if signal:
                        all_signals.append(signal)
                        signals_generated += 1
                
                except Exception as e:
                    logger.error(f"Error generating signal for {stock_symbol}: {e}")
                    continue
            
            # Rank and select top signals
            top_signals = generator.rank_signals(all_signals)
            
            # Save to database
            for sig in top_signals:
                signal_record = Signal(
                    user_id=user.id,
                    instrument_id=sig['instrument_id'],
                    signal_type=sig['signal_type'],
                    confidence=sig['prob_quality'],
                    entry_price=sig['entry_price'],
                    stop_price=sig['stop_price'],
                    target_price=sig['target_price'],
                    created_at=sig['timestamp'],
                    metadata_json={
                        'engine_version': 'v2',
                        'regime': sig['regime'],
                        'prob_direction': sig['prob_direction'],
                        'expected_r': sig['expected_r'],
                        'priority': sig['priority']
                    }
                )
                db.add(signal_record)
            
            db.commit()
        
        return {
            "status": "success",
            "signals_generated": signals_generated,
            "signals_executed": len(top_signals) if 'top_signals' in locals() else 0
        }
    
    except Exception as e:
        logger.error(f"V2 signal generation failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}
    
    finally:
        db.close()


def get_signal_generator() -> SignalGeneratorV2:
    """Factory function to get signal generator based on version."""
    return SignalGeneratorV2()
