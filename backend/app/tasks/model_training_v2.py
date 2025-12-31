"""
V2 Model Training Task
Trains the dual-model architecture: Model A (Direction) + Model B (Quality)

Training Flow:
1. Fetch historical data (Yahoo Finance)
2. Compute regime indicators
3. Train Model A (Direction Scout) - multi-class
4. Train Model B Long (Quality Gatekeeper) - binary
5. Train Model B Short (Quality Gatekeeper) - binary
6. Save models to disk with version tracking
7. Register in database
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Model, Instrument
from app.config import settings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


@celery_app.task(bind=True, time_limit=7200, soft_time_limit=6000)
def train_v2_models(
    self,
    stock_symbol: str,
    model_name: str = None,
    interval: str = '15m',
    start_date: str = None,
    end_date: str = None,
    hyperparams_direction: dict = None,
    hyperparams_quality: dict = None
):
    """
    Train V2 dual-model architecture for a single stock.
    
    Creates 3 models:
    - Model A: Direction Scout (multi-class: Neutral/Long/Short)
    - Model B Long: Quality Gatekeeper for long trades
    - Model B Short: Quality Gatekeeper for short trades
    
    Args:
        stock_symbol: Stock symbol (e.g., 'RELIANCE')
        model_name: Base name for models (defaults to stock_symbol + timestamp)
        interval: Candle interval (default '15m')
        start_date: Training start date (YYYY-MM-DD)
        end_date: Training end date (YYYY-MM-DD)
        hyperparams_direction: Optional hyperparams for Model A
        hyperparams_quality: Optional hyperparams for Model B
        
    Returns:
        Dict with model IDs and metrics
    """
    db = SessionLocal()
    task_id = self.request.id
    stock_symbol = stock_symbol.strip().upper()
    
    if not model_name:
        model_name = f"{stock_symbol}_v2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # =====================================================================
        # STEP 1: Validate and prepare
        # =====================================================================
        logger.info(f"[{task_id}] Starting V2 training for {stock_symbol}")
        self.update_state(state='PROGRESS', meta={
            'status': f'Initializing V2 training for {stock_symbol}...',
            'progress': 5
        })
        
        # Set default dates
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            days_back = 59 if interval in ['1m', '5m', '15m', '30m', '1h'] else 365
            start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        MODELS_DIR.mkdir(exist_ok=True)
        
        # =====================================================================
        # STEP 2: Fetch training data
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': f'Fetching data for {stock_symbol}...',
            'progress': 10
        })
        
        from app.tasks.data_ingestion import fetch_historical_yahoo
        
        logger.info(f"Fetching Yahoo Finance data: {stock_symbol}, {start_date} to {end_date}, interval={interval}")
        
        try:
            df = fetch_historical_yahoo(
                symbol=stock_symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                exchange='NS'
            )
        except Exception as fetch_error:
            logger.error(f"Data fetch failed for {stock_symbol}: {str(fetch_error)}")
            raise ValueError(
                f"Failed to fetch data for {stock_symbol}: {str(fetch_error)}. "
                f"Ensure the stock symbol is valid on Yahoo Finance (try adding .NS suffix) "
                f"or check your internet connection."
            )
        
        if df is None or len(df) < 200:
            available = len(df) if df is not None else 0
            raise ValueError(
                f"Insufficient data for {stock_symbol}: {available} bars (need >= 200). "
                f"The symbol may not have enough trading history for interval={interval}, "
                f"or the date range {start_date} to {end_date} may be too narrow."
            )
        
        # Normalize columns
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Yahoo Finance returns 'timestamp', need to rename to ts_utc
        # Check all possible column names in order of priority
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'ts_utc'})
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'ts_utc'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'ts_utc'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'ts_utc'})
        
        # Validate ts_utc exists after all attempts
        if 'ts_utc' not in df.columns:
            logger.error(f"Available columns after normalization: {list(df.columns)}")
            raise ValueError(f"Could not find timestamp column in data. Available: {list(df.columns)}")
        
        # Log data summary
        logger.info(f"✓ Fetched {len(df)} candles for {stock_symbol}")
        logger.info(f"  Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # =====================================================================
        # STEP 3: Compute regime indicators
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Computing regime indicators...',
            'progress': 15
        })
        
        from app.ml.regime_detector import RegimeDetector
        
        try:
            detector = RegimeDetector(
                adx_trend_threshold=settings.V2_ADX_TREND_THRESHOLD,
                adx_range_threshold=settings.V2_ADX_RANGE_THRESHOLD,
                atr_z_volatile_threshold=settings.V2_ATR_Z_VOLATILE_THRESHOLD,
                atr_z_panic_threshold=settings.V2_ATR_Z_PANIC_THRESHOLD
            )
            
            df = detector.compute_indicators(df)
            df['regime'] = detector.detect_from_df(df)
            
            # Validate regime indicators were computed
            required_indicators = ['atr_14', 'atr_percent', 'atr_z', 'adx_14']
            for col in required_indicators:
                if col not in df.columns:
                    logger.warning(f"Regime indicator '{col}' missing, will be computed by models")
            
            logger.info(f"✓ Regime indicators computed. Columns: {[c for c in df.columns if c in required_indicators]}")
            
        except Exception as regime_error:
            logger.error(f"Regime computation failed: {str(regime_error)}")
            raise ValueError(f"Failed to compute regime indicators: {str(regime_error)}")
        
        # =====================================================================
        # STEP 4: Train Model A (Direction Scout)
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Training Model A (Direction Scout)...',
            'progress': 25
        })
        
        from app.ml.model_a_direction import DirectionScout
        
        try:
            direction_model = DirectionScout(hyperparams=hyperparams_direction)
            
            # Compute features and labels
            logger.info("Computing Direction Scout features...")
            df_direction = direction_model.compute_features(df.copy())
            
            logger.info("Generating direction labels...")
            df_direction = direction_model.generate_labels(
                df_direction,
                lookahead_bars=settings.V2_DIRECTION_LOOKAHEAD_BARS,
                threshold_mult=settings.V2_DIRECTION_THRESHOLD_MULTIPLIER
            )
            
            # Validate labels were generated
            valid_labels = df_direction['direction_label'].notna().sum()
            if valid_labels < 50:
                raise ValueError(f"Insufficient direction labels: {valid_labels} (need >= 50)")
            
            # Prepare data
            logger.info("Preparing Direction Scout training data...")
            X_train_a, y_train_a, X_val_a, y_val_a, X_test_a, y_test_a = direction_model.prepare_data(df_direction)
            
            self.update_state(state='PROGRESS', meta={
                'status': f'Training Direction Scout on {len(X_train_a)} samples...',
                'progress': 35
            })
            
            # Train
            logger.info(f"Training Direction Scout: {len(X_train_a)} train, {len(X_val_a)} val, {len(X_test_a)} test")
            direction_model.train(X_train_a, y_train_a, X_val_a, y_val_a)
            
            # Evaluate
            metrics_direction = direction_model.evaluate(X_test_a, y_test_a)
            
            # Save
            direction_path = str(MODELS_DIR / f"{stock_symbol}_v2_direction.joblib")
            direction_model.save(direction_path)
            
            logger.info(f"✓ Model A trained: AUC_Long={metrics_direction['auc_long']:.3f}, "
                        f"AUC_Short={metrics_direction['auc_short']:.3f}")
                        
        except Exception as model_a_error:
            logger.error(f"Model A (Direction Scout) training failed: {str(model_a_error)}")
            raise ValueError(f"Direction Scout training failed: {str(model_a_error)}")
        
        # =====================================================================
        # STEP 5: Train Model B Long (Quality Gatekeeper - Long)
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Training Model B (Quality - Long)...',
            'progress': 50
        })
        
        from app.ml.model_b_quality import QualityGatekeeper
        
        try:
            quality_model_long = QualityGatekeeper(hyperparams=hyperparams_quality)
            
            # Compute features and triple-barrier labels for LONG trades
            logger.info("Computing Quality Gatekeeper (Long) features...")
            df_quality_long = quality_model_long.compute_features(df.copy(), direction=1)
            
            logger.info("Generating triple-barrier labels (Long)...")
            df_quality_long = quality_model_long.generate_labels_triple_barrier(
                df_quality_long,
                direction=1,
                target_atr_mult=settings.V2_QUALITY_TARGET_ATR_MULT,
                stop_atr_mult=settings.V2_QUALITY_STOP_ATR_MULT,
                time_limit_bars=settings.V2_QUALITY_TIME_LIMIT_BARS,
                slippage_pct=settings.V2_QUALITY_SLIPPAGE_PCT
            )
            
            # Validate labels
            valid_labels = df_quality_long['quality_label'].notna().sum() if 'quality_label' in df_quality_long else 0
            if valid_labels < 50:
                logger.warning(f"Low quality labels (Long): {valid_labels}. May produce weak model.")
            
            # Prepare data
            logger.info("Preparing Quality (Long) training data...")
            X_train_bl, y_train_bl, X_val_bl, y_val_bl, X_test_bl, y_test_bl = quality_model_long.prepare_data(df_quality_long)
            
            self.update_state(state='PROGRESS', meta={
                'status': f'Training Quality (Long) on {len(X_train_bl)} samples...',
                'progress': 60
            })
            
            # Train
            logger.info(f"Training Quality (Long): {len(X_train_bl)} train, {len(X_val_bl)} val, {len(X_test_bl)} test")
            quality_model_long.train(X_train_bl, y_train_bl, X_val_bl, y_val_bl)
            
            # Evaluate
            metrics_quality_long = quality_model_long.evaluate(X_test_bl, y_test_bl)
            
            # Save
            quality_long_path = str(MODELS_DIR / f"{stock_symbol}_v2_quality_long.joblib")
            quality_model_long.save(quality_long_path)
            
            logger.info(f"✓ Model B (Long) trained: AUC={metrics_quality_long['auc_roc']:.3f}, "
                        f"Precision={metrics_quality_long['precision']:.3f}")
                        
        except Exception as model_b_long_error:
            logger.error(f"Model B (Quality Long) training failed: {str(model_b_long_error)}")
            raise ValueError(f"Quality Gatekeeper (Long) training failed: {str(model_b_long_error)}")
        
        # =====================================================================
        # STEP 6: Train Model B Short (Quality Gatekeeper - Short)
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Training Model B (Quality - Short)...',
            'progress': 75
        })
        
        try:
            quality_model_short = QualityGatekeeper(hyperparams=hyperparams_quality)
            
            # Compute features and triple-barrier labels for SHORT trades
            logger.info("Computing Quality Gatekeeper (Short) features...")
            df_quality_short = quality_model_short.compute_features(df.copy(), direction=2)
            
            logger.info("Generating triple-barrier labels (Short)...")
            df_quality_short = quality_model_short.generate_labels_triple_barrier(
                df_quality_short,
                direction=2,
                target_atr_mult=settings.V2_QUALITY_TARGET_ATR_MULT,
                stop_atr_mult=settings.V2_QUALITY_STOP_ATR_MULT,
                time_limit_bars=settings.V2_QUALITY_TIME_LIMIT_BARS,
                slippage_pct=settings.V2_QUALITY_SLIPPAGE_PCT
            )
            
            # Validate labels
            valid_labels = df_quality_short['quality_label'].notna().sum() if 'quality_label' in df_quality_short else 0
            if valid_labels < 50:
                logger.warning(f"Low quality labels (Short): {valid_labels}. May produce weak model.")
            
            # Prepare data
            logger.info("Preparing Quality (Short) training data...")
            X_train_bs, y_train_bs, X_val_bs, y_val_bs, X_test_bs, y_test_bs = quality_model_short.prepare_data(df_quality_short)
            
            self.update_state(state='PROGRESS', meta={
                'status': f'Training Quality (Short) on {len(X_train_bs)} samples...',
                'progress': 85
            })
            
            # Train
            logger.info(f"Training Quality (Short): {len(X_train_bs)} train, {len(X_val_bs)} val, {len(X_test_bs)} test")
            quality_model_short.train(X_train_bs, y_train_bs, X_val_bs, y_val_bs)
            
            # Evaluate
            metrics_quality_short = quality_model_short.evaluate(X_test_bs, y_test_bs)
            
            # Save
            quality_short_path = str(MODELS_DIR / f"{stock_symbol}_v2_quality_short.joblib")
            quality_model_short.save(quality_short_path)
            
            logger.info(f"✓ Model B (Short) trained: AUC={metrics_quality_short['auc_roc']:.3f}, "
                        f"Precision={metrics_quality_short['precision']:.3f}")
                        
        except Exception as model_b_short_error:
            logger.error(f"Model B (Quality Short) training failed: {str(model_b_short_error)}")
            raise ValueError(f"Quality Gatekeeper (Short) training failed: {str(model_b_short_error)}")
        
        # =====================================================================
        # STEP 7: Save to database
        # =====================================================================
        self.update_state(state='PROGRESS', meta={
            'status': 'Saving models to database...',
            'progress': 95
        })
        
        # Ensure instrument exists
        instrument = db.query(Instrument).filter(Instrument.symbol == stock_symbol).first()
        if not instrument:
            instrument = Instrument(
                symbol=stock_symbol,
                name=stock_symbol,
                exchange='NSE',
                instrument_type='EQ'
            )
            db.add(instrument)
            db.commit()
            db.refresh(instrument)
        
        # Helper to convert numpy types to JSON-serializable Python natives
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(item) for item in obj]
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save Model A record
        model_a_record = Model(
            name=f"{model_name}_direction",
            model_type='xgboost_v2_direction',
            model_path=direction_path,
            model_format='joblib',
            stock_symbol=stock_symbol,
            trained_at=datetime.utcnow(),
            metrics_json=to_native(metrics_direction),
            is_active=True  # V2 models auto-active
        )
        db.add(model_a_record)
        
        # Save Model B Long record
        model_bl_record = Model(
            name=f"{model_name}_quality_long",
            model_type='xgboost_v2_quality_long',
            model_path=quality_long_path,
            model_format='joblib',
            stock_symbol=stock_symbol,
            trained_at=datetime.utcnow(),
            metrics_json=to_native(metrics_quality_long),
            is_active=True
        )
        db.add(model_bl_record)
        
        # Save Model B Short record
        model_bs_record = Model(
            name=f"{model_name}_quality_short",
            model_type='xgboost_v2_quality_short',
            model_path=quality_short_path,
            model_format='joblib',
            stock_symbol=stock_symbol,
            trained_at=datetime.utcnow(),
            metrics_json=to_native(metrics_quality_short),
            is_active=True
        )
        db.add(model_bs_record)
        
        db.commit()
        
        logger.info(f"✓ V2 models saved for {stock_symbol}")
        
        return {
            "status": "success",
            "stock_symbol": stock_symbol,
            "engine_version": "v2",
            "models": {
                "direction": {
                    "path": direction_path,
                    "metrics": to_native(metrics_direction)
                },
                "quality_long": {
                    "path": quality_long_path,
                    "metrics": to_native(metrics_quality_long)
                },
                "quality_short": {
                    "path": quality_short_path,
                    "metrics": to_native(metrics_quality_short)
                }
            },
            "samples": len(df),
            "date_range": f"{start_date} to {end_date}"
        }
    
    except Exception as e:
        logger.error(f"[{task_id}] V2 training failed: {str(e)}", exc_info=True)
        db.rollback()
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True)
def scheduled_retrain_v2(self):
    """Weekly V2 retraining task."""
    if settings.TRADING_ENGINE_VERSION != 'v2':
        logger.info("V2 retrain skipped - engine version is v1")
        return {"status": "skipped", "reason": "v1_mode"}
    
    stock_symbol = settings.DEFAULT_RETRAIN_STOCK
    logger.info(f"Scheduled V2 retrain for {stock_symbol}")
    
    return train_v2_models.delay(
        stock_symbol=stock_symbol,
        model_name=f"auto_v2_retrain_{datetime.utcnow().strftime('%Y%m%d')}"
    )
