"""
Spec 5: Model training Celery task
"""
from app.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Model, Feature, Instrument, HistoricalCandle
from app.ml.trainer import ModelTrainer
from app.ml.lstm_model import LSTMPredictor
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels
from sqlalchemy import text
import pandas as pd
from datetime import datetime, timedelta
import logging
import os  # CRITICAL: needed for model saving

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def train_model(
    self, 
    model_name: str,
    instrument_filter: str,  # Changed from stock_symbol - matches API parameter name
    model_type: str = 'xgboost',
    interval: str = '15m',  # NEW: Candle interval
    hyperparams: dict = None,
    start_date: str = None,
    end_date: str = None
):
    """
    Spec 5: Train ML model with full pipeline
    Phase 3: Stock-specific model training (one model per stock)
    
    Args:
        model_name: Name for the model
        instrument_filter: Stock symbol to train on (e.g., 'RELIANCE') - REQUIRED
        model_type: 'xgboost', 'lstm', or 'transformer'
        interval: Candle interval ('15m', '1h', '1d', etc.)
        hyperparams: Optional hyperparameter overrides
        start_date: Training start date (YYYY-MM-DD), defaults to 2 years ago
        end_date: Training end date (YYYY-MM-DD), defaults to today
    
    Returns:
        Dict with model_id, metrics, and stock_symbol
    """
    db = SessionLocal()
    task_id = self.request.id
    
    # Map API parameter name to internal variable name
    stock_symbol = instrument_filter
    
    try:
        # Phase 3: Validate stock_symbol is provided
        if not stock_symbol:
            raise ValueError(
                "instrument_filter is required for model training. "
                "Each model must be trained on a specific stock. "
                "Example: instrument_filter='RELIANCE'"
            )
        
        # Normalize stock symbol
        stock_symbol = stock_symbol.strip().upper()
        
        logger.info(f"Starting {model_type} model training for {stock_symbol}: {model_name} (task: {task_id})")
        self.update_state(state='PROGRESS', meta={'status': f'Initializing training for {stock_symbol}...', 'progress': 5})
        
        # Set default date range based on actual data in DB
        if not end_date or not start_date:
            from sqlalchemy import func
            # Query actual date range from features table  
            date_range = db.query(
                func.min(Feature.ts_utc).label('min_date'),
                func.max(Feature.ts_utc).label('max_date')
            ).first()
            
            if date_range and date_range.min_date and date_range.max_date:
                if not start_date:
                    start_date = date_range.min_date.strftime('%Y-%m-%d')
                if not end_date:
                    end_date = date_range.max_date.strftime('%Y-%m-%d')
                logger.info(f"Using actual DB date range: {start_date} to {end_date}")
            else:
                # Fallback if no features exist
                if not end_date:
                    end_date = datetime.utcnow().strftime('%Y-%m-%d')
                if not start_date:
                    start_dt = datetime.utcnow() - timedelta(days=730)
                    start_date = start_dt.strftime('%Y-%m-%d')
                logger.warning(f"No features in DB, using fallback dates: {start_date} to {end_date}")
        
        self.update_state(state='PROGRESS', meta={'status': f'Loading data for {stock_symbol} from {start_date} to {end_date}...', 'progress': 10})
        
        # Phase 3: Load feature data for specific stock only
        query = db.query(Feature).join(Instrument)
        
        # CRITICAL: Filter by stock_symbol (required for Phase 3)
        query = query.filter(Instrument.symbol == stock_symbol)
        
        # Apply date range
        query = query.filter(
            Feature.ts_utc >= start_date,
            Feature.ts_utc <= end_date
        )
        
        features_data = query.all()
        
        if not features_data:
            # Check if there's any data at all
            total_features = db.query(Feature).count()
            total_instruments = db.query(Instrument).count()
            
            logger.warning(f"No features found. Attempting auto-fetch from Yahoo Finance...")
            
            # AUTO-FETCH: Dynamically fetch data from Yahoo Finance if missing
            try:
                from app.tasks.data_ingestion import ingest_historical_data
                from app.ml.labeler import generate_labels
        
                # Determine which symbols to fetch
                # Phase 3: Only fetch the specific stock we're training on
                symbols_to_fetch = [stock_symbol]
                
                
                logger.info(f"Auto-fetching data for {stock_symbol} with interval={interval}")
                
                # Calculate appropriate date range based on interval
                # Yahoo Finance limits: intraday (15m, 1h) = 60 days max, daily (1d) = 730 days max
                auto_end = datetime.utcnow().strftime('%Y-%m-%d')
                
                intraday_intervals = ['1m', '5m', '15m', '30m', '1h']
                if interval in intraday_intervals:
                    # Intraday: max 60 days
                    auto_start = (datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%d')
                    logger.info(f"Using 60-day range for intraday interval {interval}")
                else:
                    # Daily/Weekly/Monthly: can go back further
                    auto_start = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
                    logger.info(f"Using 365-day range for interval {interval}")
                
                
                self.update_state(state='PROGRESS', meta={
                    'status': f'Fetching historical data from Yahoo Finance for {stock_symbol}...',
                    'progress': 15
                })
                
                # Call data ingestion with error handling
                try:
                    ingest_result = ingest_historical_data(
                        symbols=symbols_to_fetch,
                        start_date=auto_start,
                        end_date=auto_end,
                        interval=interval,  # Use user-selected interval
                        exchange='NSE'
                    )
                    
                    logger.info(f"Data ingestion complete: {ingest_result.get('total_candles', 0)} candles")
                    
                    if ingest_result.get('total_candles', 0) == 0:
                        raise ValueError(
                            f"No data fetched for {stock_symbol}. "
                            f"Yahoo Finance may not have data for this symbol with interval={interval}. "
                            f"Try using a different interval (e.g., '1d' instead of '15m') or a different stock."
                        )
                        
                except Exception as fetch_error:
                    logger.error(f"Data fetch failed for {stock_symbol}: {str(fetch_error)}")
                    raise ValueError(
                        f"Failed to fetch data for {stock_symbol}: {str(fetch_error)}. "
                        f"This stock may not be available on Yahoo Finance or the interval '{interval}' "
                        f"may not be supported. Try: 1) Use interval='1d', 2) Try a different stock like RELIANCE, "
                        f"3) Check if data already exists in database."
                    )
                
                # Now compute features and labels for each symbol
                self.update_state(state='PROGRESS', meta={
                    'status': 'Computing features and labels...',
                    'progress': 20
                })
                
                # Track fetch statistics
                fetch_stats = {'success': [], 'failed': [], 'insufficient_data': []}
                
                for symbol in symbols_to_fetch:
                    instrument_obj = db.query(Instrument).filter(Instrument.symbol == symbol).first()
                    if not instrument_obj:
                        continue
                    
                    # Get historical candles
                    candles = db.query(HistoricalCandle).filter(
                        HistoricalCandle.instrument_id == instrument_obj.id,
                        HistoricalCandle.timeframe == interval  # Use user-selected interval
                    ).order_by(HistoricalCandle.ts_utc).all()
                    
                    # DATA VALIDATION: Check for sufficient candles
                    MIN_CANDLES = 200  # Need 200+ for EMA200 calculation
                    if not candles:
                        logger.warning(f"No candles found for {symbol} after ingestion")
                        fetch_stats['failed'].append(symbol)
                        continue
                    
                    if len(candles) < MIN_CANDLES:
                        logger.warning(
                            f"Insufficient candles for {symbol}: {len(candles)} (need {MIN_CANDLES}+ for feature computation)"
                        )
                        fetch_stats['insufficient_data'].append(f"{symbol} ({len(candles)} candles)")
                        continue
                    
                    # Convert to DataFrame
                    candle_data = [{
                        'ts_utc': c.ts_utc,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in candles]
                    
                    df = pd.DataFrame(candle_data)
                    
                    # Compute features
                    df_with_features = compute_features(df, instrument_id=str(instrument_obj.id), db_session=db)
                    
                    # Generate labels
                    df_labeled = generate_labels(df_with_features)
                    
                    # Store features in database
                    for idx, row in df_labeled.iterrows():
                        if pd.isna(row.get('target')):
                            continue
                        
                        feature_dict = {
                            'returns_1': row.get('returns_1'),
                            'returns_5': row.get('returns_5'),
                            'ema_20': row.get('ema_20'),
                            'ema_50': row.get('ema_50'),
                            'ema_200': row.get('ema_200'),
                            'distance_from_ema200': row.get('distance_from_ema200'),
                            'rsi_14': row.get('rsi_14'),
                            'rsi_slope': row.get('rsi_slope'),
                            'atr_14': row.get('atr_14'),
                            'atr_percent': row.get('atr_percent'),
                            'volume_ratio': row.get('volume_ratio'),
                            'nifty_trend': row.get('nifty_trend'),
                            'vix': row.get('vix'),
                            'sentiment_1d': row.get('sentiment_1d', 0.0),
                            'sentiment_3d': row.get('sentiment_3d', 0.0),
                            'sentiment_7d': row.get('sentiment_7d', 0.0)
                        }
                        
                        # Check if feature already exists
                        existing = db.query(Feature).filter(
                            Feature.instrument_id == instrument_obj.id,
                            Feature.ts_utc == row['ts_utc']
                        ).first()
                        
                        if not existing:
                            feature_record = Feature(
                                instrument_id=instrument_obj.id,
                                ts_utc=row['ts_utc'],
                                feature_json=feature_dict,
                                target=int(row['target'])
                            )
                            db.add(feature_record)
                    
                    db.commit()
                    fetch_stats['success'].append(symbol)
                    logger.info(f"✓ Computed and stored features for {symbol}")
                
                # Report fetch statistics
                logger.info(
                    f"Auto-fetch complete: "
                    f"Success={len(fetch_stats['success'])}, "
                    f"Failed={len(fetch_stats['failed'])}, "
                    f"Insufficient data={len(fetch_stats['insufficient_data'])}"
                )
                
                if fetch_stats['success']:
                    logger.info(f"✓ Successfully processed: {', '.join(fetch_stats['success'])}")
                if fetch_stats['failed']:
                    logger.warning(f"⚠ Failed to fetch: {', '.join(fetch_stats['failed'])}")
                if fetch_stats['insufficient_data']:
                    logger.warning(f"⚠ Insufficient data: {', '.join(fetch_stats['insufficient_data'])}")
                
                # Retry feature query after auto-fetch
                features_data = query.all()
                
                if not features_data:
                    error_msg = f"Auto-fetch completed but no usable features generated.\n"
                    error_msg += f"Total candles fetched: {ingest_result.get('total_candles', 0)}\n"
                    error_msg += f"Successfully processed: {len(fetch_stats['success'])} instruments\n"
                    error_msg += f"Failed: {len(fetch_stats['failed'])} instruments\n"
                    error_msg += f"Insufficient data: {len(fetch_stats['insufficient_data'])} instruments\n"
                    
                    if fetch_stats['failed']:
                        error_msg += f"\nFailed symbols: {', '.join(fetch_stats['failed'])}. "
                        error_msg += "These symbols may not exist on Yahoo Finance or use different tickers.\n"
                    
                    if fetch_stats['insufficient_data']:
                        error_msg += f"\nInsufficient data: {', '.join(fetch_stats['insufficient_data'])}. "
                        error_msg += "Need 200+ candles for feature computation (EMA200 calculation).\n"
                    
                    error_msg += "\nSuggestion: Try training without instrument filter to use all available data."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info(
                    f"✓ Auto-fetch successful! Proceeding with training on {len(features_data)} features from "
                    f"{len(fetch_stats['success'])} instrument(s)"
                )
            
            except Exception as auto_fetch_error:
                # If auto-fetch fails, provide detailed error message
                error_msg = f"Training data not found and auto-fetch failed: {str(auto_fetch_error)}. "
                if instrument_filter:
                    error_msg += f"Filter: {instrument_filter}. "
                error_msg += f"Date range: {start_date} to {end_date}. "
                error_msg += f"Total features in DB: {total_features}, Total instruments: {total_instruments}. "
                error_msg += "You may need to check your internet connection or Yahoo Finance availability."
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Convert to DataFrame
        df_rows = []
        for feat in features_data:
            row = feat.feature_json.copy() if feat.feature_json else {}
            row['target'] = feat.target
            row['ts_utc'] = feat.ts_utc
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        logger.info(f"Loaded {len(df)} feature rows from {start_date} to {end_date}")
        
        # CRITICAL VALIDATION: Ensure we have enough data for training
        MIN_SAMPLES = 100  # Minimum samples needed for reliable training
        if len(df) < MIN_SAMPLES:
            error_msg = f"Insufficient training data: {len(df)} samples (need at least {MIN_SAMPLES}). "
            if instrument_filter:
                error_msg += f"Try training without instrument filter or expanding date range."
            else:
                error_msg += f"Please ingest more historical data."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate target distribution
        if 'target' not in df.columns:
            raise ValueError("No 'target' column found in features. Check labeling logic.")
        
        target_dist = df['target'].value_counts()
        if len(target_dist) < 2:
            raise ValueError(f"Target has only one class ({target_dist.index[0]}). Cannot train. Need both buy/sell signals.")
        
        # Check for reasonable class balance (warn if too imbalanced)
        pos_ratio = (df['target'] == 1).sum() / len(df)
        if pos_ratio < 0.01 or pos_ratio > 0.99:
            logger.warning(f"Severe class imbalance: {pos_ratio*100:.1f}% positive samples. Training may be poor.")
        
        # Validate essential features exist
        essential_features = ['rsi_14', 'atr_percent', 'volume_ratio']
        missing_features = [f for f in essential_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}. Feature engineering may be incomplete.")
        
        logger.info(f"✓ Validation passed: {len(df)} samples, {pos_ratio*100:.1f}% positive class")
        self.update_state(state='PROGRESS', meta={'status': f'Preparing {len(df)} samples for {model_type} training...', 'progress': 25})
        
        # Train based on model type
        if model_type == 'lstm':
            metrics, model_path = _train_lstm(df, model_name, hyperparams, self)
        elif model_type == 'transformer':
            metrics, model_path = _train_transformer(df, model_name, hyperparams, self)
        else:  # xgboost
            metrics, model_path = _train_xgboost(df, model_name, hyperparams, self)
        
        # Phase 1: Auto-activate model if it meets quality thresholds
        should_activate = False
        if settings.AUTO_ACTIVATE_MODELS:
            auc = metrics.get('auc_roc', 0)
            accuracy = metrics.get('test_accuracy', metrics.get('accuracy', 0))
            
            if auc >= settings.MODEL_MIN_AUC and accuracy >= settings.MODEL_MIN_ACCURACY:
                should_activate = True
                logger.info(f"Model meets quality thresholds (AUC: {auc:.4f}, Accuracy: {accuracy:.4f}) - auto-activating")
            else:
                logger.info(f"Model below thresholds (AUC: {auc:.4f} < {settings.MODEL_MIN_AUC}, Accuracy: {accuracy:.4f} < {settings.MODEL_MIN_ACCURACY}) - manual activation required")
        
        # Save to database with stock_symbol
        model_record = Model(
            name=model_name,
            model_type=model_type,
            model_path=model_path,
            stock_symbol=stock_symbol,  # Phase 3: Store which stock this model is for
            trained_at=datetime.utcnow(),
            metrics_json=metrics,
            is_active=should_activate  # Auto-activate if quality is good
        )
        db.add(model_record)
        db.commit()
        
        logger.info(f"Model training complete for {stock_symbol}: {model_name} (AUC: {metrics.get('auc_roc', 0):.4f})")
        
        return {
            "status": "success",
            "model_id": str(model_record.id),
            "model_name": model_name,
            "model_type": model_type,
            "stock_symbol": stock_symbol,  # Phase 3: Return which stock
            "metrics": metrics,
            "date_range": f"{start_date} to {end_date}",
            "samples": len(df),
            "auto_activated": should_activate
        }
    
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


def _train_xgboost(df, model_name, hyperparams, task):
    """Train XGBoost model"""
    task.update_state(state='PROGRESS', meta={'status': 'Splitting data into train/validation/test sets...', 'progress': 35})
    
    trainer = ModelTrainer(hyperparams=hyperparams)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = trainer.prepare_data(df)
    
    task.update_state(state='PROGRESS', meta={'status': f'Training XGBoost on {len(X_train)} samples...', 'progress': 50})
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating model performance...', 'progress': 75})
    metrics = trainer.evaluate(model, X_test, y_test, feature_cols)
    
    task.update_state(state='PROGRESS', meta={'status': 'Saving model to disk...', 'progress': 90})
    model_path = trainer.save_model(model, model_name, metrics)
    
    return metrics, model_path


def _train_lstm(df, model_name, hyperparams, task):
    """Train LSTM model"""
    import os
    
    task.update_state(state='PROGRESS', meta={'status': 'Initializing LSTM neural network...', 'progress': 35})
    
    lstm = LSTMPredictor()
    
    # Validate sufficient data
    if len(df) < 200:
        raise ValueError(
            f"LSTM needs at least 200 samples. Got {len(df)}. "
            f"Try: 1) Use longer date range, 2) Use daily (1d) interval, 3) Use XGBoost instead"
        )
    
    # Prepare sequences
    task.update_state(state='PROGRESS', meta={'status': 'Creating time-series sequences...', 'progress': 45})
    X, y = lstm.prepare_sequences(df)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Train
    task.update_state(state='PROGRESS', meta={'status': f'Training LSTM neural network (50 epochs)...', 'progress': 55})
    history = lstm.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating LSTM performance...', 'progress': 80})
    y_pred = lstm.predict(X_test)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Store training distributions for drift detection
    feature_distributions = {}
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    for col in feature_cols[:10]:  # First 10 features
        if col in df.columns:
            feature_distributions[col] = df[col].dropna().values.tolist()[:1000]
    
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, (y_pred > 0.5).astype(int))),
        'samples_train': len(X_train),
        'samples_val': len(X_val),
        'samples_test': len(X_test),
        'feature_distributions': feature_distributions
    }
    
    # Save
    task.update_state(state='PROGRESS', meta={'status': 'Saving LSTM model...', 'progress': 92})
    model_path = f"models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    lstm.save_model(model_path)
    
    return metrics, model_path


def _train_transformer(df, model_name, hyperparams, task):
    """Train Transformer model with multi-head attention"""
    task.update_state(state='PROGRESS', meta={'status': 'Training Transformer', 'progress': 50})
    
    from app.ml.transformer_model import TransformerPredictor
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    logger.info("Starting Transformer model training")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    X = df[feature_cols].values
    y = df['target'].values
    
    # Split data: 70% train, 15% val, 15% test
    total_samples = len(df)
    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Get hyperparameters
    seq_length = hyperparams.get('seq_length', 20)
    d_model = hyperparams.get('d_model', 128)
    num_heads = hyperparams.get('num_heads', 8)
    num_layers = hyperparams.get('num_layers', 3)
    dff = hyperparams.get('dff', 256)
    epochs = hyperparams.get('n_estimators', 100)  # Reuse n_estimators
    batch_size = hyperparams.get('batch_size', 32)
    
    # CRITICAL: Auto-adjust sequence length based on available data
    min_samples_needed = seq_length * 10  # Need at least 10x seq_length for training
    if len(X_train) < min_samples_needed:
        # Reduce sequence length to fit available data
        new_seq_length = max(5, len(X_train) // 20)  # At least 5, prefer 1/20 of data
        logger.warning(f"Insufficient samples ({len(X_train)}) for seq_length={seq_length}. Reducing to {new_seq_length}")
        seq_length = new_seq_length
    
    # Validate minimum samples
    if len(X_train) < 100:
        raise ValueError(
            f"Transformer needs at least 100 samples. Got {len(X_train)}. "
            f"Try: 1) Use longer date range, 2) Use daily (1d) interval instead of 15m, "
            f"3) Use XGBoost which works with smaller datasets"
        )
    
    # Initialize transformer
    transformer = TransformerPredictor(
        seq_length=seq_length,
        features=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dff=dff
    )
    
    # Build model
    transformer.build_model()
    logger.info(f"Transformer model built with {transformer.model.count_params():,} parameters")
    
    # Prepare sequences
    task.update_state(state='PROGRESS', meta={'status': 'Preparing sequences', 'progress': 60})
    X_train_seq, y_train_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_train, y_train]), columns=feature_cols + ['target'])
    )
    X_val_seq, y_val_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_val, y_val]), columns=feature_cols + ['target'])
    )
    X_test_seq, y_test_seq = transformer.prepare_sequences(
        pd.DataFrame(np.column_stack([X_test, y_test]), columns=feature_cols + ['target'])
    )
    
    logger.info(f"Sequences prepared - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # Train model with error handling
    task.update_state(state='PROGRESS', meta={'status': 'Training Transformer network', 'progress': 70})
    try:
        history = transformer.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=epochs,
            batch_size=batch_size
        )
        logger.info("✓ Transformer training completed successfully")
    except Exception as train_error:
        logger.error(f"Transformer training failed: {str(train_error)}")
        raise ValueError(
            f"Transformer training failed: {str(train_error)}. "
            f"This may be due to: 1) Insufficient GPU memory (need 4-8GB), "
            f"2) Data quality issues, 3) TensorFlow/CUDA errors. "
            f"Try: 1) Use XGBoost instead, 2) Use daily (1d) data, 3) Reduce num_layers to 2"
        )
    
    # Evaluate on test set
    task.update_state(state='PROGRESS', meta={'status': 'Evaluating model', 'progress': 90})
    y_pred_proba = transformer.predict(X_test_seq).flatten()
    y_pred = (y_pred_proba > 0.3).astype(int)  # Use lower threshold like XGBoost
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test_seq, y_pred_proba)
    test_acc = accuracy_score(y_test_seq, y_pred)
    test_precision = precision_score(y_test_seq, y_pred, zero_division=0)
    test_recall = recall_score(y_test_seq, y_pred, zero_division=0)
    
    val_loss = history.history['val_loss'][-1]
    val_auc = history.history.get('val_auc_1', history.history.get('val_auc', [test_auc]))[-1]
    
    # Store training distributions for drift detection
    feature_distributions = {}
    feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
    for col in feature_cols[:10]:  # First 10 features
        if col in df.columns:
            feature_distributions[col] = df[col].dropna().values.tolist()[:1000]
    
    metrics = {
        'train_auc': history.history.get('auc_1', history.history.get('auc', [test_auc]))[-1],
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'val_loss': val_loss,
        'epochs_trained': len(history.history['loss']),
        'total_params': int(transformer.model.count_params()),
        'feature_distributions': feature_distributions
    }
    
    # Calculate class metrics
    true_positives = int(((y_pred == 1) & (y_test_seq == 1)).sum())
    false_positives = int(((y_pred == 1) & (y_test_seq == 0)).sum())
    true_negatives = int(((y_pred == 0) & (y_test_seq == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_test_seq == 1)).sum())
    
    logger.info(f"Transformer Test Metrics - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    logger.info(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    
    # Save model
    model_path = f"models/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    transformer.save_model(model_path)
    
    return metrics, model_path


@celery_app.task(bind=True)
def scheduled_retrain(self):
    """
    Spec 12: Weekly retraining task
    """
    # Trigger retraining for all active instruments
    logger.info("Scheduled retrain started")
    
    model_name = f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d')}"
    # TODO: Make this configurable - which stock to retrain on schedule
    return train_model.delay(
        model_name=model_name,
        instrument_filter="RELIANCE",  # Default to RELIANCE, make configurable later
        model_type='xgboost'
    )
