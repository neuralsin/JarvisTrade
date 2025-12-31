#!/usr/bin/env python
"""
COMPREHENSIVE TRAINING PIPELINE DRY RUN
Finds ALL bugs before attempting real training
"""
import sys
import traceback

print("=" * 60)
print("TRAINING PIPELINE COMPREHENSIVE DEBUG")
print("=" * 60)

errors_found = []

# ============================================================================
# TEST 1: Database Connectivity
# ============================================================================
print("\n[1/10] Testing database connectivity...")
try:
    from app.db.database import SessionLocal
    db = SessionLocal()
    db.execute("SELECT 1")
    print("  ✓ Database connection OK")
    db.close()
except Exception as e:
    errors_found.append(f"DB Connection: {e}")
    print(f"  ✗ FAILED: {e}")

# ============================================================================
# TEST 2: Model Table Schema
# ============================================================================
print("\n[2/10] Checking Model table schema...")
try:
    from app.db.database import SessionLocal
    from sqlalchemy import inspect
    db = SessionLocal()
    inspector = inspect(db.bind)
    columns = {c['name'] for c in inspector.get_columns('models')}
    required = {'id', 'name', 'model_type', 'model_path', 'stock_symbol', 
                'is_active', 'is_inverted', 'model_state', 'inversion_metadata'}
    missing = required - columns
    if missing:
        errors_found.append(f"Missing DB columns: {missing}")
        print(f"  ✗ Missing columns: {missing}")
    else:
        print(f"  ✓ All required columns present")
    db.close()
except Exception as e:
    errors_found.append(f"Model table check: {e}")
    print(f"  ✗ FAILED: {e}")

# ============================================================================
# TEST 3: Data Fetch (Yahoo Finance)
# ============================================================================
print("\n[3/10] Testing Yahoo Finance data fetch...")
try:
    from app.tasks.data_ingestion import fetch_historical_yahoo
    from datetime import datetime, timedelta
    
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    df = fetch_historical_yahoo('TATAELXSI', start, end, '15m', 'NS')
    if df is None or df.empty:
        errors_found.append("Yahoo Finance returned no data for TATAELXSI")
        print(f"  ✗ No data returned")
    else:
        print(f"  ✓ Fetched {len(df)} rows")
        print(f"    Columns: {list(df.columns)}")
        
        # Check for ts_utc column
        if 'ts_utc' not in df.columns:
            if 'timestamp' in df.columns or 'Datetime' in df.columns or 'Date' in df.columns:
                print(f"  ⚠ Need to rename timestamp column")
            else:
                errors_found.append("No timestamp column in data")
                print(f"  ✗ No timestamp column found")
except Exception as e:
    errors_found.append(f"Data fetch: {e}")
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 4: Feature Engineer Import
# ============================================================================
print("\n[4/10] Testing feature engineer...")
try:
    from app.ml.feature_engineer import compute_features
    print("  ✓ Feature engineer imports OK")
except Exception as e:
    errors_found.append(f"Feature engineer import: {e}")
    print(f"  ✗ FAILED: {e}")

# ============================================================================
# TEST 5: Labeler Import
# ============================================================================
print("\n[5/10] Testing labeler...")
try:
    from app.ml.labeler import generate_labels, generate_labels_v2
    print("  ✓ Labeler imports OK")
except Exception as e:
    errors_found.append(f"Labeler import: {e}")
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 6: Trainer Import
# ============================================================================
print("\n[6/10] Testing trainer...")
try:
    from app.ml.trainer import ModelTrainer
    trainer = ModelTrainer()
    print("  ✓ Trainer initializes OK")
except Exception as e:
    errors_found.append(f"Trainer: {e}")
    print(f"  ✗ FAILED: {e}")

# ============================================================================
# TEST 7: Full Pipeline Test (with real data)
# ============================================================================
print("\n[7/10] Testing full pipeline with real data...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from app.tasks.data_ingestion import fetch_historical_yahoo
    from app.ml.feature_engineer import compute_features
    from app.ml.labeler import generate_labels
    from app.ml.trainer import ModelTrainer
    
    # Fetch data
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df_raw = fetch_historical_yahoo('TATAELXSI', start, end, '1d', 'NS')
    
    if df_raw is None or len(df_raw) < 100:
        errors_found.append(f"Insufficient data: got {len(df_raw) if df_raw is not None else 0}")
        print(f"  ✗ Insufficient data")
    else:
        print(f"  Step 7a: Got {len(df_raw)} raw rows")
        print(f"    Raw columns: {list(df_raw.columns)}")
        
        # Normalize columns
        df_raw = df_raw.reset_index()
        # Rename timestamp column
        for col in ['timestamp', 'Datetime', 'Date', 'index']:
            if col in df_raw.columns:
                df_raw = df_raw.rename(columns={col: 'ts_utc'})
                break
        df_raw.columns = [c.lower() for c in df_raw.columns]
        print(f"    Normalized columns: {list(df_raw.columns)}")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'ts_utc']
        missing_cols = [c for c in required_cols if c not in df_raw.columns]
        if missing_cols:
            errors_found.append(f"Missing columns after normalization: {missing_cols}")
            print(f"  ✗ Missing columns: {missing_cols}")
        else:
            # Compute features
            df_features = compute_features(df_raw, 'test', None)
            print(f"  Step 7b: Got {len(df_features)} feature rows")
            
            if len(df_features) == 0:
                errors_found.append("Feature computation returned 0 rows")
                print(f"  ✗ No features computed")
            else:
                # Generate labels
                df_labeled = generate_labels(df_features)
                print(f"  Step 7c: Got {len(df_labeled)} labeled rows")
                
                if 'target' not in df_labeled.columns:
                    errors_found.append("No 'target' column after labeling")
                    print(f"  ✗ No target column")
                else:
                    # Remove nulls
                    df_labeled = df_labeled[df_labeled['target'].notna()]
                    print(f"  Step 7d: {len(df_labeled)} valid samples")
                    
                    if len(df_labeled) < 100:
                        errors_found.append(f"Only {len(df_labeled)} samples after labeling")
                        print(f"  ✗ Insufficient samples")
                    else:
                        # Prepare for training
                        trainer = ModelTrainer()
                        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = trainer.prepare_data(df_labeled)
                        print(f"  Step 7e: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                        
                        # Train
                        model = trainer.train(X_train, y_train, X_val, y_val)
                        print(f"  ✓ Training completed!")
                        
                        # Evaluate
                        metrics = trainer.evaluate(model, X_test, y_test, feature_cols)
                        print(f"  ✓ AUC: {metrics.get('auc_roc', 'N/A')}")

except Exception as e:
    errors_found.append(f"Full pipeline: {e}")
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 8: V3 Modules
# ============================================================================
print("\n[8/10] Testing V3 modules...")
try:
    from app.ml.execution_simulator import ExecutionSimulator, get_labeler_simulator
    print("  ✓ ExecutionSimulator OK")
except Exception as e:
    print(f"  ⚠ ExecutionSimulator: {e} (optional)")

try:
    from app.ml.calibration import ProbabilityCalibrator
    print("  ✓ Calibration OK")
except Exception as e:
    print(f"  ⚠ Calibration: {e} (optional)")

try:
    from app.data.news_sentiment import get_news_provider
    print("  ✓ News Sentiment OK")
except Exception as e:
    print(f"  ⚠ News Sentiment: {e} (optional)")

# ============================================================================
# TEST 9: Celery Task Import
# ============================================================================
print("\n[9/10] Testing Celery tasks...")
try:
    from app.tasks.model_training import train_model
    print(f"  ✓ train_model task: {train_model.name}")
except Exception as e:
    errors_found.append(f"train_model import: {e}")
    print(f"  ✗ FAILED: {e}")

try:
    from app.tasks.model_training_v2 import train_v2_models
    print(f"  ✓ train_v2_models task: {train_v2_models.name}")
except Exception as e:
    print(f"  ⚠ train_v2_models: {e} (optional)")

# ============================================================================
# TEST 10: Model Save Path
# ============================================================================
print("\n[10/10] Testing model save path...")
try:
    from pathlib import Path
    import os
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    test_file = models_dir / "test_write.tmp"
    test_file.write_text("test")
    test_file.unlink()
    print(f"  ✓ Model directory writable: {models_dir.absolute()}")
except Exception as e:
    errors_found.append(f"Model save path: {e}")
    print(f"  ✗ FAILED: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if errors_found:
    print(f"\n❌ FOUND {len(errors_found)} ERRORS:\n")
    for i, err in enumerate(errors_found, 1):
        print(f"  {i}. {err}")
    print("\nFix these issues before training can succeed.")
    sys.exit(1)
else:
    print("\n✓ ALL TESTS PASSED - Training pipeline is ready!")
    sys.exit(0)
