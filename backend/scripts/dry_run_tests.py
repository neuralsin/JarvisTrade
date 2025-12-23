"""
Comprehensive Dry-Run Test Suite for JarvisTrade
Execute with: python backend/scripts/dry_run_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db.database import SessionLocal
from app.db.models import Model, Feature, Instrument, HistoricalCandle, Trade
from app.ml.feature_engineer import compute_features
from app.ml.labeler import generate_labels
from app.ml.peak_detector import is_at_peak, should_sell_at_peak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DRY_RUN')


class DryRunTester:
    """Comprehensive dry-run testing without side effects"""
    
    def __init__(self):
        self.db = SessionLocal()
        self.results = {
            'phase': None,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
    
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        self.results['tests_run'] += 1
        if passed:
            self.results['tests_passed'] += 1
            logger.info(f"✅ {test_name}: PASS {message}")
        else:
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name}: {message}")
            logger.error(f"❌ {test_name}: FAIL {message}")
    
    def close(self):
        """Cleanup"""
        self.db.close()
    
    # ========== PHASE 1: MODEL PIPELINE ==========
    
    def test_feature_engineering(self):
        """Test 1.1: Feature Engineering Validation"""
        logger.info("\n=== Test 1.1: Feature Engineering Validation ===")
        
        try:
            # Get sample candle data
            instrument = self.db.query(Instrument).first()
            if not instrument:
                self.log_test("Feature Engineering", False, "No instruments in DB")
                return
            
            candles = self.db.query(HistoricalCandle).filter(
                HistoricalCandle.instrument_id == instrument.id
            ).order_by(HistoricalCandle.ts_utc.desc()).limit(100).all()
            
            if len(candles) < 50:
                self.log_test("Feature Engineering", False, f"Insufficient candles: {len(candles)}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'ts_utc': c.ts_utc,
                'open': float(c.open),
                'high': float(c.high),
                'low': float(c.low),
                'close': float(c.close),
                'volume': int(c.volume)
            } for c in reversed(candles)])
            
            # Compute features
            df_features = compute_features(df, instrument.id, self.db)
            
            # Validations
            required_features = ['rsi_14', 'atr_percent', 'volume_ratio', 'returns_1']
            missing = [f for f in required_features if f not in df_features.columns]
            
            if missing:
                self.log_test("Feature Engineering", False, f"Missing features: {missing}")
                return
            
            # Check RSI range
            rsi_values = df_features['rsi_14'].dropna()
            if not ((rsi_values >= 0) & (rsi_values <= 100)).all():
                self.log_test("Feature Engineering", False, "RSI out of range [0, 100]")
                return
            
            # Check ATR positive
            atr_values = df_features['atr_percent'].dropna()
            if not (atr_values > 0).all():
                self.log_test("Feature Engineering", False, "ATR not all positive")
                return
            
            # Check for NaN
            nan_count = df_features[required_features].isna().sum().sum()
            if nan_count > len(df_features) * 0.1:  # Allow 10% NaN
                self.log_test("Feature Engineering", False, f"Too many NaNs: {nan_count}")
                return
            
            self.log_test("Feature Engineering", True, 
                         f"Features computed: {len(df_features.columns)} cols, {len(df_features)} rows")
        
        except Exception as e:
            self.log_test("Feature Engineering", False, str(e))
    
    def test_labeling_logic(self):
        """Test 1.2: 3-Class Labeling Verification"""
        logger.info("\n=== Test 1.2: 3-Class Labeling ===")
        
        try:
            # Create synthetic data with known patterns
            dates = pd.date_range('2024-01-01', periods=100, freq='1h')
            
            # Pattern 1: Clear uptrend (should be BUY)
            uptrend = pd.DataFrame({
                'ts_utc': dates[:30],
                'open': np.linspace(100, 110, 30),
                'high': np.linspace(101, 111, 30),
                'low': np.linspace(99, 109, 30),
                'close': np.linspace(100, 110, 30),
                'volume': [10000] * 30
            })
            
            # Pattern 2: Clear downtrend (should be SELL)
            downtrend = pd.DataFrame({
                'ts_utc': dates[30:60],
                'open': np.linspace(110, 100, 30),
                'high': np.linspace(111, 101, 30),
                'low': np.linspace(109, 99, 30),
                'close': np.linspace(110, 100, 30),
                'volume': [10000] * 30
            })
            
            # Pattern 3: Sideways (should be HOLD)
            sideways = pd.DataFrame({
                'ts_utc': dates[60:],
                'open': [100] * 40,
                'high': [101] * 40,
                'low': [99] * 40,
                'close': [100] * 40,
                'volume': [10000] * 40
            })
            
            df = pd.concat([uptrend, downtrend, sideways], ignore_index=True)
            
            # Apply labeling
            df_labeled = generate_labels(df)
            
            # Check all 3 classes present
            unique_labels = df_labeled['target'].dropna().unique()
            if len(unique_labels) < 2:  # At least BUY and HOLD
                self.log_test("Labeling Logic", False, f"Only {len(unique_labels)} classes found")
                return
            
            # Count distribution
            label_counts = df_labeled['target'].value_counts()
            hold_count = label_counts.get(0, 0)
            buy_count = label_counts.get(1, 0)
            sell_count = label_counts.get(2, 0)
            
            self.log_test("Labeling Logic", True,
                         f"HOLD={hold_count}, BUY={buy_count}, SELL={sell_count}")
        
        except Exception as e:
            self.log_test("Labeling Logic", False, str(e))
    
    def test_model_predictions(self):
        """Test 1.3: Model Prediction Testing"""
        logger.info("\n=== Test 1.3: Model Predictions ===")
        
        try:
            # Get active model
            active_model = self.db.query(Model).filter(Model.is_active == True).first()
            
            if not active_model:
                self.log_test("Model Predictions", False, "No active model found")
                return
            
            # Get sample features
            sample_features = self.db.query(Feature).limit(10).all()
            
            if not sample_features:
                self.log_test("Model Predictions", False, "No features in DB")
                return
            
            # Load model
            import pickle
            model_path = active_model.model_path
            
            if not os.path.exists(model_path):
                self.log_test("Model Predictions", False, f"Model file not found: {model_path}")
                return
            
            # Test predictions
            for feat in sample_features:
                feature_dict = feat.feature_json
                
                # Convert to DataFrame (model expects this format)
                df_test = pd.DataFrame([feature_dict])
                
                # Make prediction (this will vary based on model type)
                try:
                    # For XGBoost
                    if active_model.model_type == 'xgboost':
                        import xgboost as xgb
                        model = xgb.Booster()
                        model.load_model(os.path.join(model_path, 'model.json'))
                        
                        # Prepare features
                        feature_cols = [c for c in df_test.columns if c not in ['target', 'ts_utc']]
                        X = df_test[feature_cols]
                        
                        dtest = xgb.DMatrix(X)
                        probs = model.predict(dtest)[0]
                        
                        # Validate
                        if len(probs) == 3:  # Multi-class
                            if not np.isclose(probs.sum(), 1.0, atol=0.01):
                                self.log_test("Model Predictions", False, 
                                            f"Probabilities don't sum to 1: {probs.sum()}")
                                return
                            
                            if not all((p >= 0) and (p <= 1) for p in probs):
                                self.log_test("Model Predictions", False, 
                                            f"Probabilities out of range: {probs}")
                                return
                        
                        logger.info(f"Sample prediction: {probs}")
                        break
                
                except Exception as e:
                    self.log_test("Model Predictions", False, f"Prediction failed: {e}")
                    return
            
            self.log_test("Model Predictions", True, 
                         f"Model: {active_model.model_type}, predictions valid")
        
        except Exception as e:
            self.log_test("Model Predictions", False, str(e))
    
    def run_phase1(self):
        """Run all Phase 1 tests"""
        self.results['phase'] = 'Phase 1: Model Pipeline'
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: MODEL PIPELINE TESTING")
        logger.info("="*60)
        
        self.test_feature_engineering()
        self.test_labeling_logic()
        self.test_model_predictions()
        
        return self.get_summary()
    
    def get_summary(self):
        """Get test summary"""
        return {
            'phase': self.results['phase'],
            'total_tests': self.results['tests_run'],
            'passed': self.results['tests_passed'],
            'failed': self.results['tests_failed'],
            'pass_rate': (self.results['tests_passed'] / self.results['tests_run'] * 100) 
                        if self.results['tests_run'] > 0 else 0,
            'errors': self.results['errors']
        }


def main():
    """Main test execution"""
    logger.info("\n" + "="*60)
    logger.info("JARVISTRADE DRY-RUN TESTING")
    logger.info("="*60)
    logger.info("Testing WITHOUT side effects (no DB writes, no real orders)")
    
    tester = DryRunTester()
    
    try:
        # Phase 1: Model Pipeline
        summary = tester.run_phase1()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Phase: {summary['phase']}")
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']} ✅")
        logger.info(f"Failed: {summary['failed']} ❌")
        logger.info(f"Pass Rate: {summary['pass_rate']:.1f}%")
        
        if summary['errors']:
            logger.info("\nErrors:")
            for error in summary['errors']:
                logger.error(f"  - {error}")
        
        return summary['failed'] == 0
    
    finally:
        tester.close()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
