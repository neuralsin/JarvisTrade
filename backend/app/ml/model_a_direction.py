"""
V2 Model A: Direction Scout
Multi-class XGBoost predicting directional bias.

Classes:
- 0: Neutral (no clear direction)
- 1: Long (bullish bias)
- 2: Short (bearish bias)

Labels are generated dynamically based on ATR-adjusted thresholds.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class DirectionScout:
    """
    V2 Model A: Predicts directional bias (Long/Short/Neutral).
    
    This model focuses on momentum features to determine if a directional
    move is likely. It does NOT predict profitability - that's Model B's job.
    """
    
    FEATURE_COLUMNS = [
        'returns_1', 'returns_5', 'returns_10',
        'ema_slope_20', 'ema_slope_50',
        'rsi_14', 'rsi_slope',
        'macd_hist',
        'adx_14'
    ]
    
    def __init__(self, hyperparams: Optional[Dict] = None):
        """
        Initialize Direction Scout.
        
        Args:
            hyperparams: XGBoost hyperparameters override
        """
        self.hyperparams = hyperparams or {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_weights = {0: 0.5, 1: 1.0, 2: 1.0}  # Downweight neutral
        self.feature_importance: Optional[Dict] = None
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum-focused features for direction prediction.
        
        All features are LAGGED (shift(1)) to prevent lookahead bias.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        
        # Validate required columns
        required_cols = ['close', 'high', 'low', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature computation: {missing}")
        
        logger.info(f"Computing Direction Scout features on {len(df)} rows...")
        
        try:
            # Returns (lagged)
            df['returns_1'] = df['close'].pct_change(1).shift(1)
            df['returns_5'] = df['close'].pct_change(5).shift(1)
            df['returns_10'] = df['close'].pct_change(10).shift(1)
            
            # EMA Slopes (lagged)
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_slope_20'] = ema_20.pct_change(5).shift(1)
            df['ema_slope_50'] = ema_50.pct_change(5).shift(1)
            
            # RSI (lagged)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1)
            df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)
            df['rsi_slope'] = df['rsi_14'].diff(1)
            
            # MACD Histogram (lagged)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            df['macd_hist'] = (macd - signal).shift(1)
            
            # ADX (lagged) - import from regime detector
            if 'adx_14' not in df.columns:
                from app.ml.regime_detector import RegimeDetector
                detector = RegimeDetector()
                df = detector.compute_indicators(df)
            
            if 'adx_14' in df.columns:
                df['adx_14'] = df['adx_14'].shift(1)
            else:
                logger.warning("ADX computation failed, using zero")
                df['adx_14'] = 0
            
            # Fill any remaining NaNs with 0 for feature columns only
            for col in self.FEATURE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            logger.info(f"✓ Feature computation complete. Non-null rows: {df[self.FEATURE_COLUMNS].dropna().shape[0]}")
            
        except Exception as e:
            logger.error(f"Feature computation failed: {str(e)}")
            raise
        
        return df
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        lookahead_bars: int = 8,
        threshold_mult: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate dynamic direction labels based on ATR-adjusted thresholds.
        
        Labels:
        - 0: Neutral (return within threshold)
        - 1: Long (return > threshold)
        - 2: Short (return < -threshold)
        
        Args:
            df: DataFrame with OHLCV and atr_percent columns
            lookahead_bars: Number of bars to look ahead for return
            threshold_mult: Multiplier for ATR-based threshold
            
        Returns:
            DataFrame with 'direction_label' column
        """
        df = df.copy()
        
        # Ensure ATR percent is computed
        if 'atr_percent' not in df.columns:
            from app.ml.regime_detector import RegimeDetector
            detector = RegimeDetector()
            df = detector.compute_indicators(df)
        
        labels = []
        for i in range(len(df)):
            if i + lookahead_bars >= len(df):
                labels.append(None)
                continue
            
            current_close = df['close'].iloc[i]
            future_close = df['close'].iloc[i + lookahead_bars]
            atr_pct = df['atr_percent'].iloc[i]
            
            if pd.isna(atr_pct) or atr_pct == 0:
                labels.append(None)
                continue
            
            threshold = threshold_mult * atr_pct
            ret = (future_close - current_close) / current_close
            
            if ret > threshold:
                labels.append(1)  # Long
            elif ret < -threshold:
                labels.append(2)  # Short
            else:
                labels.append(0)  # Neutral
        
        df['direction_label'] = labels
        
        # Log distribution
        valid = df['direction_label'].dropna()
        if len(valid) > 0:
            neutral_pct = (valid == 0).mean() * 100
            long_pct = (valid == 1).mean() * 100
            short_pct = (valid == 2).mean() * 100
            logger.info(f"Direction labels: Neutral={neutral_pct:.1f}%, Long={long_pct:.1f}%, Short={short_pct:.1f}%")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare train/val/test splits (time-series, no shuffle).
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        df = df.dropna(subset=self.FEATURE_COLUMNS + ['direction_label'])
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} samples (need >= 100)")
        
        X = df[self.FEATURE_COLUMNS].values
        y = df['direction_label'].values.astype(int)
        
        # Time-series split (no shuffle to preserve temporal order)
        # Ensure minimum sizes: train 70%, val 15%, test 15%
        # For small datasets, use at least 10 samples for val/test
        n_samples = len(X)
        
        if n_samples < 200:
            # Small dataset: use fixed minimum sizes
            min_val = max(10, int(0.10 * n_samples))
            min_test = max(10, int(0.10 * n_samples))
            train_size = n_samples - min_val - min_test
        else:
            # Normal split
            train_size = int(0.70 * n_samples)
            min_val = int(0.15 * n_samples)
        
        val_size = min(int(0.15 * n_samples), n_samples - train_size - 10)
        val_size = max(val_size, 10)  # At least 10 samples
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Validate all sets have data
        if len(X_train) < 50:
            raise ValueError(f"Training set too small: {len(X_train)} (need >= 50)")
        if len(X_val) < 5:
            raise ValueError(f"Validation set too small: {len(X_val)} (need >= 5)")
        if len(X_test) < 5:
            raise ValueError(f"Test set too small: {len(X_test)} (need >= 5)")
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> xgb.XGBClassifier:
        """
        Train the direction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained XGBClassifier
        """
        # Compute sample weights for class balancing
        sample_weights = np.array([self.class_weights[int(y)] for y in y_train])
        
        self.model = xgb.XGBClassifier(**self.hyperparams)
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_COLUMNS,
            self.model.feature_importances_
        ))
        
        logger.info(f"Direction model trained. Best iteration: {self.model.best_iteration}")
        logger.info(f"Top features: {sorted(self.feature_importance.items(), key=lambda x: -x[1])[:3]}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Predict direction probabilities.
        
        Args:
            X: Feature array (1 or more samples)
            
        Returns:
            Dict with probabilities and predicted direction
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        probs = self.model.predict_proba(X)
        
        # Single sample case
        if len(probs) == 1:
            return {
                'p_neutral': float(probs[0, 0]),
                'p_long': float(probs[0, 1]),
                'p_short': float(probs[0, 2]),
                'direction': int(np.argmax(probs[0])),
                'confidence': float(np.max(probs[0]))
            }
        
        # Batch case
        return {
            'p_neutral': probs[:, 0].tolist(),
            'p_long': probs[:, 1].tolist(),
            'p_short': probs[:, 2].tolist(),
            'direction': np.argmax(probs, axis=1).tolist(),
            'confidence': np.max(probs, axis=1).tolist()
        }
    
    def predict_single(self, features: Dict) -> Dict:
        """
        Predict from feature dictionary (for live inference).
        
        Args:
            features: Dict with feature values
            
        Returns:
            Prediction dict
        """
        X = np.array([[features.get(col, 0) for col in self.FEATURE_COLUMNS]])
        return self.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance with inversion detection.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # =========================================================================
        # INVERTED SIGNAL WEAPONIZATION - Per-class inversion detection
        # For multi-class, we check Long and Short classes separately
        # =========================================================================
        from app.ml.model_inverter import detect_model_state, ModelState
        
        # Long class (class 1) inversion check
        y_long_binary = (y_test == 1).astype(int)
        y_long_prob = y_proba[:, 1]
        
        long_state, long_meta = detect_model_state(y_long_binary, y_long_prob)
        if long_state == ModelState.INVERTIBLE:
            logger.info(f"🔄 Long class INVERTIBLE: Raw AUC={long_meta['raw_auc']:.4f}, Flipped={long_meta['flipped_auc']:.4f}")
            auc_long = long_meta['flipped_auc']
            is_long_inverted = True
        else:
            auc_long = long_meta.get('raw_auc', 0.5)
            is_long_inverted = False
        
        # Short class (class 2) inversion check
        y_short_binary = (y_test == 2).astype(int)
        y_short_prob = y_proba[:, 2]
        
        short_state, short_meta = detect_model_state(y_short_binary, y_short_prob)
        if short_state == ModelState.INVERTIBLE:
            logger.info(f"🔄 Short class INVERTIBLE: Raw AUC={short_meta['raw_auc']:.4f}, Flipped={short_meta['flipped_auc']:.4f}")
            auc_short = short_meta['flipped_auc']
            is_short_inverted = True
        else:
            auc_short = short_meta.get('raw_auc', 0.5)
            is_short_inverted = False
        
        # Determine overall model state
        if is_long_inverted or is_short_inverted:
            model_state = ModelState.INVERTED
            is_inverted = True
        elif long_state == ModelState.REJECT and short_state == ModelState.REJECT:
            model_state = ModelState.REJECT
            is_inverted = False
        else:
            model_state = ModelState.NORMAL
            is_inverted = False
        
        metrics = {
            'accuracy': float(report['accuracy']),
            'auc_long': float(auc_long),
            'auc_short': float(auc_short),
            'precision_neutral': float(report.get('0', {}).get('precision', 0)),
            'precision_long': float(report.get('1', {}).get('precision', 0)),
            'precision_short': float(report.get('2', {}).get('precision', 0)),
            'recall_long': float(report.get('1', {}).get('recall', 0)),
            'recall_short': float(report.get('2', {}).get('recall', 0)),
            'f1_long': float(report.get('1', {}).get('f1-score', 0)),
            'f1_short': float(report.get('2', {}).get('f1-score', 0)),
            'support_neutral': int(report.get('0', {}).get('support', 0)),
            'support_long': int(report.get('1', {}).get('support', 0)),
            'support_short': int(report.get('2', {}).get('support', 0)),
            'feature_importance': self.feature_importance,
            
            # Inverted Signal Weaponization
            'is_inverted': is_inverted,
            'model_state': model_state,
            'is_long_inverted': is_long_inverted,
            'is_short_inverted': is_short_inverted,
            'inversion_metadata': {
                'long': long_meta,
                'short': short_meta
            }
        }
        
        # Store inversion state for prediction
        self.is_long_inverted = is_long_inverted
        self.is_short_inverted = is_short_inverted
        
        logger.info(f"Direction model evaluation: Accuracy={metrics['accuracy']:.3f}, "
                    f"AUC_Long={metrics['auc_long']:.3f}, AUC_Short={metrics['auc_short']:.3f}")
        if is_inverted:
            logger.info(f"🔄 INVERTED: Long={is_long_inverted}, Short={is_short_inverted}")
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'hyperparams': self.hyperparams,
            'class_weights': self.class_weights
        }, path)
        logger.info(f"Direction model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_importance = data.get('feature_importance')
        self.hyperparams = data.get('hyperparams', self.hyperparams)
        self.class_weights = data.get('class_weights', self.class_weights)
        logger.info(f"Direction model loaded from {path}")
