"""
V2 Model B: Quality Gatekeeper
Binary XGBoost predicting P(target hit before stop).

Uses Triple Barrier labeling with path dependency:
- Upper barrier: Entry ± 1.5 * ATR (target)
- Lower barrier: Entry ∓ 1.0 * ATR (stop)
- Time barrier: 24 bars (timeout)

Labels:
- 1: Target hit before stop (win)
- 0: Stop hit before target (loss)
- -1: Timeout (neither hit) - excluded from training
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class QualityGatekeeper:
    """
    V2 Model B: Predicts trade quality P(win).
    
    Uses structure-focused features and Triple Barrier labeling to
    estimate the probability of hitting target before stop.
    """
    
    FEATURE_COLUMNS = [
        'atr_z_score',
        'bollinger_width',
        'volume_force',
        'range_position',
        'hour_sin',
        'hour_cos',
        'distance_from_vwap',
        'direction_context'  # Injected from Model A
    ]
    
    def __init__(self, hyperparams: Optional[Dict] = None):
        """
        Initialize Quality Gatekeeper.
        
        Args:
            hyperparams: XGBoost hyperparameters override
        """
        self.hyperparams = hyperparams or {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_importance: Optional[Dict] = None
        self.direction: int = 1  # 1=Long, 2=Short
    
    def compute_features(self, df: pd.DataFrame, direction: int = 1) -> pd.DataFrame:
        """
        Compute structure-focused features for quality prediction.
        
        All features are LAGGED to prevent lookahead bias.
        
        Args:
            df: DataFrame with OHLCV columns
            direction: Trade direction (1=Long, 2=Short)
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        self.direction = direction
        
        # Validate required columns
        required_cols = ['close', 'high', 'low', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for quality features: {missing}")
        
        logger.info(f"Computing Quality Gatekeeper features on {len(df)} rows (direction={direction})...")
        
        try:
            # ATR Z-Score (from regime detector)
            if 'atr_z' not in df.columns:
                from app.ml.regime_detector import RegimeDetector
                detector = RegimeDetector()
                df = detector.compute_indicators(df)
            
            if 'atr_z' in df.columns:
                df['atr_z_score'] = df['atr_z'].shift(1)
            else:
                logger.warning("ATR Z-score computation failed, using zero")
                df['atr_z_score'] = 0
            
            # Bollinger Width (lagged)
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bollinger_width'] = ((std_20 * 2) / sma_20.replace(0, 1)).shift(1)
            
            # Volume Force (lagged)
            vol_sma = df['volume'].rolling(20).mean()
            df['volume_force'] = (df['volume'] / vol_sma.replace(0, 1)).shift(1)
            
            # Range Position (lagged) - where is price in recent range
            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            range_size = high_20 - low_20
            df['range_position'] = ((df['close'] - low_20) / range_size.replace(0, 1)).shift(1)
            
            # Hour encoding (lagged) - time-of-day patterns
            if 'ts_utc' in df.columns:
                try:
                    hours = pd.to_datetime(df['ts_utc']).dt.hour
                    df['hour_sin'] = np.sin(2 * np.pi * hours / 24).shift(1)
                    df['hour_cos'] = np.cos(2 * np.pi * hours / 24).shift(1)
                except Exception as e:
                    logger.warning(f"Hour encoding failed: {e}")
                    df['hour_sin'] = 0
                    df['hour_cos'] = 0
            else:
                df['hour_sin'] = 0
                df['hour_cos'] = 0
            
            # VWAP Distance (lagged)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cumulative_tp_vol = (typical_price * df['volume']).cumsum()
            cumulative_vol = df['volume'].cumsum()
            df['vwap'] = cumulative_tp_vol / cumulative_vol.replace(0, 1)
            df['distance_from_vwap'] = ((df['close'] - df['vwap']) / df['close'].replace(0, 1)).shift(1)
            
            # Direction context (injected from Model A)
            df['direction_context'] = direction
            
            # Fill NaNs with 0 for feature columns
            for col in self.FEATURE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            logger.info(f"✓ Quality feature computation complete. Non-null rows: {df[self.FEATURE_COLUMNS].dropna().shape[0]}")
            
        except Exception as e:
            logger.error(f"Quality feature computation failed: {str(e)}")
            raise
        
        return df
    
    def generate_labels_triple_barrier(
        self,
        df: pd.DataFrame,
        direction: int,
        target_atr_mult: float = 1.5,
        stop_atr_mult: float = 1.0,
        time_limit_bars: int = 24,
        slippage_pct: float = 0.0005
    ) -> pd.DataFrame:
        """
        Generate Triple Barrier labels (path-dependent).
        
        For each bar, simulates entering a trade and checks which barrier
        is hit first (target, stop, or time).
        
        Labels:
        - 1: Target hit before stop
        - 0: Stop hit before target
        - -1: Timeout (neither hit)
        
        Args:
            df: DataFrame with OHLCV and atr_14 columns
            direction: 1=Long, 2=Short
            target_atr_mult: Target distance as ATR multiple
            stop_atr_mult: Stop distance as ATR multiple
            time_limit_bars: Maximum bars to hold
            slippage_pct: Slippage percentage on entry
            
        Returns:
            DataFrame with 'quality_label' column
        """
        df = df.copy()
        
        # Ensure ATR is computed
        if 'atr_14' not in df.columns:
            from app.ml.regime_detector import RegimeDetector
            detector = RegimeDetector()
            df = detector.compute_indicators(df)
        
        labels = []
        
        for i in range(len(df)):
            # Need enough future data
            if i + time_limit_bars >= len(df):
                labels.append(None)
                continue
            
            atr = df['atr_14'].iloc[i]
            if pd.isna(atr) or atr == 0:
                labels.append(None)
                continue
            
            entry = df['close'].iloc[i]
            
            # Apply slippage based on direction
            if direction == 1:  # Long
                entry_slipped = entry * (1 + slippage_pct)
                target = entry_slipped + (target_atr_mult * atr)
                stop = entry_slipped - (stop_atr_mult * atr)
            else:  # Short
                entry_slipped = entry * (1 - slippage_pct)
                target = entry_slipped - (target_atr_mult * atr)
                stop = entry_slipped + (stop_atr_mult * atr)
            
            # Check path within time window
            future_slice = df.iloc[i+1:i+1+time_limit_bars]
            
            target_hit_bar = None
            stop_hit_bar = None
            
            for j, (idx, row) in enumerate(future_slice.iterrows()):
                if direction == 1:  # Long
                    if row['high'] >= target and target_hit_bar is None:
                        target_hit_bar = j
                    if row['low'] <= stop and stop_hit_bar is None:
                        stop_hit_bar = j
                else:  # Short
                    if row['low'] <= target and target_hit_bar is None:
                        target_hit_bar = j
                    if row['high'] >= stop and stop_hit_bar is None:
                        stop_hit_bar = j
                
                # Early exit if both found
                if target_hit_bar is not None and stop_hit_bar is not None:
                    break
            
            # Determine label based on which barrier hit first
            if target_hit_bar is not None and stop_hit_bar is not None:
                if target_hit_bar < stop_hit_bar:
                    labels.append(1)  # Win
                else:
                    labels.append(0)  # Loss
            elif target_hit_bar is not None:
                labels.append(1)  # Win
            elif stop_hit_bar is not None:
                labels.append(0)  # Loss
            else:
                labels.append(-1)  # Timeout
        
        df['quality_label'] = labels
        
        # Log distribution
        valid = df['quality_label'].dropna()
        if len(valid) > 0:
            win_pct = (valid == 1).sum()
            loss_pct = (valid == 0).sum()
            timeout_pct = (valid == -1).sum()
            total = len(valid)
            logger.info(f"Quality labels (dir={direction}): "
                        f"Win={win_pct}/{total} ({win_pct/total*100:.1f}%), "
                        f"Loss={loss_pct}/{total} ({loss_pct/total*100:.1f}%), "
                        f"Timeout={timeout_pct}/{total} ({timeout_pct/total*100:.1f}%)")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare training data (excludes timeout labels).
        
        Timeouts are NOT counted as wins - they're excluded entirely.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Exclude timeout trades from training
        df_filtered = df[df['quality_label'] >= 0].copy()
        df_filtered = df_filtered.dropna(subset=self.FEATURE_COLUMNS)
        
        if len(df_filtered) < 100:
            raise ValueError(f"Insufficient data: {len(df_filtered)} samples (need >= 100)")
        
        X = df_filtered[self.FEATURE_COLUMNS].values
        y = df_filtered['quality_label'].values.astype(int)
        
        # Time-series split with minimum size guarantees
        n_samples = len(X)
        
        if n_samples < 200:
            # Small dataset: ensure minimum validation/test sizes
            min_val = max(10, int(0.10 * n_samples))
            min_test = max(10, int(0.10 * n_samples))
            train_size = n_samples - min_val - min_test
        else:
            # Normal split
            train_size = int(0.70 * n_samples)
        
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
        
        # Compute scale_pos_weight for class imbalance
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0:
            self.hyperparams['scale_pos_weight'] = neg_count / pos_count
        else:
            self.hyperparams['scale_pos_weight'] = 1.0
            logger.warning("No positive samples in training set - using scale_pos_weight=1.0")
        
        logger.info(f"Quality data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Class balance: Pos={pos_count}, Neg={neg_count}, "
                    f"Scale={self.hyperparams['scale_pos_weight']:.2f}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> xgb.XGBClassifier:
        """
        Train the quality gatekeeper model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained XGBClassifier
        """
        self.model = xgb.XGBClassifier(**self.hyperparams)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_COLUMNS,
            self.model.feature_importances_
        ))
        
        logger.info(f"Quality model trained. Best iteration: {self.model.best_iteration}")
        logger.info(f"Top features: {sorted(self.feature_importance.items(), key=lambda x: -x[1])[:3]}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> float:
        """
        Predict quality probability P(win).
        
        Args:
            X: Feature array (1 sample expected)
            
        Returns:
            Probability of hitting target before stop
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        proba = self.model.predict_proba(X)
        return float(proba[0, 1])
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict quality probabilities for batch."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        proba = self.model.predict_proba(X)
        return proba[:, 1]
    
    def predict_single(self, features: Dict) -> float:
        """
        Predict from feature dictionary (for live inference).
        
        Args:
            features: Dict with feature values
            
        Returns:
            Quality probability
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
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # =========================================================================
        # INVERTED SIGNAL WEAPONIZATION
        # =========================================================================
        from app.ml.model_inverter import detect_model_state, apply_inversion, validate_inverted_model, ModelState
        
        model_state, inversion_meta = detect_model_state(y_test, y_proba)
        
        if model_state == ModelState.INVERTIBLE:
            # Apply inversion and validate
            y_proba_flipped = apply_inversion(y_proba)
            is_valid, validation_meta = validate_inverted_model(
                y_test, y_proba_flipped,
                model_type='xgboost',
                n_samples=len(y_test)
            )
            
            if is_valid:
                logger.info(f"🔄 INVERTED Quality Model: Raw AUC={inversion_meta['raw_auc']:.4f} → Flipped AUC={inversion_meta['flipped_auc']:.4f}")
                y_proba = y_proba_flipped
                y_pred = (y_proba >= 0.5).astype(int)
                model_state = ModelState.INVERTED
                is_inverted = True
                inversion_meta['validation'] = validation_meta
            else:
                logger.warning(f"❌ Inversion validation failed: {validation_meta.get('checks_failed', [])}")
                model_state = ModelState.REJECT
                is_inverted = False
                inversion_meta['validation'] = validation_meta
        elif model_state == ModelState.REJECT:
            is_inverted = False
            logger.warning(f"❌ Quality model rejected: {inversion_meta.get('reason', 'unknown')}")
        else:
            is_inverted = False
        
        # Store inversion state for prediction
        self.is_inverted = is_inverted
        
        metrics = {
            'auc_roc': float(roc_auc_score(y_test, y_proba)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'win_rate_predicted': float(np.mean(y_pred)),
            'actual_win_rate': float(np.mean(y_test)),
            'threshold_calibration': float(np.mean(y_proba)),
            'feature_importance': self.feature_importance,
            
            # Inverted Signal Weaponization
            'is_inverted': is_inverted,
            'model_state': model_state,
            'inversion_metadata': inversion_meta
        }
        
        logger.info(f"Quality model evaluation: AUC={metrics['auc_roc']:.3f}, "
                    f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        if is_inverted:
            logger.info(f"🔄 Quality model INVERTED")
        
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
            'direction': self.direction
        }, path)
        logger.info(f"Quality model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_importance = data.get('feature_importance')
        self.hyperparams = data.get('hyperparams', self.hyperparams)
        self.direction = data.get('direction', 1)
        logger.info(f"Quality model loaded from {path}")
