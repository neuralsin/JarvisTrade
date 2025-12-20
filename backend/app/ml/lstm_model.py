"""
LSTM model implementation for time series prediction
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM-based price prediction model
    """
    def __init__(self, seq_length=60, features=20, hidden_units=128):
        self.seq_length = seq_length
        self.features = features
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self):
        """
        Build LSTM architecture
        """
        model = keras.Sequential([
            layers.LSTM(self.hidden_units, return_sequences=True, 
                       input_shape=(self.seq_length, self.features)),
            layers.Dropout(0.3),
            layers.LSTM(self.hidden_units // 2, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, df):
        """
        Convert dataframe to sequences for LSTM
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            X: (samples, seq_length, features)
            y: (samples,)
        """
        # Select feature columns (exclude target)
        feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
        
        # Normalize features
        scaled_data = self.scaler.fit_transform(df[feature_cols].values)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i + self.seq_length])
            if 'target' in df.columns:
                y.append(df.iloc[i + self.seq_length]['target'])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train LSTM model
        """
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Generate predictions
        """
        return self.model.predict(X)
    
    def save_model(self, path):
        """
        Save model weights and scaler
        """
        self.model.save(f"{path}/lstm_model.h5")
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"LSTM model saved to {path}")
    
    def load_model(self, path):
        """
        Load model weights and scaler
        """
        self.model = keras.models.load_model(f"{path}/lstm_model.h5")
        with open(f"{path}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"LSTM model loaded from {path}")
