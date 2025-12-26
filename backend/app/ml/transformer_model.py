"""
Transformer model for time series prediction
Attention-based architecture with positional encoding
GPU-accelerated with CUDA support
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logger = logging.getLogger(__name__)

# Configure TensorFlow to use GPU if available
try:
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✓ CUDA enabled! Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        logger.info(f"  TensorFlow will use GPU for transformer training")
    else:
        logger.warning("No GPU found. Transformer will train on CPU (slow!)")
except Exception as e:
    logger.warning(f"GPU configuration failed: {e}. Using CPU for transformer training")


class PositionalEncoding(layers.Layer):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


class TransformerPredictor:
    """
    Transformer-based price prediction model
    """
    def __init__(
        self,
        seq_length=100,
        features=20,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dff=512
    ):
        self.seq_length = seq_length
        self.features = features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self):
        """
        Build Transformer architecture
        """
        inputs = keras.Input(shape=(self.seq_length, self.features))
        
        # Project features to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = PositionalEncoding(self.d_model, self.seq_length)
        x = pos_encoding(x)
        
        # Stack transformer encoder layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # Add & Norm
            x1 = layers.Add()([x, attention_output])
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            
            # Feed forward network
            ffn_output = layers.Dense(self.dff, activation='relu')(x1)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            
            # Add & Norm
            x = layers.Add()([x1, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, df):
        """
        Convert dataframe to sequences
        """
        feature_cols = [col for col in df.columns if col not in ['target', 'ts_utc']]
        
        # Normalize
        scaled_data = self.scaler.fit_transform(df[feature_cols].values)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i + self.seq_length])
            if 'target' in df.columns:
                y.append(df.iloc[i + self.seq_length]['target'])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train transformer model
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.000001
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
    
    def get_attention_weights(self, X):
        """
        Extract attention weights for interpretability
        """
        # Create a model that outputs attention weights
        attention_model = keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in self.model.layers if 'multi_head_attention' in layer.name]
        )
        
        attention_weights = attention_model.predict(X)
        return attention_weights
    
    def save_model(self, path):
        """
        Save model
        """
        self.model.save(f"{path}/transformer_model.h5")
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Transformer model saved to {path}")
    
    def load_model(self, path):
        """
        Load model
        """
        self.model = keras.models.load_model(f"{path}/transformer_model.h5")
        with open(f"{path}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Transformer model loaded from {path}")
