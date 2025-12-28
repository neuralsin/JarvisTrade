# JarvisTrade - System Logic & Conditions Reference

> **Complete technical reference** for all business logic, trading conditions, and decision rules used in the JarvisTrade ML trading system.

---

## Table of Contents

1. [Model Training Pipeline](#1-model-training-pipeline)
2. [Feature Engineering](#2-feature-engineering)
3. [Signal Generation](#3-signal-generation)
4. [Trading Execution Logic](#4-trading-execution-logic)
5. [Position Monitoring](#5-position-monitoring)
6. [Data Fetching](#6-data-fetching)
7. [News Sentiment Analysis](#7-news-sentiment-analysis)
8. [Celery Task Scheduling](#8-celery-task-scheduling)
9. [Model Quality Thresholds](#9-model-quality-thresholds)
10. [Database Schema Constraints](#10-database-schema-constraints)

---

## 1. Model Training Pipeline

### File: `backend/app/tasks/model_training.py`

#### 1.1 Minimum Training Samples
```python
MIN_TRAINING_SAMPLES = 100
```
**Location**: Line ~35  
**Logic**: Training fails if fewer than 100 clean samples after feature computation and label generation.

---

#### 1.2 Data Split Ratios
**Location**: `_train_xgboost()`, `_train_lstm()`, `_train_transformer()` functions

| Split | Percentage |
|-------|------------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

```python
train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))
```

---

#### 1.3 Model Type Handling

| Type | Save Format | File Extension | Loader |
|------|-------------|----------------|--------|
| XGBoost | `joblib` | `.joblib` | `joblib.load()` |
| LSTM | `keras` | Directory | `LSTMPredictor.load_model()` |
| Transformer | `keras` | Directory | `TransformerPredictor.load_model()` |

**Location**: `save_model_artifact()` function (Lines ~45-85)

---

#### 1.4 Default Date Ranges (Yahoo Finance Limits)

**Location**: `train_model()` function (Lines ~230-240)

| Interval | Max Days | Default |
|----------|----------|---------|
| 1m, 5m, 15m, 30m, 1h | 59 days | 59 days |
| 1d | 730 days | 365 days |
| 1wk, 1mo | 730 days | 365 days |

---

#### 1.5 XGBoost Multi-Class Detection

**Location**: `backend/app/ml/trainer.py` (Lines ~70-90)

```python
n_classes = len(np.unique(y_train))
if n_classes == 3:
    objective = 'multi:softprob'
    num_class = 3
else:
    objective = 'binary:logistic'
```

**Classes**:
- 0 = HOLD
- 1 = BUY  
- 2 = SELL

---

#### 1.6 Auto-Activation Thresholds

**Location**: `train_model()` (Lines ~340-350) and `backend/app/config.py`

```python
AUTO_ACTIVATE_MODELS = True
MODEL_MIN_AUC = 0.55
MODEL_MIN_ACCURACY = 0.55
```

**Condition**: Model auto-activates if:
```python
if auc >= MODEL_MIN_AUC and accuracy >= MODEL_MIN_ACCURACY:
    should_activate = True
```

---

## 2. Feature Engineering

### File: `backend/app/ml/feature_engineer.py`

#### 2.1 Feature Columns (Standard Set)

**Location**: Lines ~15-30

```python
FEATURE_COLUMNS = [
    'returns_1',        # 1-day returns
    'returns_5',        # 5-day returns
    'ema_20',           # 20-period EMA
    'ema_50',           # 50-period EMA
    'ema_200',          # 200-period EMA
    'distance_from_ema200',  # Price distance from EMA200
    'rsi_14',           # 14-period RSI
    'rsi_slope',        # RSI momentum
    'atr_14',           # 14-period ATR
    'atr_percent',      # ATR as % of price
    'volume_ratio',     # Current volume / 20-day avg
    'nifty_trend',      # Nifty 50 trend indicator
    'vix',              # India VIX value
    'sentiment_1d',     # 1-day news sentiment
    'sentiment_3d',     # 3-day news sentiment  
    'sentiment_7d'      # 7-day news sentiment
]
```

---

#### 2.2 EMA Drop Condition

**Location**: `compute_features()` function

```python
df = df.dropna(subset=['ema_200', 'atr_14'])
```

**Logic**: Rows without EMA200 or ATR14 are dropped (need 200+ candles for EMA200).

---

#### 2.3 RSI Calculation

**Location**: Lines ~40-60

```python
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

---

#### 2.4 Sentiment Defaults (No News API)

**Location**: Lines ~120-130

```python
if sentiment is None:
    df['sentiment_1d'] = 0.0
    df['sentiment_3d'] = 0.0
    df['sentiment_7d'] = 0.0
```

**Logic**: If `NEWS_API_KEY` not configured, sentiment features default to 0.0 (neutral).

---

## 3. Signal Generation

### File: `backend/app/tasks/signal_generation.py`

#### 3.1 Feature Staleness Check

**Location**: Lines ~25-30

```python
FEATURE_MAX_AGE_SECONDS = 3600  # 1 hour
```

**Condition**: Warning logged if features older than 1 hour, but signal still generated for paper trading.

---

#### 3.2 Duplicate Signal Prevention

**Location**: `_process_model_signal()` (Lines ~210-220)

```python
recent_signal = db.query(Signal).filter(
    Signal.model_id == model_record.id,
    Signal.instrument_id == instrument.id,
    Signal.timestamp >= datetime.utcnow() - timedelta(minutes=5),
    Signal.executed == False
).first()

if recent_signal:
    return  # Skip - recent unexecuted signal exists
```

**Logic**: No duplicate signal created within 5 minutes for same model+stock combination.

---

#### 3.3 XGBoost Signal Thresholds (Multi-Class)

**Location**: `predict_with_model()` (Lines ~80-110)

```python
if len(proba) == 3:  # Multi-class
    signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    predicted_class = model.predict(features)[0]
    confidence = proba[predicted_class]
```

---

#### 3.4 XGBoost Signal Thresholds (Binary)

```python
if len(proba) == 2:  # Binary
    confidence = proba[1]  # Probability of class 1 (BUY)
    if confidence >= 0.3:
        signal = 'BUY'
    else:
        signal = 'HOLD'
```

---

#### 3.5 LSTM/Transformer Signal Thresholds

**Location**: Lines ~115-125

```python
if proba >= 0.6:
    return 'BUY', proba
elif proba <= 0.4:
    return 'SELL', 1.0 - proba
else:
    return 'HOLD', 0.5
```

| Probability | Signal |
|-------------|--------|
| ≥ 0.6 | BUY |
| ≤ 0.4 | SELL |
| 0.4 < p < 0.6 | HOLD |

---

## 4. Trading Execution Logic

### File: `backend/app/tasks/paper_trading.py`

#### 4.1 Paper Trade Entry Conditions

**Location**: `execute_paper_trades()` function

```python
# Only execute BUY or SELL signals
if signal.signal_type not in ['BUY', 'SELL']:
    continue

# Check instrument trading enabled
if not instrument.is_trading_enabled:
    continue

# Check confidence threshold
if signal.confidence < instrument.buy_confidence_threshold:
    continue
```

---

#### 4.2 Position Sizing

**Location**: Lines ~80-100

```python
position_size = min(
    instrument.max_position_size,
    int(capital_per_trade / current_price)
)
```

Default: `max_position_size = 100` shares

---

#### 4.3 Stop Loss & Target Calculation

**Location**: Lines ~110-130

```python
atr = get_latest_atr(instrument_id)

stop_loss = entry_price - (atr * instrument.stop_multiplier)
target = entry_price + (atr * instrument.target_multiplier)
```

| Parameter | Default | Stored In |
|-----------|---------|-----------|
| `stop_multiplier` | 1.5 | instruments table |
| `target_multiplier` | 2.5 | instruments table |

---

## 5. Position Monitoring

### File: `backend/app/tasks/position_monitor.py`

#### 5.1 Exit Conditions

**Location**: `monitor_open_positions()` function

| Condition | Action |
|-----------|--------|
| `current_price <= stop_loss` | EXIT - Stop Loss Hit |
| `current_price >= target` | EXIT - Target Reached |
| Position age > 5 days | EXIT - Time Expired |

```python
if current_price <= position.stop_price:
    exit_reason = 'STOP_LOSS'
elif current_price >= position.target_price:
    exit_reason = 'TARGET'
elif (datetime.utcnow() - position.entry_ts).days > 5:
    exit_reason = 'TIME_EXPIRED'
```

---

#### 5.2 Trailing Stop Logic

**Location**: Lines ~80-100 (if enabled)

```python
if current_price > position.highest_price:
    position.highest_price = current_price
    # Move stop to lock in profits
    new_stop = current_price - (atr * 1.0)
    if new_stop > position.stop_price:
        position.stop_price = new_stop
```

---

## 6. Data Fetching

### File: `backend/app/utils/market_data_fetcher.py`

#### 6.1 Exchange Fallback Order

**Location**: `fetch_ohlcv()` (Lines ~60-80)

```python
exchanges = ['NS', 'BO']  # NSE first, then BSE
```

**Logic**: Try `SYMBOL.NS` first, if not found try `SYMBOL.BO`.

---

#### 6.2 Rate Limit Handling

**Location**: `_fetch_from_yahoo()` (Lines ~100-120)

```python
if response.status_code == 429:
    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
    time.sleep(wait_time)
```

---

#### 6.3 Retry Logic

**Location**: Lines ~90-130

```python
for attempt in range(3):
    try:
        response = requests.get(url, timeout=30)
        # ... process
    except requests.exceptions.Timeout:
        time.sleep(2)
        continue
```

---

## 7. News Sentiment Analysis

### File: `backend/app/ml/sentiment_analyzer.py`

#### 7.1 FinBERT Sentiment Scores

**Location**: `analyze_sentiment()` function

```python
# FinBERT outputs: [negative, neutral, positive]
sentiment_score = probs[2] - probs[0]  # positive - negative
# Result: -1.0 to +1.0
```

---

#### 7.2 Aggregation Windows

**Location**: `getSentimentFeatures()` (Lines ~100-120)

| Feature | Window |
|---------|--------|
| `sentiment_1d` | Last 24 hours |
| `sentiment_3d` | Last 72 hours |
| `sentiment_7d` | Last 168 hours |

---

## 8. Celery Task Scheduling

### File: `backend/app/celery_app.py`

#### 8.1 Beat Schedule (Production)

| Task | Frequency | Purpose |
|------|-----------|---------|
| `compute_fresh_features` | Every 5 min | Update features |
| `generate_signals` | Every 5 min | Generate signals |
| `execute_paper_trades` | Every 2 min | Execute signals |
| `monitor_paper_trades` | Every 2 min | Check P&L |
| `monitor_positions` | Every 2 min | Check exits |
| `fetch_eod_bhavcopy` | Daily 6PM | EOD data |
| `fetch_news_sentiment` | Daily 6AM | News analysis |
| `scheduled_retrain` | Weekly Mon 2AM | Retrain models |
| `detect_model_drift` | Daily 3AM | Drift check |

---

#### 8.2 Worker Settings

**Location**: `celery_app.conf.update()`

```python
task_time_limit = 7200          # 2 hours max
task_soft_time_limit = 6000     # Warning at 100 min
worker_max_tasks_per_child = 1000
worker_max_memory_per_child = 2000000  # 2GB
worker_prefetch_multiplier = 1
```

---

## 9. Model Quality Thresholds

### File: `backend/app/config.py`

```python
# Auto-activation thresholds
AUTO_ACTIVATE_MODELS = True
MODEL_MIN_AUC = 0.55
MODEL_MIN_ACCURACY = 0.55

# XGBoost hyperparameters
XGBOOST_N_ESTIMATORS = 500
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.01
```

---

## 10. Database Schema Constraints

### File: `backend/app/db/schemas.sql`

#### 10.1 Signal Types

```sql
signal_type text NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD'))
```

---

#### 10.2 Trade Modes

```sql
mode text CHECK(mode IN ('paper','live')) NOT NULL
```

---

#### 10.3 Required Fields

| Table | Column | Constraint |
|-------|--------|------------|
| `models` | `stock_symbol` | NOT NULL |
| `models` | `name` | NOT NULL |
| `signals` | `signal_type` | NOT NULL |
| `signals` | `confidence` | NOT NULL (float) |
| `users` | `email` | UNIQUE NOT NULL |
| `instruments` | `symbol` | UNIQUE NOT NULL |

---

## Quick Reference: Threshold Summary

| Parameter | Value | Location |
|-----------|-------|----------|
| Min training samples | 100 | model_training.py |
| Feature staleness | 1 hour | signal_generation.py |
| Signal dedup window | 5 min | signal_generation.py |
| Binary BUY threshold | 0.3 | signal_generation.py |
| Neural BUY threshold | 0.6 | signal_generation.py |
| Neural SELL threshold | 0.4 | signal_generation.py |
| Stop multiplier | 1.5 ATR | instruments table |
| Target multiplier | 2.5 ATR | instruments table |
| Max position age | 5 days | position_monitor.py |
| Auto-activate AUC | 0.55 | config.py |
| Auto-activate accuracy | 0.55 | config.py |

---

*Last Updated: December 28, 2025*
