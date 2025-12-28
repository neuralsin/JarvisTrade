# JarvisTrade - System Logic & Conditions Reference

> **Complete technical reference** for all business logic, trading conditions, and decision rules.
> 
> *Last Updated: December 28, 2025 - Post Interpretation Layer Fixes*

---

## Table of Contents

1. [Model Training Pipeline](#1-model-training-pipeline)
2. [Metric Computation (CRITICAL)](#2-metric-computation-critical)
3. [Model Activation Logic](#3-model-activation-logic)
4. [Model-Specific Logic](#4-model-specific-logic)
5. [Feature Engineering](#5-feature-engineering)
6. [Signal Generation](#6-signal-generation)
7. [Celery Task Scheduling](#7-celery-task-scheduling)
8. [Database Constraints](#8-database-constraints)

---

## 1. Model Training Pipeline

### File: `backend/app/ml/trainer.py`

#### 1.1 Classification Type
```python
# BINARY ONLY - Multi-class removed
objective = 'binary:logistic'
# Labels: 1 = TRADE, 0 = NO_TRADE
```

#### 1.2 Sample Thresholds (ENFORCED)
| Model | Minimum Samples | Enforced |
|-------|-----------------|----------|
| XGBoost | 100 | ✅ |
| LSTM | 500 | ✅ |
| Transformer | 1000 | ✅ |

```python
if model_type == 'transformer' and samples < 1000:
    raise ValueError("Insufficient samples for Transformer")
if model_type == 'lstm' and samples < 500:
    raise ValueError("Insufficient samples for LSTM")
```

---

## 2. Metric Computation (CRITICAL)

### File: `backend/app/ml/trainer.py`

#### 2.1 Precision@TopK (THE METRIC THAT MATTERS)
```python
def precision_at_k(y_true, y_prob, k=0.10):
    n = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:n]
    return float(y_true[idx].sum() / n)
```

#### 2.2 Dead Model Detector (STRENGTHENED)
```python
def is_dead_model(y_prob):
    if np.std(y_prob) < 0.01:       # Was 1e-4
        return True
    if (np.max(y_prob) - np.min(y_prob)) < 0.05:  # NEW
        return True
    return False
```

#### 2.3 AUC Flip Check (CRITICAL)
```python
auc = roc_auc_score(y_binary, y_prob)
auc_flipped = roc_auc_score(y_binary, 1.0 - y_prob)

if auc_flipped > auc:
    y_prob = 1.0 - y_prob  # Correct inverted direction
    auc = auc_flipped
```
**Why**: Model may learn p↑ = bad trade. Both are valid, but needs correction.

#### 2.4 What NOT to Use
| ❌ Metric | Why Wrong |
|----------|-----------|
| Accuracy | Invalid for imbalanced classes |
| SELL signal | Binary model cannot predict direction |

---

## 3. Model Activation Logic

### File: `backend/app/tasks/model_training.py`

#### 3.1 Activation Conditions
```python
if is_dead:
    reject("collapsed predictions")
elif auc >= 0.55 and precision_at_10 >= 0.60:
    activate()
```

#### 3.2 Config Thresholds
| Setting | Value |
|---------|-------|
| `MODEL_MIN_AUC` | 0.55 |
| `MODEL_MIN_PRECISION_AT_10` | 0.60 |

---

## 4. Model-Specific Logic

### 4.1 XGBoost Model
| Parameter | Default |
|-----------|---------|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |

**Best For**: Tabular data, limited samples, fast training

### 4.2 LSTM Model
| Parameter | Default |
|-----------|---------|
| `seq_length` | 60 |
| `hidden_units` | 128 |
| `dropout` | 0.3 |

**Minimum Samples**: 500 (ENFORCED)

### 4.3 Transformer Model
| Parameter | Default |
|-----------|---------|
| `seq_length` | 100 |
| `d_model` | 128 |
| `num_heads` | 8 |
| `num_layers` | 4 |

**Minimum Samples**: 1000 (ENFORCED)
**GPU Required**: Yes

### 4.4 Model Comparison
| Aspect | XGBoost | LSTM | Transformer |
|--------|---------|------|-------------|
| Min Samples | 100 | 500 | 1000 |
| GPU Required | No | Optional | **Yes** |
| Dead Model Risk | Low | Medium | **High** |

---

## 5. Feature Engineering

### File: `backend/app/ml/feature_engineer.py`

#### 5.1 Feature Columns (13 features)
```python
FEATURE_COLUMNS = [
    'returns_1', 'returns_5',
    'ema_20', 'ema_50', 'ema_100',
    'distance_from_ema100',
    'rsi_14', 'rsi_slope',
    'atr_14', 'atr_percent',
    'volume_ratio', 'nifty_trend', 'vix'
]
```
- ✅ EMA reduced from 200 to 100
- ❌ Sentiment features REMOVED

---

## 6. Signal Generation

### File: `backend/app/tasks/signal_generation.py`

#### 6.1 Signal Thresholds (CRITICAL FIX)
```python
# Binary model: predicts trade-worthiness, NOT direction
if confidence >= 0.65:
    signal = 'BUY'
else:
    signal = 'HOLD'
# SELL REMOVED - it's position management, not prediction
```

#### 6.2 Key Principle
> **Binary model predicts "is this trade worth taking?"**
> **NOT "what direction should I trade?"**
> **SELL is position management (stop/target), not ML prediction.**

#### 6.3 Feature Staleness
```python
FEATURE_MAX_AGE_SECONDS = 3600  # 1 hour
```

---

## 7. Celery Task Scheduling

### File: `backend/app/celery_app.py`

#### 7.1 Queue Routing
```python
task_routes = {
    'app.tasks.model_training.*': {'queue': 'training'},
    'app.tasks.signal_generation.*': {'queue': 'signals'},
}
```

---

## 8. Database Constraints

### File: `backend/app/db/schemas.sql`

| Table | Column | Constraint |
|-------|--------|------------|
| `models` | `stock_symbol` | NOT NULL |
| `models` | `model_format` | 'joblib' or 'keras' |
| `signals` | `signal_type` | IN ('BUY', 'HOLD') |

---

## Quick Reference: All Thresholds

| Parameter | Value | Location |
|-----------|-------|----------|
| **AUC threshold** | 0.55 | config.py |
| **Precision@10%** | 0.60 | config.py |
| **BUY threshold** | 0.65 | signal_generation.py |
| **Dead model std** | < 0.01 | trainer.py |
| **Dead model range** | < 0.05 | trainer.py |
| **LSTM min samples** | 500 | model_training.py |
| **Transformer min samples** | 1000 | model_training.py |

---

## Key Principles

> **AUC tells you if the model knows anything.**
> **Precision@TopK tells you if it can make money.**
> **Accuracy tells you almost nothing.**

> **Binary model = trade-worthiness prediction.**
> **SELL = position management, NOT ML prediction.**
