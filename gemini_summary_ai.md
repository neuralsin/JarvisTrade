# SYSTEM_KERNEL_LOGIC: JARVIS_TRADE

## MODULE_01: FEATURE_ENGINEERING (Input Processor)
**Source**: `backend/app/ml/feature_engineer.py`
**Input**: `OHLCV_DataFrame` (Timezone-Naive, Sorted Ascending)
**Output**: `Feature_Vector` (dim=13)

```python
def COMPUTE_FEATURES(df):
    # 1. EMAs (Trend) - EMA200 DEPRECATED
    df['ema_20']  = EMA(df.close, span=20)
    df['ema_50']  = EMA(df.close, span=50)
    df['ema_100'] = EMA(df.close, span=100) # Replaces EMA200
    df['dist_ema100'] = (df.close - df.ema_100) / df.close

    # 2. Momentum
    df['rsi_14']    = RSI(df.close, period=14)
    df['rsi_slope'] = df.rsi_14.diff(1)

    # 3. Volatility
    df['atr_14']      = ATR(df.high, df.low, df.close, n=14)
    df['atr_percent'] = df.atr_14 / df.close
    
    # 4. Market Context (External)
    df['nifty_trend'] = FETCH_NIFTY_TREND() # 1=UP, -1=DOWN
    df['vix']         = FETCH_INDIA_VIX()   # Volatility Index

    # 5. Volume
    df['vol_ratio'] = df.volume / ROLLING_MEAN(df.volume, 20)

    # 6. Returns (Target Proxies)
    df['ret_1'] = PCT_CHANGE(df.close, 1)
    df['ret_5'] = PCT_CHANGE(df.close, 5)

    # SAFETY_GUARD: NO_SENTIMENT_FEATURES
    # REASON: Zero-bias contamination observed in training
    return df.fillna(0)
```

## MODULE_02: MODEL_OPTIMIZER (Training Logic)
**Source**: `backend/app/ml/trainer.py`
**Objective**: `Binary Classification` (Is_Trade_Worthy?)
**Direction**: `Long Only` (Shorting Disabled)

```python
class TRAINER_LOGIC:
    def EVALUATE(y_true, y_prob):
        # METRIC_01: PRECISION_AT_TOP_K (Critical)
        # Focus: Accuracy of highest confidence predictions
        k = 0.10 # Top 10%
        top_k_indices = argsort(y_prob)[-k*N:]
        precision_at_k = SUM(y_true[top_k_indices]) / LEN(top_k_indices)

        # METRIC_02: AUC_FLIP_CHECK (Calibration)
        auc         = ROC_AUC(y_true, y_prob)
        auc_inverse = ROC_AUC(y_true, 1 - y_prob)
        if auc_inverse > auc:
            y_prob = 1 - y_prob # INVERT_PROBABILITIES
            auc = auc_inverse

        # METRIC_03: DEAD_MODEL_GUARD (Safety)
        variance = STD(y_prob)
        range    = MAX(y_prob) - MIN(y_prob)
        is_dead  = (variance < 0.01) OR (range < 0.05)

        return {
            "auc": auc,
            "p_at_10": precision_at_k,
            "is_dead": is_dead
        }

    def ACTIVATION_DECISION(metrics):
        # HARD_THRESHOLDS
        MIN_AUC = 0.55
        MIN_PRECISION = 0.60

        if metrics.is_dead:
            return REJECT("DEAD_MODEL")
        
        if (metrics.auc >= MIN_AUC) AND (metrics.p_at_10 >= MIN_PRECISION):
            return ACTIVATE()
        
        return REJECT("BELOW_THRESHOLDS")
```

## MODULE_03: SIGNAL_GENERATOR (Inference)
**Source**: `backend/app/tasks/signal_generation.py`
**Constraint**: `Time_Freshness < 3600s`

```python
def GENERATE_SIGNAL(model, features):
    # INPUT_VALIDATION
    if (NOW - features.timestamp) > 3600:
        return NONE # STALE_DATA
    
    # INFERENCE
    prob_trade = model.predict(features) # [0.0 - 1.0]

    # BINARY_DECISION_TREE
    # THRESHOLD = 0.65
    if prob_trade >= 0.65:
        return "BUY", prob_trade
    else:
        return "HOLD", prob_trade
    # NOTE: "SELL" is NOT an ML output. 
    # ML predicts Entry Quality Only.
```

## MODULE_04: EXECUTION_KERNEL (Decision Engine)
**Source**: `backend/app/trading/decision_engine.py`

```python
def EXECUTE_ORDER_CHECK(signal, user_state, market_state):
    # LAYER 1: SYSTEM SAFETY
    if SYSTEM.KILL_SWITCH == TRUE: return ABORT("KILL_SWITCH")
    
    # LAYER 2: USER CONSTRAINTS
    if user_state.daily_loss >= 2.0%: return ABORT("DAILY_LOSS_LIMIT")
    if user_state.trade_count >= MAX_TRADES: return ABORT("MAX_TRADES")
    if signal.prob < 0.50: return ABORT("LOW_CONFIDENCE") # Config: PROB_MIN

    # LAYER 3: MARKET CIRCUIT BREAKERS
    if market_state.nifty_trend != 1: return ABORT("MARKET_BEARISH")
    if market_state.vix > 35.0:       return ABORT("VIX_PANIC")
    if market_state.gap > 3.0%:       return ABORT("GAP_TOO_LARGE")
    if market_state.atr_pct > 5.0%:   return ABORT("ASSET_TOO_VOLATILE")
    if market_state.vol_ratio < 0.5:  return ABORT("LOW_LIQUIDITY")
    if market_state.rsi < 45:         return ABORT("WEAK_MOMENTUM")

    # LAYER 4: POSITION SIZING
    risk_amt = user.balance * 0.01 # 1% Risk
    risk_per_share = entry - stop
    qty = FLOOR(risk_amt / risk_per_share)
    
    return EXECUTE_ORDER(qty)
```

## MODULE_05: EXIT_MONITOR (State Machine)
**Source**: `backend/app/tasks/position_monitor.py`
**Frequency**: `30s`

```python
def CHECK_EXIT(position, current_price, current_features):
    # 1. STOP LOSS (Hard)
    if current_price <= position.stop_price:
        return CLOSE("STOP_HIT")

    # 2. TARGET (Hard)
    if current_price >= position.target_price:
        return CLOSE("TARGET_HIT")

    # 3. PEAK_DETECTION (Dynamic)
    # Source: backend/app/ml/peak_detector.py
    current_profit = (current_price - position.entry) / position.entry
    
    if current_profit > 0.01: # Min 1% Profit to Engage
        peak_score = CALCULATE_PEAK_SCORE(current_features)
        # Score Components:
        # +1: Price near High
        # +1: RSI > 70
        # +1: Volume > 2x Avg
        # +1: Divergence (Price Up, RSI Down)
        # +1: Momentum Slowing

        if peak_score >= 3:
            return CLOSE("PEAK_DETECTED")

    return HOLD
```

## MODULE_06: SYSTEM_CALIBRATION (Exact Specifics)

### A. Label Generation Logic
**Source**: `backend/app/ml/labeler.py`
**Method**: `ONE_VS_REST` (Binary), `ANY_DIRECTION` (Long or Short success = 1)

```python
WINDOW = 20_CANDLES
TARGET = 1.5% (0.015)
STOP   = 0.8% (0.008)

def IS_GOOD_TRADE(idx):
    # Logic: Does price hit Target BEFORE Stop within Window?
    # Valid for Long OR Short
    if (FUTURE_HIGH >= ENTRY * 1.015) AND (MIN_LOW_BEFORE_TARGET > ENTRY * 0.992):
        return 1 # Successful Long
    if (FUTURE_LOW <= ENTRY * 0.985) AND (MAX_HIGH_BEFORE_TARGET < ENTRY * 1.008):
        return 1 # Successful Short
    return 0
```

### B. Universe & Timeframe
**Source**: `backend/app/tasks/data_ingestion.py`
*   **Timeframe**: `15m` (Fifteen Minutes)
*   **Universe (10 Stocks)**:
    `['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']`

### C. Maintenance Operations
**Source**: `celery_beat_config.py`
*   **Retraining**: Daily at `18:00 (6 PM)` local time.
*   **Feature Refresh**: Every `60s`.
*   **Signal Check**: Every `60s`.

### D. Performance Tracking
**Source**: `backend/app/ml/trainer.py`
*   **EV Tracking**: `NULL` (Not explicitly calculated).
*   **Metric**: `Precision` and `AUC` snapshot at training time.
*   **Rolling Performance**: `NULL`.

### E. Execution Limits
**Source**: `backend/app/config.py`
*   **Max trades/day**: `3`.
*   **Signal Conflict**: `FCFS` (First Come, First Served). No priority queuing.
*   **EOD Exit**: `DISABLED` (Positions hold overnight if Stop/Target/Peak not hit).

## MODULE_07: SYSTEM_INTERNALS (Hyperparameters & Data)
**Source**: `backend/app/config.py` & `backend/app/tasks/model_training.py`

### A. Data Fetch Scope
*   **Intraday (15m)**: `59 days` lookback (Yahoo Finance Limit).
*   **Daily**: `365 days` lookback.
*   **Split**: `70% Train` / `15% Val` / `15% Test`.

### B. Model Hyperparameters (Defaults)

| Model | Epochs | Batch | Params | Min Samples |
|-------|--------|-------|--------|-------------|
| **XGBoost** | N/A | N/A | Est=500, Depth=6, LR=0.05 | 100 |
| **LSTM** | 50 | Auto | Seq=60, Units=128, Drop=0.3 | 500 |
| **Transformer** | 100 | 32 | Seq=20, Heads=8, Layers=3 | 1000 |

### C. Call Chain Stack Trace
```text
1. CRON(60s) -> celery_beat
2. TRIGGER -> app.tasks.signal_generation.generate_signals()
3. FETCH -> components: [active_models, user_watchlist]
4. FOR EACH model IN active_models:
   a. DB_READ -> Feature_Vector (Latest)
   b. CHECK -> Freshness < 3600s
   c. LOAD -> Model (Disk/Cache)
   d. CALL -> model.predict(features) -> prob
   e. DB_WRITE -> Signal(prob, type)
   f. BROADCAST -> WebSocket
5. TRIGGER -> app.tasks.execution.execute_trades()
6. FOR EACH signal IN database:
   a. DECISION_ENGINE -> checks (KillSwitch, Safety, etc.)
   b. IF PASS -> EXECUTE -> Broker_API / Paper_Sim
```    
