# Diagnosis Report: Training Failures

## Root Cause Found! 🔴

**Problem**: Database contains ONLY 15-minute candles, NO daily candles

**Data in DB**:
- HINDUNILVR: 1,023 candles (15m interval)
- Other stocks: 15m interval data
- Daily (1d) candles: **0 rows!**

**Why Training Fails**:
1. Frontend defaults to `interval='15m'` ✅
2. But auto-fetch code was hardcoded to fetch `1d` data
3. Database query looks for wrong interval
4. No candles found → training stops at 15%

## Solutions

### Immediate Fix (for existing data):
Train with `interval='15m'` since that's what exists in DB

### Proper Fix:
Make sure auto-fetch uses the same interval as user-selected

**Testing**:
- Use stock: HINDUNILVR  
- Interval: 15m
- Should find 1,023 candles and train successfully
