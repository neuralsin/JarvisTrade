-- Architecture Reset Migration
-- Run this to update the database schema for the new architecture

BEGIN;

-- 1. Add model_format column to models table (if not exists)
ALTER TABLE models 
ADD COLUMN IF NOT EXISTS model_format TEXT DEFAULT 'joblib';

COMMENT ON COLUMN models.model_format IS 'Model save format: joblib for XGBoost, keras for LSTM/Transformer';

-- 2. Make stock_symbol NOT NULL (with a default for existing rows)
-- First, update any NULL values to a placeholder
UPDATE models SET stock_symbol = 'UNKNOWN' WHERE stock_symbol IS NULL;

-- Then alter the column to be NOT NULL
ALTER TABLE models ALTER COLUMN stock_symbol SET NOT NULL;

-- Add index on stock_symbol for faster lookups
CREATE INDEX IF NOT EXISTS idx_models_stock_symbol ON models(stock_symbol);

-- 3. Ensure signals table exists (should already from phase3_signals.sql)
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES instruments(id) ON DELETE CASCADE,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    confidence FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_model ON signals(model_id);
CREATE INDEX IF NOT EXISTS idx_signals_instrument ON signals(instrument_id);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed) WHERE executed = false;

COMMIT;

-- Verification
\echo '✅ Architecture Reset Migration Complete'
\echo ''
\echo 'Schema changes applied:'
\echo '  - models.model_format column added'
\echo '  - models.stock_symbol made NOT NULL with index'
\echo '  - signals table verified'
