-- Phase 2 Database Migration (Raw SQL)
-- Can be run directly in PostgreSQL if you prefer SQL over Python script
--
-- Usage: psql -U postgres -d jarvistrade -f migrations/phase2_schema.sql

BEGIN;

-- 1. Add stock_symbol to models table
ALTER TABLE models 
ADD COLUMN IF NOT EXISTS stock_symbol TEXT;

COMMENT ON COLUMN models.stock_symbol IS 'Stock symbol this model is trained on (e.g., RELIANCE). One model per stock.';

-- 2. Add selected_model_ids to users table
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS selected_model_ids JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN users.selected_model_ids IS 'Array of model UUIDs that are active for this user. Enables multi-model parallel execution.';

-- 3. Add paper_trading_enabled to users table
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS paper_trading_enabled BOOLEAN DEFAULT true;

COMMENT ON COLUMN users.paper_trading_enabled IS 'Master switch for paper trading. When false, all paper trading is disabled for this user.';

-- 4. Create signal_logs table
CREATE TABLE IF NOT EXISTS signal_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts_utc TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- References
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    instrument_id UUID NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Decision details
    probability FLOAT,
    action TEXT NOT NULL CHECK (action IN ('EXECUTE', 'REJECT')),
    reason TEXT,
    
    -- Context snapshots
    filters_passed JSONB,
    feature_snapshot JSONB,
    
    -- Link to trade if executed
    trade_id UUID REFERENCES trades(id) ON DELETE SET NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_signal_logs_user_ts 
ON signal_logs(user_id, ts_utc DESC);

CREATE INDEX IF NOT EXISTS idx_signal_logs_model 
ON signal_logs(model_id);

CREATE INDEX IF NOT EXISTS idx_signal_logs_instrument 
ON signal_logs(instrument_id);

CREATE INDEX IF NOT EXISTS idx_signal_logs_action 
ON signal_logs(action);

COMMENT ON TABLE signal_logs IS 'Phase 2: Tracks all trading signals (EXECUTE and REJECT) for real-time monitoring and historical analysis';

-- 5. Update existing users (backfill)
UPDATE users 
SET paper_trading_enabled = true,
    selected_model_ids = '[]'::jsonb
WHERE paper_trading_enabled IS NULL 
   OR selected_model_ids IS NULL;

COMMIT;

-- Verification
\echo '✅ Phase 2 Migration Complete'
\echo ''
\echo 'Verification:'
SELECT COUNT(*) as total_models, COUNT(stock_symbol) as models_with_stock FROM models;
SELECT COUNT(*) as total_users, COUNT(*) FILTER (WHERE paper_trading_enabled) as users_with_paper_trading FROM users;
SELECT COUNT(*) as signal_logs_count FROM signal_logs;
