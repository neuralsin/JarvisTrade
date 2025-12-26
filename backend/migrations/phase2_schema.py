"""
Phase 2 Database Migration Script

This script adds Phase 2 schema changes:
1. Stock symbol to Model table
2. Selected models JSON to User table  
3. Paper trading enabled to User table
4. SignalLog table for real-time signal tracking

Usage:
    python migrations/phase2_schema.py up      # Apply migration
    python migrations/phase2_schema.py down    # Rollback migration
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MIGRATION_UP = """
-- Phase 2 Migration: Multi-Model Architecture
-- Author: Antigravity AI
-- Date: 2025-12-26

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

-- Create indexes for signal_logs
CREATE INDEX IF NOT EXISTS idx_signal_logs_user_ts 
ON signal_logs(user_id, ts_utc DESC);

CREATE INDEX IF NOT EXISTS idx_signal_logs_model 
ON signal_logs(model_id);

CREATE INDEX IF NOT EXISTS idx_signal_logs_instrument 
ON signal_logs(instrument_id);

CREATE INDEX IF NOT EXISTS idx_signal_logs_action 
ON signal_logs(action);

COMMENT ON TABLE signal_logs IS 'Phase 2: Tracks all trading signals (EXECUTE and REJECT) for real-time monitoring and historical analysis';

-- 5. Update existing users to have paper trading enabled (for existing data)
UPDATE users 
SET paper_trading_enabled = true,
    selected_model_ids = '[]'::jsonb
WHERE paper_trading_enabled IS NULL 
   OR selected_model_ids IS NULL;

COMMIT;

-- Verification queries
SELECT 'Phase 2 Migration Applied Successfully!' as status;
SELECT COUNT(*) as total_models, COUNT(stock_symbol) as models_with_stock FROM models;
SELECT COUNT(*) as total_users, COUNT(*) FILTER (WHERE paper_trading_enabled) as users_with_paper_trading FROM users;
"""

MIGRATION_DOWN = """
-- Phase 2 Rollback Migration
-- Removes Phase 2 schema changes

BEGIN;

-- Drop signal_logs table (cascades to indexes)
DROP TABLE IF EXISTS signal_logs CASCADE;

-- Remove columns from users
ALTER TABLE users DROP COLUMN IF EXISTS selected_model_ids;
ALTER TABLE users DROP COLUMN IF EXISTS paper_trading_enabled;

-- Remove column from models
ALTER TABLE models DROP COLUMN IF EXISTS stock_symbol;

COMMIT;

SELECT 'Phase 2 Migration Rolled Back' as status;
"""


def apply_migration():
    """Apply Phase 2 migration"""
    logger.info("=" * 80)
    logger.info("APPLYING PHASE 2 MIGRATION")
    logger.info("=" * 80)
    
    try:
        with engine.connect() as conn:
            # Execute migration
            conn.execute(text(MIGRATION_UP))
            conn.commit()
            
            logger.info("\n✅ Phase 2 migration applied successfully!")
            logger.info("\nChanges:")
            logger.info("  - Added models.stock_symbol")
            logger.info("  - Added users.selected_model_ids (JSONB)")
            logger.info("  - Added users.paper_trading_enabled")
            logger.info("  - Created signal_logs table with 4 indexes")
            
            # Verify
            result = conn.execute(text("SELECT COUNT(*) FROM signal_logs"))
            logger.info(f"\n✓ signal_logs table created (0 rows initially)")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise


def rollback_migration():
    """Rollback Phase 2 migration"""
    logger.info("=" * 80)
    logger.info("ROLLING BACK PHASE 2 MIGRATION")
    logger.info("=" * 80)
    
    confirmation = input("\n⚠️  This will delete all signal logs. Are you sure? (yes/no): ")
    if confirmation.lower() != 'yes':
        logger.info("Rollback cancelled")
        return
    
    try:
        with engine.connect() as conn:
            # Execute rollback
            conn.execute(text(MIGRATION_DOWN))
            conn.commit()
            
            logger.info("\n✅ Phase 2 migration rolled back successfully!")
            
    except Exception as e:
        logger.error(f"❌ Rollback failed: {str(e)}")
        raise


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python migrations/phase2_schema.py up      # Apply migration")
        print("  python migrations/phase2_schema.py down    # Rollback migration")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'up':
        apply_migration()
    elif command == 'down':
        rollback_migration()
    else:
        print(f"Unknown command: {command}")
        print("Use 'up' or 'down'")
        sys.exit(1)


if __name__ == "__main__":
    main()
