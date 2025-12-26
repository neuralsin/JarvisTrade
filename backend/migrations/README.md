# Database Migrations

This directory contains database migration scripts for JarvisTrade.

## Phase 2: Multi-Model Architecture

**Files:**
- `phase2_schema.py` - Python migration script (recommended)
- `phase2_schema.sql` - Raw SQL migration (alternative)

### Python Migration (Recommended)

```bash
# Apply migration
python migrations/phase2_schema.py up

# Rollback (includes safety confirmation)
python migrations/phase2_schema.py down
```

### SQL Migration (Alternative)

If you prefer raw SQL:

```bash
# Via psql
psql -U postgres -d jarvistrade -f migrations/phase2_schema.sql

# Or via Docker
docker exec -i jarvistrade_db psql -U postgres -d jarvistrade < migrations/phase2_schema.sql
```

## What Phase 2 Adds

1. **models.stock_symbol** (TEXT, nullable)
   - Tracks which stock a model is trained on
   - Enables per-stock model architecture
   
2. **users.selected_model_ids** (JSONB, default=[])
   - Array of model UUIDs active for the user
   - Enables multi-model parallel execution
   
3. **users.paper_trading_enabled** (BOOLEAN, default=true)
   - Master switch for paper trading
   
4. **signal_logs** table (NEW)
   - Tracks all EXECUTE and REJECT signals
   - Includes probability, reason, filters, features
   - Indexed for fast queries
   - Enables real-time monitoring

## Verification

After running migration:

```sql
-- Check new columns exist
\d models
\d users

-- Check signal_logs table
\d signal_logs

-- Verify indexes
\di signal_logs*
```

## Rollback

The Python script includes a rollback function that:
- Drops signal_logs table
- Removes added columns from users and models
- Requires explicit confirmation to prevent accidents

**⚠️ Warning**: Rollback will delete all signal logs permanently.
