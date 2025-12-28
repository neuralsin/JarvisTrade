-- Phase 3: Training Datasets Table
-- Stores metadata about uploaded CSV files for model training

CREATE TABLE IF NOT EXISTS training_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    row_count INTEGER,
    date_start TIMESTAMP WITH TIME ZONE,
    date_end TIMESTAMP WITH TIME ZONE,
    columns JSONB,
    file_size_bytes INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    interval VARCHAR(10),
    notes TEXT
);

-- Index for fast lookups by symbol
CREATE INDEX IF NOT EXISTS idx_training_datasets_symbol ON training_datasets(symbol);

-- Index for ordering by upload date
CREATE INDEX IF NOT EXISTS idx_training_datasets_uploaded_at ON training_datasets(uploaded_at DESC);

-- Comment on table
COMMENT ON TABLE training_datasets IS 'Stores metadata about CSV files uploaded for model training';
