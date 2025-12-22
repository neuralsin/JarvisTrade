-- Spec 1: DB schema (exact SQL create statements)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with encrypted Kite credentials
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  email text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  kite_api_key_encrypted text,
  kite_api_secret_encrypted text,
  kite_access_token_encrypted text,
  kite_request_token text,
  auto_execute boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Instruments (stocks/symbols)
CREATE TABLE IF NOT EXISTS instruments (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  symbol text UNIQUE NOT NULL,
  name text,
  exchange text,
  instrument_type text,
  created_at timestamptz DEFAULT now()
);

-- Historical candle data
CREATE TABLE IF NOT EXISTS historical_candles (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  instrument_id uuid REFERENCES instruments(id) ON DELETE CASCADE,
  timeframe text NOT NULL,
  ts_utc timestamptz NOT NULL,
  open numeric,
  high numeric,
  low numeric,
  close numeric,
  volume numeric,
  created_at timestamptz DEFAULT now(),
  UNIQUE(instrument_id, timeframe, ts_utc)
);

-- Computed features for ML
CREATE TABLE IF NOT EXISTS features (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  instrument_id uuid REFERENCES instruments(id) ON DELETE CASCADE,
  ts_utc timestamptz NOT NULL,
  feature_json jsonb,
  target integer,
  created_at timestamptz DEFAULT now(),
  UNIQUE(instrument_id, ts_utc)
);

-- Trained models registry
CREATE TABLE IF NOT EXISTS models (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  name text NOT NULL,
  model_type text DEFAULT 'xgboost',
  model_path text,
  trained_at timestamptz DEFAULT now(),
  metrics_json jsonb,
  is_active boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Trades (paper and live)
CREATE TABLE IF NOT EXISTS trades (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid REFERENCES users(id) ON DELETE CASCADE,
  instrument_id uuid REFERENCES instruments(id) ON DELETE CASCADE,
  mode text CHECK(mode IN ('paper','live')) NOT NULL,
  entry_ts timestamptz,
  entry_price numeric,
  exit_ts timestamptz,
  exit_price numeric,
  qty integer,
  stop_price numeric,
  target_price numeric,
  pnl numeric,
  reason text,
  probability numeric,
  model_id uuid REFERENCES models(id),
  status text DEFAULT 'open',
  kite_order_id text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Trade audit logs
CREATE TABLE IF NOT EXISTS trade_logs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  trade_id uuid REFERENCES trades(id) ON DELETE CASCADE,
  log_text text,
  log_level text DEFAULT 'INFO',
  ts timestamptz DEFAULT now()
);

-- System settings and state
CREATE TABLE IF NOT EXISTS system_state (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  key text UNIQUE NOT NULL,
  value text,
  updated_at timestamptz DEFAULT now()
);

-- Equity snapshots for performance tracking
CREATE TABLE IF NOT EXISTS equity_snapshots (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid REFERENCES users(id) ON DELETE CASCADE,
  mode text CHECK(mode IN ('paper','live')),
  ts_utc timestamptz NOT NULL,
  equity_value numeric NOT NULL,
  daily_pnl numeric,
  created_at timestamptz DEFAULT now(),
  UNIQUE(user_id, mode, ts_utc)
);

-- News sentiment data
CREATE TABLE IF NOT EXISTS news_sentiment (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  instrument_id uuid REFERENCES instruments(id) ON DELETE CASCADE,
  ts_utc timestamptz NOT NULL,
  sentiment_1d numeric,
  sentiment_3d numeric,
  sentiment_7d numeric,
  news_count integer,
  source text DEFAULT 'newsapi',
  created_at timestamptz DEFAULT now(),
  UNIQUE(instrument_id, ts_utc, source)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_candles_instrument_time ON historical_candles(instrument_id, timeframe, ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_features_instrument_time ON features(instrument_id, ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_trades_user_mode ON trades(user_id, mode, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_user ON equity_snapshots(user_id, mode, ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_instrument_time ON news_sentiment(instrument_id, ts_utc DESC);

-- Insert default system state
INSERT INTO system_state (key, value) VALUES 
  ('kill_switch', 'false'),
  ('kws_status', 'DOWN'),
  ('market_status', 'UNKNOWN')
ON CONFLICT (key) DO NOTHING;
