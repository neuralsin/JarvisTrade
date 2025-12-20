# JarvisTrade

A production-ready ML-driven trading platform for NSE/BSE stocks with paper trading simulation, model training interface, and live trading via Zerodha Kite API.

## ✨ Features

### Core Trading Platform
- 📊 **Paper Trading** - Simulated trading with real market data
- ⚡ **Live Trading** - Automated trading via Zerodha Kite API
- 📈 **Real-time Dashboard** - Equity curves, P&L tracking, performance metrics
- 🛡️ **Safety Features** - Kill switch, circuit breakers, risk management
- 📧 **Notifications** - Email alerts for trade execution
- ⚙️ **Parameter Management** - Editable trading parameters for paper/live modes

### ML & AI Capabilities
- 🤖 **XGBoost Models** - Gradient boosting with SHAP explainability
- 🧠 **LSTM Support** - Time series prediction with sequence modeling
- 📰 **News Sentiment** - FinBERT-powered sentiment analysis (optional)
- 📊 **Model Metrics** - Comprehensive performance tracking

### Portfolio Management
- 💼 **Bucket Tracking** - Organize portfolio by sector, market cap, strategy
- 🎯 **Allocation Monitoring** - Target vs actual with visual indicators
- 🔄 **Rebalance Suggestions** - Automated recommendations

### UI/UX
- 🎨 **Modern Dark Theme** - Glassmorphism with smooth animations
- 📱 **Responsive Design** - Works on desktop and mobile
- ⚡ **Smooth Scrolling** - Enhanced animations and micro-interactions
- 📊 **Interactive Charts** - Recharts visualizations

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) Zerodha Kite API credentials for live trading

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd JarvisTrade
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Edit `.env` and configure:
   - Database credentials
   - Email SMTP settings (for notifications)
   - Zerodha Kite API credentials (for live trading)
   - Trading parameters (risk, capital, etc.)

4. Start all services:
```bash
docker-compose up --build
```

This will start:
- PostgreSQL database (port 5432)
- Redis (port 6379)
- Backend API (port 8000)
- Celery workers
- Frontend (port 3000)

5. Access the application:
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Default login: `admin@jarvistrade.com` / `admin123`

### Initial Data Seeding

Populate the database with historical data:

```bash
docker-compose exec backend python -m app.seed --years 2
```

Options:
- `--years`: Number of years of historical data (default: 2)
- `--email`: Admin email (default: admin@jarvistrade.com)
- `--password`: Admin password (default: admin123)

This script will:
1. Create a superuser
2. Seed instruments (top NIFTY stocks)
3. Fetch historical data from yfinance
4. Compute features and labels

## 📖 Usage

### 1. Train a Model

Navigate to **Models** page and click "Train New Model":
- Enter model name (e.g., `xgb_v1`)
- Optionally filter by instrument
- Training progress will update in real-time

### 2. Activate Model

Once trained:
- Select the model from the list
- Click "Activate Model"
- The system will use this model for trading signals

### 3. Paper Trading

- Navigate to **Paper Trading** page
- View simulated trades and P&L
- No real money at risk

### 4. Live Trading

⚠️ **Warning**: Live trading executes real trades with real money.

1. Navigate to **Live Trading** page
2. Configure Kite credentials
3. Enable "Auto Execute" (requires password confirmation)
4. System will automatically execute trades based on model signals

**Safety Features**:
- **Kill Switch**: Emergency stop for all trading
- **Daily Loss Limit**: Auto-disable if daily loss exceeds threshold
- **Max Trades/Day**: Limit number of trades per day
- **Circuit Breakers**: Auto-disable on high VIX or market crashes

## 🏗️ Architecture

```
JarvisTrade/
├── backend/                 # FastAPI + Python
│   ├── app/
│   │   ├── routers/        # API endpoints
│   │   ├── db/             # Database models
│   │   ├── ml/             # Feature engineering, training
│   │   ├── trading/        # Decision engine, paper sim, Kite client
│   │   ├── tasks/          # Celery background tasks
│   │   └── utils/          # Helpers (retry, crypto, mailer)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # React + Vite
│   ├── src/
│   │   ├── pages/         # Dashboard, Trading, Models
│   │   ├── components/    # Reusable UI components
│   │   └── styles/        # Dark theme CSS
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── .env.example
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Trading Parameters
ACCOUNT_CAPITAL=100000.0      # Starting capital
RISK_PER_TRADE=0.01           # Risk 1% per trade
MAX_DAILY_LOSS=0.02           # Max 2% daily loss
STOP_MULTIPLIER=1.5           # Stop loss multiplier (1.5x ATR)
TARGET_MULTIPLIER=2.5         # Target multiplier (2.5x ATR)
PROB_MIN=0.65                 # Minimum probability threshold
MAX_TRADES_PER_DAY=3          # Max trades per day

# Zerodha Kite
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret

# Email Notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
```

## 📊 ML Pipeline

### Feature Engineering

Computed features include:
- Returns (1, 5 period)
- EMAs (20, 50, 200)
- RSI-14 and RSI slope
- ATR-14 and ATR percentage
- Volume ratio
- Nifty trend indicator
- VIX

### Labeling

Target label = 1 if:
- Future high reaches 1.2% gain before
- Future low hits 0.6% stop loss

### Model Training

- Algorithm: XGBoost Classifier
- Train/Val/Test split: 2015-2022 / 2023 / 2024+
- Early stopping on validation set
- SHAP feature importance
- Metrics: AUC-ROC, Precision@K, F1, Sharpe

### Retraining

- Weekly scheduled retraining (Monday 2 AM UTC)
- Drift detection using KS-test
- Auto-retrain if >3 features show drift

## 🔐 Security

- JWT-based authentication
- Encrypted Kite API credentials (AES-256)
- Password confirmation for critical actions
- Two-factor enable for live trading

## 🧪 Testing

Run tests:
```bash
docker-compose exec backend pytest
```

Test coverage:
```bash
docker-compose exec backend pytest --cov=app
```

## 📈 Monitoring

- Prometheus metrics at `/metrics`
- Structured JSON logging
- Trade execution email notifications

Metrics exposed:
- `jarvis_trades_executed_total{mode}`
- `jarvis_trade_pnl_sum`
- `jarvis_daily_loss{user_id}`
- `jarvis_model_train_duration_seconds`
- `jarvis_kite_ws_status`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

**Trading involves substantial risk**. This software is provided "as is" without warranty. The authors are not responsible for any financial losses incurred through the use of this platform. Always do your own research and consider consulting with a financial advisor.

## 🙏 Acknowledgments

- NSE/BSE for market data
- Zerodha Kite for trading APIs
- yfinance for historical data
- XGBoost for ML framework

---

**Happy Trading! 🚀**
