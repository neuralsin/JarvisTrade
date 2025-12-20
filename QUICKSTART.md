# 🚀 Quick Start Guide - JarvisTrade

Get JarvisTrade up and running in 5 minutes!

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- 8GB+ RAM available
- 10GB+ disk space

## Step 1: Configure Environment

The `.env` file has been created for you. Edit it to configure:

```bash
# Open .env file in your editor
notepad .env  # Windows
# or
code .env     # VS Code
```

**Minimum required changes:**

1. **Email notifications** (optional but recommended):
   ```
   SMTP_USER=your-email@gmail.com
   SMTP_PASS=your-app-password
   ```
   
2. **For live trading** (optional):
   ```
   KITE_API_KEY=your_kite_api_key
   KITE_API_SECRET=your_kite_api_secret
   ```

> **Tip**: Everything else has sensible defaults. You can start without changing anything!

## Step 2: Start the Platform

```bash
# From the JarvisTrade directory
docker-compose up --build
```

This will:
- Pull and build all Docker images (~5-10 minutes first time)
- Start PostgreSQL, Redis, Backend, Frontend, and Celery workers
- Backend will be available at http://localhost:8000
- Frontend will be available at http://localhost:3000

**Wait for**: "Application startup complete" in the logs

## Step 3: Seed Initial Data

In a **new terminal**:

```bash
# Seed with 2 years of data for top 10 NIFTY stocks
docker-compose exec backend python -m app.seed --years 2
```

This will:
- Create admin user (email: `admin@jarvistrade.com`, password: `admin123`)
- Fetch historical data from yfinance
- Compute features and labels
- Takes ~10-30 minutes depending on internet speed

> **Note**: You can use the app while seeding is in progress!

## Step 4: Access the App

1. Open browser: http://localhost:3000
2. Login with:
   - Email: `admin@jarvistrade.com`
   - Password: `admin123`
3. You're in! 🎉

## Step 5: Train Your First Model

1. Navigate to **Models** page (robot icon 🤖)
2. Click "➕ Train New Model"
3. Enter name: `my_first_model`
4. Click "Start Training"
5. Training takes ~10-30 minutes
6. Once complete, click "Activate Model"

## Step 6: Watch It Trade!

- **Paper Trading** is active by default
- Go to **Dashboard** to see:
  - Equity curve (how your capital changes)
  - Open trades
  - P&L statistics
- Trades execute automatically every 15 minutes based on model signals

---

## 🎯 What's Next?

### Explore the Platform

- **Dashboard** - View performance metrics and open trades
- **Paper Trading** - Mirror of live market without risk
- **Live Trading** - Enable after configuring Kite credentials
- **Models** - View metrics, SHAP feature importance
- **Trades** - Complete history with detailed logs

### Customize Trading Parameters

Edit `.env` to adjust:

```bash
ACCOUNT_CAPITAL=100000.0      # Starting capital
RISK_PER_TRADE=0.01           # Risk 1% per trade
MAX_DAILY_LOSS=0.02           # Stop trading if 2% daily loss
STOP_MULTIPLIER=1.5           # Stop loss distance (1.5x ATR)
TARGET_MULTIPLIER=2.5         # Take profit distance (2.5x ATR)
PROB_MIN=0.65                 # Minimum probability to trade
MAX_TRADES_PER_DAY=3          # Max 3 trades per day
```

After changes, restart:
```bash
docker-compose restart backend celery_worker
```

---

## 📊 API Documentation

Interactive API docs: http://localhost:8000/docs

Explore all endpoints and try them out directly!

---

## 🛑 Stopping the Platform

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes ALL data!)
docker-compose down -v
```

---

## 🐛 Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs -f backend

# Restart specific service
docker-compose restart backend
```

### Database errors

```bash
# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d db
# Wait 10 seconds for DB to be ready
docker-compose up backend
```

### "Module not found" errors

```bash
# Rebuild containers
docker-compose up --build
```

### Port already in use

Edit `docker-compose.yml` and change port mappings:
```yaml
ports:
  - "8001:8000"  # Changed from 8000:8000
```

---

## 📚 Need Help?

1. Check the [README.md](README.md) for detailed documentation
2. View [walkthrough.md](.gemini/antigravity/brain/.../walkthrough.md) for architecture details
3. Check logs: `docker-compose logs -f`
4. Open an issue on GitHub

---

## ⚠️ Important Notes

- **Paper trading is safe** - no real money, just simulation
- **Live trading requires** Kite API credentials and careful setup
- **First model training** may take 30+ minutes
- **Data seeding** is a one-time process
- **Always set stop losses** when trading real money

---

**Happy Trading! 🚀📈**

*Built for traders who believe in data-driven decisions*
