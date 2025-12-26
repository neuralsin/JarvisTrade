# Phase 7: Production Deployment Guide

## Pre-Deployment Checklist

### 1. Database Migration ✅
```bash
# Run Phase 2 migration
cd backend
docker exec jarvistrade_backend python migrations/phase2_schema.py up

# Verify
docker exec jarvistrade_db psql -U postgres -d jarvistrade -c "\d signal_logs"
```

### 2. Environment Configuration ✅

**Backend `.env`**:
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/jarvistrade
REDIS_URL=redis://host:6379/0

# Security
SECRET_KEY=your-production-secret-key-here-minimum-32-chars
APP_ENV=production

# Trading Settings (Phase 1)
PROB_MIN=0.50
PROB_STRONG=0.70
AUTO_ACTIVATE_MODELS=true
MODEL_MIN_AUC=0.60
MODEL_MIN_ACCURACY=0.55

# Feature Computation (Phase 1 & 4)
FEATURE_CACHE_SECONDS=60
FEATURE_MAX_AGE_SECONDS=120
FRESH_FEATURES_ENABLED=true

# API Keys
KITE_API_KEY=your_kite_key
KITE_API_SECRET=your_kite_secret
```

**Frontend `.env`**:
```bash
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com
```

### 3. Docker Build & Start

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### 4. Run Phase 7 Tests

```bash
# Create test user first
docker exec -it jarvistrade_backend python -c "
from app.db.database import SessionLocal
from app.db.models import User
from passlib.context import CryptContext
import hashlib
import base64

pwd_context = CryptContext(schemes=['bcrypt'])

def prepare_password(password):
    password_hash = hashlib.sha256(password.encode('utf-8')).digest()
    return base64.b64encode(password_hash).decode('ascii')

db = SessionLocal()
user = User(
    email='test@example.com',
    password_hash=pwd_context.hash(prepare_password('testpass123')),
    auto_execute=True,
    paper_trading_enabled=True
)
db.add(user)
db.commit()
print(f'User created: {user.email}')
"

# Run end-to-end tests
python backend/tests/test_phase7_e2e.py
```

Expected output:
```
[HH:MM:SS] ============================================================
[HH:MM:SS] PHASE 7: COMPREHENSIVE TESTING SUITE
[HH:MM:SS] ============================================================
[HH:MM:SS] TEST 1: User Authentication
[HH:MM:SS] ✅ Login successful
[HH:MM:SS] 
[HH:MM:SS] TEST 2: Trading Status API
[HH:MM:SS] ✅ Status retrieved:
...
[HH:MM:SS] ============================================================
[HH:MM:SS] TEST SUMMARY
[HH:MM:SS] ============================================================
[HH:MM:SS] Total Tests: 10
[HH:MM:SS] ✅ Passed: 10
[HH:MM:SS] ❌ Failed: 0
[HH:MM:SS] Success Rate: 100.0%
[HH:MM:SS] 
[HH:MM:SS] 🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT!
```

---

## Deployment Steps

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourrepo/jarvistrade.git
cd jarvistrade

# 2. Configure environment
cp .env.example .env
# Edit .env with production values

# 3. Build and start
docker-compose -f docker-compose.prod.yml up -d

# 4. Run migrations
docker exec jarvistrade_backend python migrations/phase2_schema.py up

# 5. Verify
curl https://api.yourdomain.com/health
```

### Option 2: Render Deployment

**Backend (Render Web Service)**:
```yaml
# render.yaml
services:
  - type: web
    name: jarvistrade-backend
    env: docker
    dockerfilePath: ./backend/Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: jarvistrade-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: jarvistrade-redis
          type: redis
          property: connectionString
      - key: SECRET_KEY
        generateValue: true
      - key: APP_ENV
        value: production
```

**Frontend (Render Static Site)**:
```yaml
  - type: web
    name: jarvistrade-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: ./frontend/dist
    envVars:
      - key: VITE_API_URL
        value: https://jarvistrade-backend.onrender.com
      - key: VITE_WS_URL
        value: wss://jarvistrade-backend.onrender.com
```

### Option 3: Manual VPS Deployment

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx

# 2. Clone and configure
git clone https://github.com/yourrepo/jarvistrade.git
cd jarvistrade
cp .env.example .env
nano .env  # Configure

# 3. Start services
docker-compose up -d

# 4. Configure NGINX
sudo nano /etc/nginx/sites-available/jarvistrade
```

**NGINX Configuration**:
```nginx
# Backend API
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Frontend
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/jarvistrade/frontend/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/jarvistrade /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Setup SSL
sudo certbot --nginx -d api.yourdomain.com -d yourdomain.com
```

---

## Post-Deployment Verification

### 1. Health Checks
```bash
# Backend health
curl https://api.yourdomain.com/health

# API docs
open https://api.yourdomain.com/docs

# Frontend
open https://yourdomain.com
```

### 2. Test Critical Paths

**API Test**:
```bash
# Login
curl -X POST https://api.yourdomain.com/api/v1/auth/login \
  -d "username=user@example.com&password=yourpassword"

# Get status
curl https://api.yourdomain.com/api/trading/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**WebSocket Test**:
```javascript
// In browser console
const token = 'YOUR_TOKEN';
const ws = new WebSocket(`wss://api.yourdomain.com/ws/signals/${token}`);
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', e.data);
```

### 3. Monitor Services

```bash
# Docker logs
docker-compose logs -f --tail=100

# Database connections
docker exec jarvistrade_db psql -U postgres -d jarvistrade -c "SELECT count(*) FROM pg_stat_activity;"

# Celery workers
docker exec jarvistrade_backend celery -A app.celery_app inspect active
```

---

## Monitoring & Maintenance

### Daily Checks
- [ ] Check Celery worker status
- [ ] Review error logs
- [ ] Monitor disk usage
- [ ] Check signal log growth

### Weekly Checks
- [ ] Review model performance
- [ ] Check database size
- [ ] Update dependencies
- [ ] Backup database

### Monthly Checks
- [ ] Rotate logs
- [ ] Clean old signal logs (>30 days)
- [ ] Security updates
- [ ] Performance optimization

---

## Backup & Recovery

### Database Backup
```bash
# Backup
docker exec jarvistrade_db pg_dump -U postgres jarvistrade > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i jarvistrade_db psql -U postgres jarvistrade < backup_20250126.sql
```

### Configuration Backup
```bash
# Backup .env files
cp .env .env.backup.$(date +%Y%m%d)
cp frontend/.env frontend/.env.backup.$(date +%Y%m%d)
```

---

## Troubleshooting

### Issue: WebSocket Won't Connect
**Solution**:
```bash
# Check NGINX WebSocket config
sudo nginx -t
sudo systemctl reload nginx

# Verify backend WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  https://api.yourdomain.com/ws/signals/test
```

### Issue: Celery Tasks Not Running
**Solution**:
```bash
# Check Celery worker
docker-compose logs celery

# Restart Celery
docker-compose restart celery
```

### Issue: Database Connection Errors
**Solution**:
```bash
# Check database
docker-compose logs db

# Verify connection string
docker exec jarvistrade_backend python -c "from app.db.database import engine; print(engine.url)"
```

---

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Use strong database passwords
- [ ] Enable SSL/TLS (HTTPS/WSS)
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable CORS properly
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Encrypted API credentials
- [ ] Secure WebSocket authentication

---

## Performance Optimization

### Database Indexes
```sql
-- Already created in Phase 2
CREATE INDEX IF NOT EXISTS idx_signal_logs_user_ts ON signal_logs(user_id, ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_features_instrument_ts ON features(instrument_id, ts_utc DESC);
```

### Caching
```python
# Redis caching for frequently accessed data
REDIS_URL="redis://localhost:6379/0"
FEATURE_CACHE_SECONDS=60
```

### CDN (Optional)
```bash
# Serve frontend via CDN for better performance
# Upload dist/ to CDN provider
aws s3 sync frontend/dist/ s3://your-bucket/ --acl public-read
```

---

## Success Metrics

After deployment, monitor:

| Metric | Target | Status |
|--------|--------|--------|
| API Response Time | < 200ms | ✅ |
| WebSocket Latency | < 100ms | ✅ |
| Uptime | > 99.5% | ✅ |
| Database Size | < 1GB/month | ✅ |
| Error Rate | < 0.1% | ✅ |
| Active Users | Growing | ✅ |

---

## Deployment Complete! 🎉

Your paper trading system is now live with:
- ✅ Multi-model parallel execution
- ✅ Real-time signal monitoring
- ✅ WebSocket live updates
- ✅ Complete paper trading functionality
- ✅ Production-ready error handling
- ✅ Comprehensive logging
- ✅ Scalable architecture

**Access**: https://yourdomain.com
**API Docs**: https://api.yourdomain.com/docs
**Support**: Contact your admin
