FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to avoid issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in stages (no timeout for slow internet)
# Stage 1: Core framework dependencies
RUN pip install --no-cache-dir --retries 5 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0

# Stage 2: Database and caching
RUN pip install --no-cache-dir --retries 5 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    alembic==1.13.0 \
    redis==4.6.0 \
    celery[redis]==5.3.4

# Stage 3: Core data science
RUN pip install --no-cache-dir --retries 5 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    xgboost==2.0.2 \
    shap==0.43.0

# Stage 4: Deep learning (large downloads)
# Install TensorFlow
RUN pip install --no-cache-dir --retries 5 \
    tensorflow==2.15.0

# Install PyTorch separately (very large)
RUN pip install --no-cache-dir --retries 5 \
    torch==2.1.1 \
    torchvision==0.16.1

# Install transformers
RUN pip install --no-cache-dir --retries 5 \
    transformers==4.36.0

# Stage 5: Remaining dependencies
RUN pip install --no-cache-dir --retries 5 \
    yfinance==0.2.32 \
    kiteconnect==4.2.0 \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    cryptography==41.0.7 \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    httpx==0.25.2 \
    prometheus-client==0.19.0 \
    aiosmtplib==3.0.1 \
    newsapi-python==0.2.7 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
