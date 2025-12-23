"""
Configuration for Yahoo Finance rate limiting.
Adjust these values if you're experiencing persistent rate limit issues.
"""

# Minimum delay between Yahoo Finance API requests (seconds)
# Increase this if you're still getting 429 errors
YFINANCE_MIN_DELAY = 5.0

# Maximum retry attempts when rate limited
# More retries = longer wait times but higher chance of success
YFINANCE_MAX_RETRIES = 5

# Base wait time for exponential backoff (seconds)
# Formula: wait_time = BASE_WAIT * (2 ** attempt)
# With BASE_WAIT=10: 10s, 30s, 60s, 120s, 240s
YFINANCE_BASE_WAIT = 10

# Delay before accessing ticker.info or ticker.history after creating Ticker (seconds)
# Yahoo Finance treats these as separate requests
YFINANCE_REQUEST_DELAY = 1.0

# Inter-symbol delay when fetching multiple stocks (seconds)
# Additional delay between processing different symbols
YFINANCE_INTER_SYMBOL_DELAY = 1.0
