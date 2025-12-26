"""
Test HCLTECH fetch directly with yfinance to see exact error
"""
import yfinance as yf
from datetime import datetime, timedelta

# Your working URL format
url = "https://query1.finance.yahoo.com/v8/finance/chart/HCLTECH.NS?interval=15m&range=60d"
print(f"Working URL: {url}\n")

# Try with yfinance using different approaches
symbol = "HCLTECH.NS"
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print("="*80)
print("TEST 1: Using yf.download with dates")
print("="*80)
try:
    df1 = yf.download(
        symbol,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='15m',
        progress=False,
        auto_adjust=False
    )
    print(f"✓ SUCCESS: Got {len(df1)} rows")
    print(df1.head())
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*80)
print("TEST 2: Using yf.download with period=60d")
print("="*80)
try:
    df2 = yf.download(
        symbol,
        period='60d',
        interval='15m',
        progress=False,
        auto_adjust=False
    )
    print(f"✓ SUCCESS: Got {len(df2)} rows")
    print(df2.head())
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*80)
print("TEST 3: Using Ticker.history with period")
print("="*80)
try:
    ticker = yf.Ticker(symbol)
    df3 = ticker.history(period='60d', interval='15m')
    print(f"✓ SUCCESS: Got {len(df3)} rows")
    print(df3.head())
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*80)
print("TEST 4: Check ticker info")
print("="*80)
try:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    print(f"✓ Ticker name: {info.get('shortName', 'N/A')}")
    print(f"✓ Exchange: {info.get('exchange', 'N/A')}")
except Exception as e:
    print(f"✗ FAILED: {e}")
