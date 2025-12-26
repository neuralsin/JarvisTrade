"""
Test why HCLTECH fails on Yahoo Finance
"""
import yfinance as yf
from datetime import datetime, timedelta

symbols_to_test = [
    "HCLTECH.NS",
    "HCLTECH.BO", 
    "HCL-TECH.NS",
    "HCLTECH"
]

end = datetime.now()
start = end - timedelta(days=60)

print("Testing different ticker formats for HCLTECH:\n")

for symbol in symbols_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Try with period
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='60d', interval='15m')
        
        if not hist.empty:
            print(f"✓ SUCCESS with period='60d'")
            print(f"  Rows: {len(hist)}")
            print(f"  Date range: {hist.index[0]} to {hist.index[-1]}")
            print(f"  Columns: {list(hist.columns)}")
            print("\nFirst 3 rows:")
            print(hist.head(3))
            break
        else:
            print(f"✗ FAILED: Got empty dataframe")
            
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")

print("\n" + "="*60)
print("Testing with download() instead of Ticker.history()")
print("="*60)

try:
    df = yf.download("HCLTECH.NS", period='60d', interval='15m', progress=False)
    if not df.empty:
        print(f"✓ download() SUCCESS")
        print(f"  Rows: {len(df)}")
    else:
        print("✗ download() returned empty")
except Exception as e:
    print(f"✗ download() ERROR: {e}")
