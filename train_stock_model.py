"""
Train a stock-specific model to verify Phase 3 implementation
"""
import requests
import time

API_URL = "http://localhost:8000"

# Login first
print("Logging in...")
r = requests.post(f"{API_URL}/api/v1/auth/login", 
                 data={"username": "test12@example.com", "password": "testpass123"})
token = r.json()["access_token"]
print(f"✓ Logged in, token: {token[:30]}...\n")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Train a stock-specific model
print("Training stock-specific model for RELIANCE...")
print("This will take a few minutes...\n")

r = requests.post(
    f"{API_URL}/api/v1/models/train",
    headers=headers,
    json={
        "model_name": "reliance_xgb_v1",
        "model_type": "xgboost",
        "instrument_filter": "RELIANCE",  # Phase 3: Stock-specific filter
        "start_date": "2024-01-01",
        "end_date": "2024-12-26"
    }
)

if r.status_code == 200:
    data = r.json()
    task_id = data.get("task_id")
    print(f"✓ Training task queued: {task_id}\n")
    
    # Poll task status
    print("Monitoring training progress...")
    for i in range(60):  # Check for up to 10 minutes
        time.sleep(10)
        
        r = requests.get(f"{API_URL}/api/v1/models/task/{task_id}", headers=headers)
        status_data = r.json()
        
        state = status_data.get("status")
        progress = status_data.get("progress", 0)
        message = status_data.get("message", "")
        
        print(f"[{i*10}s] Status: {state} | Progress: {progress}% | {message}")
        
        if state == "SUCCESS":
            print("\n✓ Training completed successfully!")
            print(f"\nResult: {status_data.get('result')}")
            break
        elif state == "FAILURE":
            print(f"\n✗ Training failed: {status_data.get('error')}")
            break
else:
    print(f"✗ Failed to queue training: {r.status_code} - {r.text}")

print("\n" + "="*80)
print("Now run: python final_test.py")
print("You should see 'Stock-specific: 1' in TEST 12")
