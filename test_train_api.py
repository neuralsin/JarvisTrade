"""
Test training API directly to see exact response
"""
import requests
import json

API_URL = "http://localhost:8000"

# Login
r = requests.post(f"{API_URL}/api/v1/auth/login", 
                 data={"username": "test12@example.com", "password": "testpass123"})
token = r.json()["access_token"]

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Try to train
print("Sending training request...")
r = requests.post(
    f"{API_URL}/api/v1/models/train",
    headers=headers,
    json={
        "model_name": "test_model",
        "model_type": "xgboost",
        "instrument_filter": "RELIANCE"
    }
)

print(f"\nStatus Code: {r.status_code}")
print(f"\nResponse Headers: {dict(r.headers)}")
print(f"\nResponse Body:")
print(json.dumps(r.json(), indent=2))

if r.status_code == 200:
    data = r.json()
    print(f"\n✓ Has 'task_id': {'task_id' in data}")
    print(f"✓ Has 'error': {'error' in data}")
    print(f"✓ task_id value: {data.get('task_id')}")
