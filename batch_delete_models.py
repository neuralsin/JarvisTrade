"""
Batch delete all inactive models
"""
import requests
import sys

API_URL = "http://localhost:8000"

# Login
print("Logging in...")
r = requests.post(f"{API_URL}/api/v1/auth/login", 
                 data={"username": "test12@example.com", "password": "testpass123"})

if r.status_code != 200:
    print(f"✗ Login failed: {r.status_code}")
    sys.exit(1)

token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("✓ Logged in\n")

# Get models
print("Fetching models...")
r = requests.get(f"{API_URL}/api/v1/models", headers=headers)
if r.status_code != 200:
    print(f"✗ Failed to get models: {r.status_code}")
    sys.exit(1)

data = r.json()
models = data.get('models', [])
print(f"Found {len(models)} models\n")

# Show all models
print("Current Models:")
print("="*80)
for m in models:
    status = "✓ ACTIVE" if m['is_active'] else "  inactive"
    print(f"{status} | {m['name'][:40]:<40} | {m['id'][:8]}...")
print("="*80 + "\n")

# Delete inactive models
inactive = [m for m in models if not m['is_active']]
print(f"Found {len(inactive)} inactive model(s) to delete\n")

if not inactive:
    print("No inactive models to delete!")
    print("\nTo delete an active model:")
    print("1. First, activate a different model")
    print("2. Then the previously active model becomes inactive")
    print("3. Run this script again to delete it")
    sys.exit(0)

deleted = 0
failed = 0

for m in inactive:
    model_id = m['id']
    name = m['name']
    
    print(f"Deleting: {name[:40]:<40} ", end="")
    
    r = requests.delete(f"{API_URL}/api/v1/models/{model_id}", headers=headers)
    
    if r.status_code == 200:
        print("✓ Deleted")
        deleted += 1
    else:
        error = r.json().get('detail', 'Unknown error')
        print(f"✗ Failed: {error}")
        failed += 1

print("\n" + "="*80)
print(f"Summary: Deleted {deleted}, Failed {failed}")
print("="*80)
