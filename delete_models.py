"""
Script to delete models safely
"""
import requests

API_URL = "http://localhost:8000"

# Login
print("Logging in...")
r = requests.post(f"{API_URL}/api/v1/auth/login", 
                 data={"username": "test12@example.com", "password": "testpass123"})
token = r.json()["access_token"]
print(f"✓ Logged in\n")

headers = {"Authorization": f"Bearer {token}"}

# Get all models
print("Fetching models...")
r = requests.get(f"{API_URL}/api/v1/models", headers=headers)
data = r.json()
models = data.get('models', [])

print(f"\nFound {len(models)} models:\n")
print(f"{'ID':<40} {'Name':<25} {'Active':<8}")
print("="*75)

for m in models:
    model_id = m['id']
    name = m['name']
    is_active = "✓ YES" if m['is_active'] else "  no"
    print(f"{model_id:<40} {name:<25} {is_active}")

# Delete non-active models
print("\n" + "="*75)
print("Deleting inactive models...\n")

deleted_count = 0
for m in models:
    if not m['is_active']:
        model_id = m['id']
        name = m['name']
        
        print(f"Deleting: {name} ({model_id[:8]}...)...", end=" ")
        r = requests.delete(f"{API_URL}/api/v1/models/{model_id}", headers=headers)
        
        if r.status_code == 200:
            print("✓ Deleted")
            deleted_count += 1
        else:
            print(f"✗ Failed: {r.json().get('detail', 'Unknown error')}")
    else:
        print(f"Skipping active model: {m['name']}")

print(f"\n✓ Deleted {deleted_count} inactive model(s)")
print("\nTo delete active models:")
print("1. First activate a different model")
print("2. Then delete the previously active one")
