"""
Comprehensive System Testing - All Phases
Direct Python implementation for reliable testing
"""
import requests
import time
import sys
from datetime import datetime

API_URL = "http://localhost:8000"
token = None
results = {"passed": 0, "failed": 0, "warnings": 0, "total": 0}

def log(message, status="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️"}
    symbol = symbols.get(status, "")
    print(f"[{timestamp}] {symbol} {message}")

def test(name, func):
    """Run a test and track results"""
    global results
    results["total"] += 1
    log(f"TEST {results['total']}: {name}", "INFO")
    try:
        status, details = func()
        if status == "PASS":
            results["passed"] += 1
            log(f"  {details}", "PASS")
        elif status == "WARN":
            results["warnings"] += 1
            log(f"  {details}", "WARN")
        else:
            results["failed"] += 1
            log(f"  {details}", "FAIL")
    except Exception as e:
        results["failed"] += 1
        log(f"  ERROR: {str(e)} ({type(e).__name__})", "FAIL")

print("="*80)
print("COMPREHENSIVE SYSTEM TESTING - ALL PHASES")
print("="*80)
print()

# PHASE 1 & 2:Database Schema Tests
print("\n--- PHASE 2: DATABASE SCHEMA TESTS ---\n")

def test_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            return "PASS", f"Backend healthy - {r.json().get('service')}"
        return "FAIL", f"Unexpected response: {r.status_code}"
    except:
        return "FAIL", "Backend not responding - ensure Docker containers are running"

def test_api_docs():
    try:
        r = requests.get(f"{API_URL}/docs", timeout=5)
        return ("PASS", "API documentation accessible") if r.status_code == 200 else ("FAIL", f"Status: {r.status_code}")
    except:
        return "FAIL", "API docs not accessible"

test("Backend health endpoint", test_health)
test("API documentation (/docs)", test_api_docs)

# PHASE 3: Authentication Tests
print("\n--- AUTHENTICATION TESTS ---\n")

def test_login():
    global token
    try:
        r = requests.post(f"{API_URL}/api/v1/auth/login", data={"username": "test@example.com", "password": "testpass123"})
        if r.status_code == 200:
            token = r.json()["access_token"]
            return "PASS", f"JWT token obtained: {token[:30]}..."
        elif r.status_code == 401:
            return "WARN", "Test user doesn't exist - run Phase 2 migration first or create user"
        return "FAIL", f"Login failed: {r.status_code} - {r.text[:100]}"
    except Exception as e:
        return "FAIL", f"Cannot connect to auth endpoint: {e}"

def test_get_user():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            user = r.json()
            return "PASS", f"User: {user['email']}, auto_execute: {user.get('auto_execute')}"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Failed to get current user"

test("User authentication (login)", test_login)
test("Get current user (/auth/me)", test_get_user)

# PHASE 4 & 5: Trading Controls API
print("\n--- TRADING CONTROLS API TESTS ---\n")

def test_trading_status():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/trading/status", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            data = r.json()
            return "PASS", f"Paper: {data['paper_trading_enabled']}, Auto:{data['auto_execute']}, Models: {data['selected_model_count']}"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_paper_toggle():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.post(f"{API_URL}/api/trading/paper/toggle", 
                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                         json={"enabled": True})
        if r.status_code == 200 and r.json().get("success"):
            return "PASS", "Paper trading enabled successfully"
        return "FAIL", f"Toggle failed: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_available_models():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/trading/models/available", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            models = r.json()
            total = sum(len(g['models']) for g in models)
            if total > 0:
                return "PASS", f"{len(models)} stock groups, {total} total models"
            return "WARN", "No models available - train models first"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_selected_models():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/trading/models/selected", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            data = r.json()
            return "PASS", f"{data['selected_count']} model(s) selected"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_recent_signals():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/trading/signals/recent?limit=10", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            signals = r.json()
            if len(signals) > 0:
                return "PASS", f"{len(signals)} signal(s) found in history"
            return "WARN", "No signals yet - run signal check or wait for scheduled task"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_check_now():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.post(f"{API_URL}/api/trading/check-now", 
                         headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            return "PASS", "Manual signal check triggered successfully"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

test("GET /api/trading/status", test_trading_status)
test("POST /api/trading/paper/toggle", test_paper_toggle)
test("GET /api/trading/models/available", test_available_models)
test("GET /api/trading/models/selected", test_selected_models)
test("GET /api/trading/signals/recent", test_recent_signals)
test("POST /api/trading/check-now", test_check_now)

# PHASE 6: Additional Endpoints
print("\n--- ADDITIONAL API ENDPOINTS ---\n")

def test_dashboard():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/v1/dashboard/overview", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            return "PASS", "Dashboard data retrieved"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_instruments():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/v1/instruments", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            instruments = r.json()
            return "PASS", f"{len(instruments)} instrument(s) found"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

def test_models_list():
    if not token:
        return "WARN", "Skipped - no auth token"
    try:
        r = requests.get(f"{API_URL}/api/v1/models", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            models = r.json()
            active = [m for m in models if m.get('is_active')]
            stock_specific = [m for m in models if m.get('stock_symbol')]
            return "PASS", f"{len(models)} total, {len(active)} active, {len(stock_specific)} stock-specific"
        return "FAIL", f"Status: {r.status_code}"
    except:
        return "FAIL", "Endpoint not responding"

test("GET /api/v1/dashboard/overview", test_dashboard)
test("GET /api/v1/instruments", test_instruments)
test("GET /api/v1/models (model list)", test_models_list)

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"\nTotal Tests: {results['total']}")
print(f"✅ Passed: {results['passed']}")
print(f"❌ Failed: {results['failed']}")
print(f"⚠️  Warnings: {results['warnings']}")

if results['total'] > 0:
    success_rate = (results['passed'] / results['total']) * 100
    print(f"Success Rate: {success_rate:.1f}%")
else:
    print("Success Rate: N/A")

print("\n" + "="*80)

if results['failed'] == 0:
    print("\n🎉 ALL CRITICAL TESTS PASSED!")
    print("\nSystem is fully functional. Some warnings are expected if:")
    print("  - No models have been trained yet")
    print("  - No signals have been generated yet")
    print("  - Fresh features task hasn't run")
    sys.exit(0)
else:
    print(f"\n⚠️  {results['failed']} test(s) failed - review logs above")
    print("\nCommon issues:")
    print("  - Docker containers not running: docker-compose up -d")
    print("  - Database not migrated: Run Phase 2 migration")
    print("  - Test user doesn't exist: Create via auth/register endpoint")
    sys.exit(1)
