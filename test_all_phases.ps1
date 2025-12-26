"""
Comprehensive System Testing - All Phases
Runs direct tests via PowerShell commands
"""

# Test Results Log
$TestResults = @()
$TestNumber = 0

function Log-Test {
    param(
        [string]$TestName,
        [string]$Status,
        [string]$Details
    )
    $global:TestNumber++
    $timestamp = Get-Date -Format "HH:mm:ss"
    $result = [PSCustomObject]@{
        Number  = $TestNumber
        Time    = $timestamp
        Test    = $TestName
        Status  = $Status
        Details = $Details
    }
    $global:TestResults += $result
    
    $statusSymbol = if ($Status -eq "PASS") { "✅" } else { "❌" }
    Write-Host "[$timestamp] TEST $TestNumber - $statusSymbol $TestName"
    Write-Host "    $Details`n"
}

Write-Host "="*80
Write-Host "COMPREHENSIVE SYSTEM TESTING - ALL PHASES"
Write-Host "="*80
Write-Host ""

# PHASE 1: Database Schema Tests
Write-Host "`n--- PHASE 2: DATABASE SCHEMA TESTS ---`n"

# Test 1: models.stock_symbol exists
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT column_name FROM information_schema.columns WHERE table_name='models' AND column_name='stock_symbol';"
    if ($result -like "*stock_symbol*") {
        Log-Test "models.stock_symbol column exists" "PASS" "Column confirmed in database schema"
    }
    else {
        Log-Test "models.stock_symbol column exists" "FAIL" "Column not found"
    }
}
catch {
    Log-Test "models.stock_symbol column exists" "FAIL" $_.Exception.Message
}

# Test 2: users.selected_model_ids exists
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='users' AND column_name='selected_model_ids';"
    if ($result -like "*selected_model_ids*" -and $result -like "*jsonb*") {
        Log-Test "users.selected_model_ids (JSONB) column exists" "PASS" "JSONB column confirmed"
    }
    else {
        Log-Test "users.selected_model_ids (JSONB) column exists" "FAIL" "Column not found or wrong type"
    }
}
catch {
    Log-Test "users.selected_model_ids (JSONB) column exists" "FAIL" $_.Exception.Message
}

# Test 3: users.paper_trading_enabled exists
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='paper_trading_enabled';"
    if ($result -like "*paper_trading_enabled*") {
        Log-Test "users.paper_trading_enabled column exists" "PASS" "Boolean column confirmed"
    }
    else {
        Log-Test "users.paper_trading_enabled column exists" "FAIL" "Column not found"
    }
}
catch {
    Log-Test "users.paper_trading_enabled column exists" "FAIL" $_.Exception.Message
}

# Test 4: signal_logs table exists
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='signal_logs';"
    if ($result -like "*1*") {
        Log-Test "signal_logs table exists" "PASS" "Table confirmed in database"
    }
    else {
        Log-Test "signal_logs table exists" "FAIL" "Table not found"
    }
}
catch {
    Log-Test "signal_logs table exists" "FAIL" $_.Exception.Message
}

# Test 5: signal_logs indexes
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM pg_indexes WHERE tablename='signal_logs';"
    $count = [int]($result.Trim())
    if ($count -ge 4) {
        Log-Test "signal_logs has required indexes" "PASS" "$count indexes found (expected >= 4)"
    }
    else {
        Log-Test "signal_logs has required indexes" "FAIL" "Only $count indexes found"
    }
}
catch {
    Log-Test "signal_logs has required indexes" "FAIL" $_.Exception.Message
}

# PHASE 3: API Health Tests
Write-Host "`n--- API HEALTH TESTS ---`n"

# Test 6: Backend health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    $content = $response.Content | ConvertFrom-Json
    if ($response.StatusCode -eq 200 -and $content.status -eq "healthy") {
        Log-Test "Backend health endpoint" "PASS" "Status: healthy, Service: $($content.service)"
    }
    else {
        Log-Test "Backend health endpoint" "FAIL" "Unexpected response: $($response.StatusCode)"
    }
}
catch {
    Log-Test "Backend health endpoint" "FAIL" $_.Exception.Message
}

# Test 7: API docs accessible
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/docs" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Log-Test "API documentation accessible" "PASS" "Swagger UI responding at /docs"
    }
    else {
        Log-Test "API documentation accessible" "FAIL" "Status code: $($response.StatusCode)"
    }
}
catch {
    Log-Test "API documentation accessible" "FAIL" $_.Exception.Message
}

# PHASE 4: Authentication Tests
Write-Host "`n--- AUTHENTICATION TESTS ---`n"

# Test 8: Create test user for API testing
try {
    $result = docker exec jarvistrade_backend python -c @"
from app.db.database import SessionLocal
from app.db.models import User
from passlib.context import CryptContext
import hashlib, base64

pwd_context = CryptContext(schemes=['bcrypt'])

def prepare_password(password):
    password_hash = hashlib.sha256(password.encode('utf-8')).digest()
    return base64.b64encode(password_hash).decode('ascii')

db = SessionLocal()
# Try to find existing test user
user = db.query(User).filter(User.email == 'test@example.com').first()
if not user:
    user = User(
        email='test@example.com',
        password_hash=pwd_context.hash(prepare_password('testpass123')),
        auto_execute=True,
        paper_trading_enabled=True
    )
    db.add(user)
    db.commit()
    print('Created new test user')
else:
    print('Test user already exists')
print(user.email)
"@
    if ($result -like "*test@example.com*") {
        Log-Test "Test user creation/verification" "PASS" "User: test@example.com ready for testing"
    }
    else {
        Log-Test "Test user creation/verification" "FAIL" "Could not create/find user"
    }
}
catch {
    Log-Test "Test user creation/verification" "FAIL" $_.Exception.Message
}

# Test 9: Login and get JWT token
$global:Token = $null
try {
    $body = @{
        username = "test@example.com"
        password = "testpass123"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/auth/login" -Method Post -Body $body -UseBasicParsing
    $json = $response.Content | ConvertFrom-Json
    $global:Token = $json.access_token
    if ($Token) {
        Log-Test "User authentication (JWT)" "PASS" "Token obtained: $($Token.Substring(0,20))..."
    }
    else {
        Log-Test "User authentication (JWT)" "FAIL" "No token in response"
    }
}
catch {
    Log-Test "User authentication (JWT)" "FAIL" $_.Exception.Message
}

# Test 10: Get current user info
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/auth/me" -Headers $headers -UseBasicParsing
    $user = $response.Content | ConvertFrom-Json
    if ($user.email -eq "test@example.com") {
        Log-Test "Get current user info (/auth/me)" "PASS" "User: $($user.email), auto_execute: $($user.auto_execute)"
    }
    else {
        Log-Test "Get current user info (/auth/me)" "FAIL" "Unexpected user data"
    }
}
catch {
    Log-Test "Get current user info (/auth/me)" "FAIL" $_.Exception.Message
}

# PHASE 5: Trading Controls API Tests
Write-Host "`n--- TRADING CONTROLS API TESTS ---`n"

# Test 11: Get trading status
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/trading/status" -Headers $headers -UseBasicParsing
    $status = $response.Content | ConvertFrom-Json
    Log-Test "GET /api/trading/status" "PASS" "Paper Trading: $($status.paper_trading_enabled), Models: $($status.selected_model_count)"
}
catch {
    Log-Test "GET /api/trading/status" "FAIL" $_.Exception.Message
}

# Test 12: Toggle paper trading ON
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
        "Content-Type"  = "application/json"
    }
    $body = '{"enabled": true}'
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/trading/paper/toggle" -Method Post -Headers $headers -Body $body -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    if ($result.success -and $result.paper_trading_enabled) {
        Log-Test "POST /api/trading/paper/toggle (enable)" "PASS" "Paper trading enabled successfully"
    }
    else {
        Log-Test "POST /api/trading/paper/toggle (enable)" "FAIL" "Toggle failed"
    }
}
catch {
    Log-Test "POST /api/trading/paper/toggle (enable)" "FAIL" $_.Exception.Message
}

# Test 13: Get available models
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/trading/models/available" -Headers $headers -UseBasicParsing
    $models = $response.Content | ConvertFrom-Json
    $totalModels = ($models | ForEach-Object { $_.models.Count } | Measure-Object -Sum).Sum
    Log-Test "GET /api/trading/models/available" "PASS" "$($models.Count) stock groups, $totalModels total models"
}
catch {
    Log-Test "GET /api/trading/models/available" "FAIL" $_.Exception.Message
}

# Test 14: Get selected models
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/trading/models/selected" -Headers $headers -UseBasicParsing
    $selected = $response.Content | ConvertFrom-Json
    Log-Test "GET /api/trading/models/selected" "PASS" "Selected: $($selected.selected_count) models"
}
catch {
    Log-Test "GET /api/trading/models/selected" "FAIL" $_.Exception.Message
}

# Test 15: Get recent signals
try {
    $headers = @{
        "Authorization" = "Bearer $Token"
    }
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/trading/signals/recent?limit=10" -Headers $headers -UseBasicParsing
    $signals = $response.Content | ConvertFrom-Json
    Log-Test "GET /api/trading/signals/recent" "PASS" "Retrieved $($signals.Count) signal(s)"
}
catch {
    Log-Test "GET /api/trading/signals/recent" "FAIL" $_.Exception.Message
}

# PHASE 6: Data Tests
Write-Host "`n--- DATA & FEATURE TESTS ---`n"

# Test 16: Check instruments exist
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM instruments;"
    $count = [int]($result.Trim())
    if ($count -gt 0) {
        Log-Test "Instruments data exists" "PASS" "$count instrument(s) in database"
    }
    else {
        Log-Test "Instruments data exists" "FAIL" "No instruments found"
    }
}
catch {
    Log-Test "Instruments data exists" "FAIL" $_.Exception.Message
}

# Test 17: Check features exist
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM features;"
    $count = [int]($result.Trim())
    if ($count -gt 0) {
        Log-Test "Features data exists" "PASS" "$count feature record(s) in database"
    }
    else {
        Log-Test "Features data exists" "WARN" "No features yet (will be computed by fresh_features task)"
    }
}
catch {
    Log-Test "Features data exists" "FAIL" $_.Exception.Message
}

# Test 18: Check models exist
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM models;"
    $count = [int]($result.Trim())
    if ($count -gt 0) {
        Log-Test "Models exist in database" "PASS" "$count model(s) in database"
    }
    else {
        Log-Test "Models exist in database" "WARN" "No models yet (train models first)"
    }
}
catch {
    Log-Test "Models exist in database" "FAIL" $_.Exception.Message
}

# Test 19: Check if any models have stock_symbol
try {
    $result = docker exec jarvistrade_db psql -U postgres -d jarvistrade -t -c "SELECT COUNT(*) FROM models WHERE stock_symbol IS NOT NULL;"
    $count = [int]($result.Trim())
    if ($count -gt 0) {
        Log-Test "Models with stock_symbol (Phase 3)" "PASS" "$count model(s) with stock_symbol"
    }
    else {
        Log-Test "Models with stock_symbol (Phase 3)" "WARN" "No stock-specific models yet (retrain with stock_symbol parameter)"
    }
}
catch {
    Log-Test "Models with stock_symbol (Phase 3)" "FAIL" $_.Exception.Message
}

# PHASE 7: Celery Tests
Write-Host "`n--- CELERY WORKER TESTS ---`n"

# Test 20: Check Celery worker status
try {
    $result = docker exec jarvistrade_backend celery -A app.celery_app inspect active 2>&1
    if ($result -notlike "*Error*" -and $result -notlike "*failed*") {
        Log-Test "Celery worker running" "PASS" "Worker responding to inspect command"
    }
    else {
        Log-Test "Celery worker running" "FAIL" "Worker not responding or error"
    }
}
catch {
    Log-Test "Celery worker running" "FAIL" $_.Exception.Message
}

# Test 21: Check Celery Beat scheduler
try {
    $result = docker ps --filter "name=celery" --format "{{.Names}}\t{{.Status}}"
    if ($result -like "*celery*" -and $result -like "*Up*") {
        Log-Test "Celery Beat scheduler running" "PASS" "Beat container is up"
    }
    else {
        Log-Test "Celery Beat scheduler running" "FAIL" "Beat container not found or down"
    }
}
catch {
    Log-Test "Celery Beat scheduler running" "FAIL" $_.Exception.Message
}

# Summary
Write-Host "`n"
Write-Host "="*80
Write-Host "TEST SUMMARY"
Write-Host "="*80

$passed = ($TestResults | Where-Object { $_.Status -eq "PASS" }).Count
$failed = ($TestResults | Where-Object { $_.Status -eq "FAIL" }).Count
$warned = ($TestResults | Where-Object { $_.Status -eq "WARN" }).Count
$total = $TestResults.Count

Write-Host "`nTotal Tests: $total"
Write-Host "✅ Passed: $passed"
Write-Host "❌ Failed: $failed"
Write-Host "⚠️  Warnings: $warned"
Write-Host "Success Rate: $([math]::Round(($passed/$total)*100, 1))%"

# Detailed results
Write-Host "`n`nDETAILED RESULTS:`n"
$TestResults | Format-Table -AutoSize

# Export results
$TestResults | Export-Csv -Path "test_results.csv" -NoTypeInformation
Write-Host "`nResults exported to: test_results.csv"

if ($failed -eq 0) {
    Write-Host "`n🎉 ALL CRITICAL TESTS PASSED!"
    exit 0
}
else {
    Write-Host "`n⚠️  $failed test(s) failed - review above"
    exit 1
}
