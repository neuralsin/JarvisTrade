# Phase 2 Migration Runner (Windows PowerShell)
# 
# This script runs the Phase 2 migration inside the Docker container
# Usage: .\run_migration.ps1 [up|down]

param(
    [Parameter(Position=0)]
    [ValidateSet('up', 'down')]
    [string]$Action = 'up'
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Phase 2 Migration ($Action)" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker ps | Out-Null
} catch {
    Write-Host "❌ Error: Docker is not running" -ForegroundColor Red
    exit 1
}

# Find the backend container
$containerName = "jarvistrade-backend-1"
$containers = docker ps --format "{{.Names}}"

if ($containers -notcontains $containerName) {
    # Try alternative naming
    $containerName = $containers | Where-Object { $_ -like "*backend*" } | Select-Object -First 1
    if (-not $containerName) {
        Write-Host "❌ Error: Backend container not found" -ForegroundColor Red
        Write-Host "Available containers:" -ForegroundColor Yellow
        docker ps --format "{{.Names}}"
        exit 1
    }
}

Write-Host "📦 Using container: $containerName" -ForegroundColor Green
Write-Host ""

# Run migration
if ($Action -eq 'up') {
    Write-Host "🚀 Applying Phase 2 migration..." -ForegroundColor Yellow
    docker exec -it $containerName python migrations/phase2_schema.py up
} else {
    Write-Host "⏪ Rolling back Phase 2 migration..." -ForegroundColor Yellow
    docker exec -it $containerName python migrations/phase2_schema.py down
}

Write-Host ""
Write-Host "✅ Migration complete!" -ForegroundColor Green
