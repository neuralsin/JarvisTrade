#!/bin/bash
# Docker Migration Runner for Phase 2
# 
# This script runs the Phase 2 migration inside the Docker container
# Usage: ./run_migration.sh [up|down]

set -e

ACTION=${1:-up}

if [ "$ACTION" != "up" ] && [ "$ACTION" != "down" ]; then
    echo "Usage: ./run_migration.sh [up|down]"
    exit 1
fi

echo "=================================="
echo "Phase 2 Migration ($ACTION)"
echo "=================================="

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Find the backend container
CONTAINER_NAME="jarvistrade-backend-1"
if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    # Try alternative naming
    CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep backend | head -1)
    if [ -z "$CONTAINER_NAME" ]; then
        echo "❌ Error: Backend container not found"
        echo "Available containers:"
        docker ps --format '{{.Names}}'
        exit 1
    fi
fi

echo "📦 Using container: $CONTAINER_NAME"
echo ""

# Run migration
if [ "$ACTION" = "up" ]; then
    echo "🚀 Applying Phase 2 migration..."
    docker exec -it "$CONTAINER_NAME" python migrations/phase2_schema.py up
else
    echo "⏪ Rolling back Phase 2 migration..."
    docker exec -it "$CONTAINER_NAME" python migrations/phase2_schema.py down
fi

echo ""
echo "✅ Migration complete!"
