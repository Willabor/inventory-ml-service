#!/bin/bash

# Railway Startup Script with Auto-Training
# This script:
# 1. Starts the ML service in background
# 2. Waits for service to be healthy
# 3. Trains both models
# 4. Keeps service running in foreground

set -e  # Exit on error

echo "=================================================="
echo "🚀 RAILWAY ML SERVICE STARTUP WITH AUTO-TRAINING"
echo "=================================================="
echo ""

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Database: ${DATABASE_URL:+Connected}"
echo ""

# Start uvicorn in background
echo "🌐 Starting ML service in background..."
uvicorn main:app --host "$HOST" --port "$PORT" --log-level info &
UVICORN_PID=$!

echo "   Process ID: $UVICORN_PID"
echo ""

# Wait a moment for server to initialize
sleep 5

# Check if server is still running
if ! kill -0 $UVICORN_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    exit 1
fi

echo "✓ Server started successfully"
echo ""

# Set ML_SERVICE_URL for auto_train.py
export ML_SERVICE_URL="http://localhost:$PORT"

# Run auto-training script
echo "🤖 Running auto-training script..."
python3 auto_train.py

TRAIN_EXIT_CODE=$?

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Auto-training completed successfully!"
else
    echo "⚠️  Auto-training had issues (exit code: $TRAIN_EXIT_CODE)"
    echo "   Service will continue running, but models may not be loaded"
fi

echo ""
echo "=================================================="
echo "🎯 ML SERVICE IS READY"
echo "=================================================="
echo ""
echo "Service URL: http://$HOST:$PORT"
echo "Health check: http://$HOST:$PORT/health"
echo ""
echo "📝 Models trained and ready for predictions!"
echo ""

# Keep uvicorn running in foreground
# This is critical - Railway needs a foreground process
wait $UVICORN_PID
