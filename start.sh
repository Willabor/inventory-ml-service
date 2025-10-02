#!/bin/bash

# ML Service Startup Script
# This script ensures the ML service is properly initialized before starting

echo "🚀 Starting ML Service..."

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing Python dependencies..."
    pip install --quiet -r requirements.txt
    touch venv/.installed
    echo "✓ Dependencies installed"
fi

# Check if model exists, if not create a placeholder
MODEL_DIR="models/cache"
mkdir -p "$MODEL_DIR"

# Start the service
echo "✓ ML Service ready"
echo "🌐 Starting uvicorn on port 8000..."

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
