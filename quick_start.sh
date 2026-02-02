#!/bin/bash

# RRIP Quick Start Script
# Downloads sample data, generates pyramids, and starts server

set -e  # Exit on error

echo "================================================"
echo "RRIP Quick Start - Setting up demo environment"
echo "================================================"

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    fi
}

echo "📋 Checking prerequisites..."
check_command python3
check_command cargo

# Check for TurboJPEG
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list jpeg-turbo &>/dev/null; then
        echo "❌ libjpeg-turbo not found. Installing..."
        brew install jpeg-turbo
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! ldconfig -p | grep libturbojpeg &>/dev/null; then
        echo "❌ libturbojpeg not found. Please install: apt-get install libturbojpeg0-dev"
        exit 1
    fi
fi

echo "✅ Prerequisites OK"

# Create data directory
echo ""
echo "📁 Setting up data directory..."
mkdir -p data
cd data

# Download sample if not exists
if [ ! -d "demo_out" ]; then
    echo "📥 Downloading sample data..."
    if [ -f "../download_sample.sh" ]; then
        bash ../download_sample.sh
    else
        echo "⚠️  Sample data not found. Using existing data if available."
    fi
else
    echo "✅ Sample data already exists"
fi

cd ..

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || true

pip install -q --upgrade pip
if [ -f "cli/requirements.txt" ]; then
    pip install -q -r cli/requirements.txt
else
    pip install -q pillow numpy pyvips requests tqdm
fi

# Build server
echo ""
echo "🔨 Building RRIP server..."
cd server

if [ -f "build.sh" ]; then
    ./build.sh --release
else
    # Fallback if build.sh doesn't exist
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export TURBOJPEG_SOURCE=explicit
        export TURBOJPEG_DYNAMIC=1
        export TURBOJPEG_LIB_DIR=$(brew --prefix jpeg-turbo)/lib
        export TURBOJPEG_INCLUDE_DIR=$(brew --prefix jpeg-turbo)/include
    fi
    cargo build --release
fi

echo ""
echo "🚀 Starting RRIP server..."
echo "================================================"
echo ""

# Kill any existing servers on port 3007
lsof -ti:3007 | xargs -r kill 2>/dev/null || true

# Start server
if [ -f "run-server.sh" ]; then
    ./run-server.sh --slides-root ../data --port 3007 &
else
    # Fallback
    ./target/release/origami-tile-server --slides-root ../data --port 3007 &
fi

SERVER_PID=$!
echo "✅ Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "⏳ Waiting for server to initialize..."
sleep 3

# Test server
if curl -s http://localhost:3007/healthz > /dev/null; then
    echo "✅ Server is healthy"
else
    echo "❌ Server failed to start"
    exit 1
fi

echo ""
echo "================================================"
echo "✨ RRIP is ready!"
echo "================================================"
echo ""
echo "🌐 Viewer URLs:"
echo "   RRIP Viewer:     http://localhost:3007/viewer/demo_out"
echo "   Health Check:    http://localhost:3007/healthz"
echo "   Metrics:         http://localhost:3007/metrics"
echo ""
echo "📚 API Endpoints:"
echo "   Tile:     http://localhost:3007/tiles/demo_out/{z}/{x}_{y}.jpg"
echo "   Manifest: http://localhost:3007/dzi/demo_out.dzi"
echo ""
echo "🛑 To stop the server:"
echo "   kill $SERVER_PID"
echo ""
echo "📖 For more options, see README.md"
echo "================================================"

# Keep script running
echo ""
echo "Press Ctrl+C to stop the server..."
trap "kill $SERVER_PID 2>/dev/null; echo 'Server stopped.'" EXIT
wait $SERVER_PID