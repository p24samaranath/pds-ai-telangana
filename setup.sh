#!/bin/bash
# PDS Optimization System — Quick Setup Script
set -e

echo ""
echo "🌾 PDS AI Optimization System — Setup"
echo "======================================="
echo ""

# ── Fix PATH so Homebrew node/npm are found ───────────────────────────────────
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.9+"
    exit 1
fi
echo "✅ Python $(python3 --version)"

# Check Node / npm
if command -v npm &> /dev/null; then
    echo "✅ Node $(node --version)  npm $(npm --version)"
    HAVE_NODE=1
else
    echo "⚠️  Node.js not found — installing via Homebrew…"
    brew install node && HAVE_NODE=1
fi

# Create .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
    echo "   ⚠️  Set your ANTHROPIC_API_KEY in .env to enable AI features"
fi

# Backend setup
echo ""
echo "📦 Installing backend dependencies…"
cd backend
rm -rf venv          # always start clean so torch version never conflicts
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
mkdir -p logs data/raw data/processed data/models
echo "✅ Backend ready  (Python $(python --version), PyTorch $(python -c 'import torch; print(torch.__version__)'))"
cd ..

# Frontend setup
if [ -n "$HAVE_NODE" ]; then
    echo ""
    echo "📦 Installing frontend dependencies…"
    cd frontend
    npm install --silent
    echo "✅ Frontend ready"
    cd ..
fi

echo ""
echo "🚀 Setup complete!  Run in two separate terminals:"
echo ""
echo "  Terminal 1 — Backend:"
echo "    cd backend && source venv/bin/activate && PYTHONPATH=. python -m uvicorn app.main:app --reload"
echo ""
echo "  Terminal 2 — Frontend:"
echo "    cd frontend && npm run dev"
echo ""
echo "  Dashboard → http://localhost:3000"
echo "  API docs  → http://localhost:8000/docs"
echo ""
