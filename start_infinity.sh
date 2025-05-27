#!/bin/bash

# INFINITY REASONER - Quick Start Script
echo "🚀 Starting Infinity Reasoner System..."
echo ""
echo "🌟 INFINITY: Self-Evolving AI Reasoner"
echo "🛡️  Enhanced with Security Monitoring"
echo "⚠️  Requires human oversight and monitoring"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if setup has been run
if [ ! -f "requirements.txt" ]; then
    echo "📦 Setting up Infinity for first time..."
    chmod +x setup.sh
    ./setup.sh
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

echo "🎯 Starting Infinity evolution process..."
echo "📊 Monitor progress and approve tasks as prompted"
echo "🛑 Press Ctrl+C to stop at any time"
echo ""

# Start the system
python3 infinity_reasoner.py

echo ""
echo "🎉 Infinity session complete!"
echo "📊 Check logs for detailed performance metrics"
