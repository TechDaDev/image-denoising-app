#!/bin/bash
# Simple script to run the Streamlit app with the virtual environment

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found!"
    echo "Please create it with: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run the app
echo "🚀 Starting Streamlit app with virtual environment..."
source venv/bin/activate

echo "📦 Using Python: $(which python)"
echo "📦 Python version: $(python --version)"
echo "📦 Streamlit version: $(streamlit --version)"

echo ""
echo "🌐 Starting Streamlit app..."
echo "   Local URL will be displayed below"
echo "   Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py