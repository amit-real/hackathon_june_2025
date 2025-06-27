#!/bin/bash
echo "Starting Research Paper Summarizer Web UI..."
echo "========================================"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set!"
    echo "Set it using: export GEMINI_API_KEY='your-key-here'"
    echo ""
fi

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the web app
echo "Starting server on http://localhost:5000"
python3 app_web.py