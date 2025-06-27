#!/bin/bash
echo "Building Paper Summarizer for Render..."

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Install dependencies
pip install -r requirements.txt

echo "Build complete!"