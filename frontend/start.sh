#!/bin/bash

echo "🚀 Starting Ollama API Gateway Frontend..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

echo "✨ Starting development server..."
npm run dev
