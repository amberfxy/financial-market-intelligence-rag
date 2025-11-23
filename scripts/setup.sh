#!/bin/bash
# Complete setup script for Financial Market Intelligence RAG System

set -e

echo "🚀 Setting up Financial Market Intelligence RAG System..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed vectorstore models

# Download dataset
echo ""
echo "📥 Downloading dataset..."
if [ -f ~/.kaggle/kaggle.json ]; then
    ./scripts/download_data.sh
else
    echo "⚠️  Kaggle credentials not found. Please download dataset manually:"
    echo "   cd data/raw && kaggle datasets download -d aaron7sun/stocknews"
fi

# Download model
echo ""
echo "🤖 Downloading LLM model..."
if [ ! -f models/mistral-7b-instruct-v0.1.Q4_K_M.gguf ]; then
    echo "Please download the model manually:"
    echo "  cd models"
    echo "  wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
else
    echo "✅ Model already exists"
fi

# Build index
echo ""
echo "🔨 Building FAISS index..."
if [ -f data/raw/*.csv ]; then
    python3 scripts/build_index.py
else
    echo "⚠️  No data found. Please download dataset first."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  streamlit run ui/app.py"

