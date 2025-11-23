#!/bin/bash
# Script to download Kaggle dataset

set -e

echo "Downloading Kaggle Stock Market News Dataset..."

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: kaggle.json not found in ~/.kaggle/"
    echo "Please download your API credentials from https://www.kaggle.com/account"
    exit 1
fi

# Create data directory
mkdir -p data/raw

# Download dataset
cd data/raw
kaggle datasets download -d aaron7sun/stocknews

# Unzip
if [ -f stocknews.zip ]; then
    unzip stocknews.zip
    echo "Dataset downloaded and extracted successfully!"
    echo "Files in data/raw/:"
    ls -lh
else
    echo "Error: Download failed"
    exit 1
fi

