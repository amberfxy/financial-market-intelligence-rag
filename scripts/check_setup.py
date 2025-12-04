#!/usr/bin/env python3
"""Check if project setup is complete."""

import os
import sys
from pathlib import Path

def check_file(path, name):
    """Check if file exists."""
    if os.path.exists(path):
        print(f"[OK] {name}: exists")
        return True
    else:
        print(f"[MISSING] {name}: not found ({path})")
        return False

def check_dir(path, name):
    """Check if directory exists and has content."""
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        if files:
            print(f"[OK] {name}: exists ({len(files)} files)")
            return True
        else:
            print(f"[WARNING] {name}: directory exists but is empty ({path})")
            return False
    else:
        print(f"[MISSING] {name}: not found ({path})")
        return False

def main():
    print("=" * 60)
    print("Project Setup Check")
    print("=" * 60)
    
    issues = []
    
    # Check core files
    print("\nCore Files:")
    check_file("ui/app.py", "Streamlit app")
    check_file("src/rag/pipeline.py", "RAG pipeline")
    check_file("scripts/build_index.py", "Index building script")
    check_file("requirements.txt", "Dependencies file")
    
    # Check data
    print("\nData:")
    data_ok = check_dir("data/raw", "Raw data")
    if not data_ok:
        issues.append("Download dataset to data/raw/")
    
    # Check models
    print("\nModels:")
    model_files = []
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.gguf')]
    
    if model_files:
        print(f"[OK] Found {len(model_files)} model file(s)")
        for f in model_files:
            size = os.path.getsize(f"models/{f}") / (1024**3)  # GB
            print(f"   - {f} ({size:.2f} GB)")
    else:
        print("[MISSING] No model files found (.gguf)")
        issues.append("Download Mistral 7B model to models/")
    
    # Check index
    print("\nVector Index:")
    index_ok = check_file("vectorstore/faiss.index", "FAISS index")
    chunks_ok = check_file("vectorstore/chunks.pkl", "Chunks data")
    
    if not (index_ok and chunks_ok):
        issues.append("Run scripts/build_index.py to build index")
    
    # Check dependencies
    print("\nPython Dependencies:")
    try:
        import streamlit
        print("[OK] streamlit")
    except ImportError:
        print("[MISSING] streamlit - needs installation")
        issues.append("Run: pip install -r requirements.txt")
    
    try:
        import torch
        print("[OK] torch")
    except ImportError:
        print("[MISSING] torch - needs installation")
        issues.append("Run: pip install -r requirements.txt")
    
    try:
        import transformers
        print("[OK] transformers")
    except ImportError:
        print("[MISSING] transformers - needs installation")
        issues.append("Run: pip install -r requirements.txt")
    
    try:
        import faiss
        print("[OK] faiss")
    except ImportError:
        print("[MISSING] faiss - needs installation")
        issues.append("Run: pip install faiss-cpu")
    
    try:
        import llama_cpp
        print("[OK] llama-cpp-python")
    except ImportError:
        print("[MISSING] llama-cpp-python - needs installation")
        issues.append("Run: pip install llama-cpp-python")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("Steps to complete:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nSee README.md for detailed instructions")
        return False
    else:
        print("All setup complete! Ready to run the project.")
        print("\nRun command:")
        print("  streamlit run ui/app.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

