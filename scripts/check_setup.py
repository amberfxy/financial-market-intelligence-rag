#!/usr/bin/env python3
"""检查项目设置是否完整"""

import os
import sys
from pathlib import Path

def check_file(path, name):
    """检查文件是否存在"""
    if os.path.exists(path):
        print(f"✅ {name}: 存在")
        return True
    else:
        print(f"❌ {name}: 不存在 ({path})")
        return False

def check_dir(path, name):
    """检查目录是否存在且有内容"""
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        if files:
            print(f"✅ {name}: 存在 ({len(files)} 个文件)")
            return True
        else:
            print(f"⚠️  {name}: 目录存在但为空 ({path})")
            return False
    else:
        print(f"❌ {name}: 不存在 ({path})")
        return False

def main():
    print("=" * 60)
    print("项目设置检查")
    print("=" * 60)
    
    issues = []
    
    # 检查核心文件
    print("\n📁 核心文件:")
    check_file("ui/app.py", "Streamlit应用")
    check_file("src/rag/pipeline.py", "RAG管道")
    check_file("scripts/build_index.py", "索引构建脚本")
    check_file("requirements.txt", "依赖文件")
    
    # 检查数据
    print("\n📊 数据:")
    data_ok = check_dir("data/raw", "原始数据")
    if not data_ok:
        issues.append("需要下载数据集到 data/raw/")
    
    # 检查模型
    print("\n🤖 模型:")
    model_files = []
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.gguf')]
    
    if model_files:
        print(f"✅ 找到 {len(model_files)} 个模型文件")
        for f in model_files:
            size = os.path.getsize(f"models/{f}") / (1024**3)  # GB
            print(f"   - {f} ({size:.2f} GB)")
    else:
        print("❌ 未找到模型文件 (.gguf)")
        issues.append("需要下载Mistral 7B模型到 models/")
    
    # 检查索引
    print("\n🔍 向量索引:")
    index_ok = check_file("vectorstore/faiss.index", "FAISS索引")
    chunks_ok = check_file("vectorstore/chunks.pkl", "分块数据")
    
    if not (index_ok and chunks_ok):
        issues.append("需要运行 scripts/build_index.py 构建索引")
    
    # 检查依赖
    print("\n📦 Python依赖:")
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit - 需要安装")
        issues.append("运行: pip install -r requirements.txt")
    
    try:
        import torch
        print("✅ torch")
    except ImportError:
        print("❌ torch - 需要安装")
        issues.append("运行: pip install -r requirements.txt")
    
    try:
        import transformers
        print("✅ transformers")
    except ImportError:
        print("❌ transformers - 需要安装")
        issues.append("运行: pip install -r requirements.txt")
    
    try:
        import faiss
        print("✅ faiss")
    except ImportError:
        print("❌ faiss - 需要安装")
        issues.append("运行: pip install faiss-cpu")
    
    try:
        import llama_cpp
        print("✅ llama-cpp-python")
    except ImportError:
        print("❌ llama-cpp-python - 需要安装")
        issues.append("运行: pip install llama-cpp-python")
    
    # 总结
    print("\n" + "=" * 60)
    if issues:
        print("⚠️  需要完成的步骤:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n详细步骤请查看 QUICKSTART.md")
        return False
    else:
        print("✅ 所有设置完成！可以运行项目了")
        print("\n运行命令:")
        print("  streamlit run ui/app.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

